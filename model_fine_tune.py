import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from huggingface_hub import login

import mlflow
import mlflow.transformers

from dotenv import load_dotenv, find_dotenv


warnings.filterwarnings('ignore')


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdvancedTrainer(Trainer):
    def __init__(self, loss_type="focal_weighted", class_weights=None, focal_gamma=2.0, loss_reduction="mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        
        if loss_type == "focal_weighted":
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction=loss_reduction)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma, reduction=loss_reduction)
        elif loss_type == "ce_weighted":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



def detect_device():
    """Detect best available device with fallback strategy"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
    else:
        device = "cpu"
    return device


def preprocess_function(examples, tokenizer, label_column, text_field, max_length):
    """Tokenize text and encode numeric labels directly"""
    tokens = tokenizer(
        examples[text_field],
        truncation=True,
        padding=False,  # Will pad later in DataCollator
        max_length=max_length,
        return_tensors=None
    )
    # MNLI labels are integers 0/1/2
    tokens["labels"] = [int(l) for l in examples[label_column]]
    return tokens


def calculate_class_weights(train_labels, max_weight, min_weight):
    label_counts = pd.Series(train_labels).value_counts().sort_index()

    class_weights = [1 / x for x in label_counts.tolist()]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Scale weights to MAX_WEIGHT and MIN_WEIGHT
    curr_max_weight = class_weights_tensor.max()
    curr_min_weight = class_weights_tensor.min()
    class_weights_tensor = (class_weights_tensor - curr_min_weight) / (curr_max_weight - curr_min_weight) * (max_weight - min_weight) + min_weight

    return class_weights_tensor


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def comprehensive_evaluation(trainer, tokenized_datasets, hyperparams, results_base_dir="results", total_time_seconds=None):
    result_dir = f"{results_base_dir}/{hyperparams['experiment_name']}"
    os.makedirs(result_dir, exist_ok=True)

    with open(f"{result_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)

    predictions = trainer.predict(tokenized_datasets["validation"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Overall metrics
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'macro_f1': round(f1_score(y_true, y_pred, average='macro'), 4),
        'weighted_f1': round(f1_score(y_true, y_pred, average='weighted'), 4),
    }
    
    # Per-class metrics
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    class_report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=[str(l) for l in all_labels],
        output_dict=True
    )
    
    def round_nested_dict(d, decimals=4):
        """Recursively round all numeric values in nested dictionary"""
        if isinstance(d, dict):
            return {k: round_nested_dict(v, decimals) for k, v in d.items()}
        elif isinstance(d, (int, float)):
            return round(d, decimals) if isinstance(d, float) else d
        else:
            return d
    
    class_report = round_nested_dict(class_report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Training progress extraction
    training_progress = []
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
        train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
        train_loss_map = {}
        for log in train_logs:
            step = log.get('step', 0)
            if step > 0:
                train_loss_map[step] = log.get('loss', 0)
        
        for log in eval_logs:
            eval_step = log.get('step', 0)
            training_loss = 0
            if train_loss_map:
                valid_train_steps = [s for s in train_loss_map.keys() if s <= eval_step]
                if valid_train_steps:
                    closest_train_step = max(valid_train_steps)
                    training_loss = train_loss_map[closest_train_step]
            progress_entry = {
                'step': eval_step,
                'training_loss': round(training_loss, 4),
                'validation_loss': round(log.get('eval_loss', 0), 4),
                'accuracy': round(log.get('eval_accuracy', 0), 4),
                'macro_f1': round(log.get('eval_macro_f1', 0), 4),
                'micro_f1': round(log.get('eval_micro_f1', 0), 4),
                'weighted_f1': round(log.get('eval_weighted_f1', 0), 4)
            }
            training_progress.append(progress_entry)
    
    # Detailed class analysis
    detailed_analysis = {}
    for idx, label in enumerate(all_labels):
        class_name = str(label)
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        true_count = (y_true == label).sum()
        pred_count = (y_pred == label).sum()
        true_pct = (true_count / len(y_true)) * 100
        pred_pct = (pred_count / len(y_pred)) * 100
        detailed_analysis[class_name] = {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
            'true_pct': round(true_pct, 4), 'pred_pct': round(pred_pct, 4), 
            'diff_pct': round(pred_pct - true_pct, 4)
        }
        
    # Class distribution analysis
    true_dist = pd.Series(y_true).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    
    # Save comprehensive results
    results = {
        'execution_time': {
            'total_seconds': round(total_time_seconds, 2) if total_time_seconds is not None else None,
            'formatted_time': f"{int(total_time_seconds // 3600):02d}:{int((total_time_seconds % 3600) // 60):02d}:{int(total_time_seconds % 60):02d}" if total_time_seconds is not None else None
        },
        'overall_metrics': metrics,
        'per_class_metrics': class_report,
        'detailed_class_analysis': detailed_analysis,
        'training_progress': training_progress,
        'confusion_matrix': cm.tolist(),
        'true_distribution': true_dist.to_dict(),
        'pred_distribution': pred_dist.to_dict(),
    }
    with open(f"{result_dir}/evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    return results


def build_contextual_dataset(dataset):
    """Build a DatasetDict with 'text_concat' per split using only current example.
    For MNLI-style data, 'text_concat' = '[PREMISE] {premise} [HYPOTHESIS] {hypothesis}'.
    """
    new_splits = {}
    for split_name, split in dataset.items():
        df = split.to_pandas().reset_index(drop=True)
        if 'premise' in df.columns and 'hypothesis' in df.columns:
            df['text_concat'] = '[PREMISE] ' + df['premise'].astype(str) + ' [HYPOTHESIS] ' + df['hypothesis'].astype(str)
        else:
            raise ValueError("Expected 'premise' and 'hypothesis' columns in the dataset")
        new_splits[split_name] = datasets.Dataset.from_pandas(df, preserve_index=False)

    return datasets.DatasetDict(new_splits)


def fine_tune_model(params, results_base_dir="results", checkpoints_base_dir="checkpoints"):
    start_time = time.time()
    device = detect_device()
    load_dotenv(find_dotenv())

    # HF login
    try:
        login(token=os.environ.get("HUGGINGFACE_HUB_TOKEN"), add_to_git_credential=False)
    except Exception as e:
        warnings.warn(f"Hugging Face login failed: {e}")

    # MLflow setup
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(params.get("logging_experiment_name"))
    mlflow.transformers.autolog(log_models=True)

    # Load MNLI
    dataset = load_dataset("nyu-mll/glue", "mnli")
    dataset = datasets.DatasetDict({
        "train": dataset["train"],
        "validation": dataset["validation_matched"]
    })
    dataset = build_contextual_dataset(dataset)
    train_labels = [int(sample["label"]) for sample in dataset['train']]
    unique_labels = sorted(list(set(train_labels)))

    # Setup model
    model = AutoModelForSequenceClassification.from_pretrained(
        params["model_name"],
        num_labels=len(unique_labels),
        problem_type="single_label_classification"
    )

    # Setup tokenizer and data collator
    try:
        tokenizer = AutoTokenizer.from_pretrained(params["model_name"], use_fast=True)
    except Exception as e:
        warnings.warn(f"Fast tokenizer load failed ({e}). Falling back to slow tokenizer. Consider installing 'sentencepiece' (and optionally 'tiktoken').")
        tokenizer = AutoTokenizer.from_pretrained(params["model_name"], use_fast=False)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {"additional_special_tokens": ["[PREMISE]", "[HYPOTHESIS]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tokenize dataset
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, "label", text_field="text_concat", max_length=params["max_length"]
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    # Setup LoRA (conditional)
    model_to_train = model
    if params.get("enable_lora", True):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules=params["target_modules"],
            lora_dropout=params["lora_dropout"],
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
        )
        model_to_train = get_peft_model(model, lora_config)

    # Setup training
    checkpoint_output_dir = f"./{checkpoints_base_dir}/{params['experiment_name']}"
    class_weights_tensor = calculate_class_weights(train_labels, params["max_weight"], params["min_weight"]).to(device)
    update_steps_per_epoch = int(np.ceil(len(tokenized_datasets["train"]) / params["gradient_accumulation_steps"] / params["batch_size"]))
    total_update_steps = int(params["no_epochs"] * update_steps_per_epoch)
    eval_steps = max(1, int(update_steps_per_epoch // 6))
    save_steps = int(eval_steps)
    logging_steps = max(1, int(update_steps_per_epoch // 12))
    advanced_training_args = TrainingArguments(
        output_dir=checkpoint_output_dir,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=params["learning_rate"],
        num_train_epochs=params["no_epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        warmup_steps=int(params["warmup_steps"] * total_update_steps),
        weight_decay=params["weight_decay"],
        fp16=True if device == "cuda" and not torch.cuda.is_bf16_supported() else False,
        bf16=True if device == "cuda" and torch.cuda.is_bf16_supported() else False,
        gradient_checkpointing=params["gradient_checkpointing"],
        report_to=["mlflow"],
        dataloader_num_workers=0,    # Important for MPS compatibility
        remove_unused_columns=False, # Required for custom trainer
    )
    advanced_trainer = AdvancedTrainer(
        loss_type=params["loss_type"],
        class_weights=class_weights_tensor,
        model=model_to_train,
        args=advanced_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        focal_gamma=params["focal_gamma"],
        loss_reduction=params["loss_reduction"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params['early_stopping_patience'], early_stopping_threshold=params['early_stopping_threshold'])]
    )

    # MLflow logging
    with mlflow.start_run(run_name=params.get("experiment_name")):
        # mlflow.log_params(params)
        # Train
        print(f"\nStarting advanced training with:")
        print(params)
        advanced_result = advanced_trainer.train()
        print(f"\nAdvanced training completed!")    
        print(f"Final train loss: {advanced_result.training_loss:.4f}")

        # Save checkpoint
        best_ckpt = advanced_trainer.state.best_model_checkpoint
        print(f"{best_ckpt=}")
        if best_ckpt:
            print(f"Saving best checkpoint to {best_ckpt}")
            mlflow.log_artifacts(best_ckpt, artifact_path="best_checkpoint")

        # Evaluate and log
        end_time = time.time()
        total_time_seconds = end_time - start_time
        comprehensive_evaluation(advanced_trainer, tokenized_datasets, params, results_base_dir=results_base_dir, total_time_seconds=total_time_seconds)
        result_dir = f"{results_base_dir}/{params['experiment_name']}"
        if os.path.isdir(result_dir):
            mlflow.log_artifacts(result_dir, artifact_path="results")


if __name__ == "__main__":
    params = {
        "model_name": "microsoft/deberta-v2-xlarge",

        "no_epochs": 6,
        "batch_size": 8,
        "gradient_accumulation_steps": 32,
        "gradient_checkpointing": False,
        "learning_rate": 0.00008,
        "warmup_steps": 0.1,
        "weight_decay": 0.03,
        
        "enable_lora": True,
        "target_modules": ["query_proj", "key_proj", "value_proj", "o_proj"],
        "lora_dropout": 0.05,
        "lora_r": 16,
        "lora_alpha": 32,

        "loss_type": "ce",
        "loss_reduction": "mean",
        "focal_gamma": 2,
        "max_weight": 1,
        "min_weight": 1,

        "max_length": 128,

        "early_stopping_patience": 4,
        "early_stopping_threshold": 0.001,

        "experiment_name": "ex15_deberta_v2_xl_lora_stabilized_lr8e-5_wu10_r16",
        "experiment_description": "DeBERTa-v2-xlarge LoRA stabilized: LR=8e-5. Similar problem to Gemma7B, similar solution.",
        "logging_experiment_name": "/Shared/SLMs"
    }


    fine_tune_model(params)
    print("Done")