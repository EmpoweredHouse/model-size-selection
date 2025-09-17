import os
import json
import argparse
import warnings
from typing import Optional, Tuple
import tempfile
import time

import numpy as np

import torch
import torch.nn.functional as F

import datasets
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from peft import PeftModel

from sklearn.metrics import accuracy_score, f1_score

import mlflow
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login as hf_login


warnings.filterwarnings("ignore")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    return os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))


def try_load_params(checkpoint_path: str, params_json: Optional[str]) -> dict:
    if params_json and os.path.isfile(params_json):
        with open(params_json, "r") as f:
            return json.load(f)

    # Infer experiment_name from checkpoints/<experiment_name>/checkpoint-xxx
    experiment_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    candidate = os.path.join(os.path.dirname(__file__), "results", experiment_name, "hyperparams.json")
    if os.path.isfile(candidate):
        with open(candidate, "r") as f:
            return json.load(f)
            
    raise ValueError(f"No hyperparams.json found in results/{experiment_name}")


def ensure_tokenizer_special_tokens(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {"additional_special_tokens": ["[PREMISE]", "[HYPOTHESIS]"]}
    try:
        tokenizer.add_special_tokens(special_tokens_dict)
    except Exception:
        # If already present or not supported, ignore
        pass


def load_tokenizer(checkpoint_path: str, model_name: str) -> AutoTokenizer:
    # Prefer tokenizer saved in checkpoint (if available)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_tokenizer_special_tokens(tokenizer)
    return tokenizer


def load_model(
    checkpoint_path: str,
    params: dict,
    tokenizer: AutoTokenizer,
    device: str,
    merge_lora: bool = False,
) -> Tuple[torch.nn.Module, bool]:
    """Load full or LoRA model based on checkpoint contents.

    Returns (model, is_lora).
    """
    torch_dtype = None
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    load_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}

    if is_lora_checkpoint(checkpoint_path):
        # If model name is missing, try to read from adapter_config.json
        if "model_name" not in params:
            adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                try:
                    with open(adapter_cfg_path, "r") as f:
                        adapter_cfg = json.load(f)
                    base_model = adapter_cfg.get("base_model_name_or_path")
                    if base_model:
                        params["model_name"] = base_model
                except Exception:
                    pass

        base_model = AutoModelForSequenceClassification.from_pretrained(
            params["model_name"],
            num_labels=params.get("num_labels", 3),
            problem_type="single_label_classification",
            **load_kwargs,
        )
        # Ensure embedding size matches tokenizer if special tokens were added
        try:
            base_model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        if merge_lora:
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
        model.to(device)
        model.eval()
        return model, True

    # Full model checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, **load_kwargs)
    model.to(device)
    model.eval()
    return model, False


def is_artifact_uri(path: str) -> bool:
    if path.startswith("runs:/"):
        return True
    if path.startswith("dbfs:/"):
        return True
    if "://" in path:
        return True
    return False


def resolve_checkpoint_path(path: str) -> Tuple[str, Optional[str]]:
    """Return a local directory path for the checkpoint. If remote, download via MLflow.

    Returns (local_path, tmp_dir) where tmp_dir is a temporary directory to be cleaned up by caller if desired.
    """
    if not is_artifact_uri(path):
        return path, None

    tmp_dir = tempfile.mkdtemp(prefix="ckpt_")
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=path, dst_path=tmp_dir)
        return local_path, tmp_dir
    except Exception as e:
        raise RuntimeError(f"Failed to download artifacts from URI: {path}. Error: {e}")


def build_contextual_dataset(dataset: datasets.DatasetDict) -> datasets.DatasetDict:
    new_splits = {}
    for split_name, split in dataset.items():
        df = split.to_pandas().reset_index(drop=True)
        if "premise" in df.columns and "hypothesis" in df.columns:
            df["text_concat"] = (
                "[PREMISE] " + df["premise"].astype(str) + " [HYPOTHESIS] " + df["hypothesis"].astype(str)
            )
        else:
            raise ValueError("Expected 'premise' and 'hypothesis' columns in the dataset")
        new_splits[split_name] = datasets.Dataset.from_pandas(df, preserve_index=False)
    return datasets.DatasetDict(new_splits)


def tokenize_dataset(dataset: datasets.Dataset, tokenizer: AutoTokenizer, max_length: int) -> datasets.Dataset:
    def _pp(examples):
        tokens = tokenizer(
            examples["text_concat"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        if "label" in examples:
            tokens["labels"] = [int(l) for l in examples["label"]]
        return tokens
    # Remove all original columns; keep only tokenizer outputs (and labels if added)
    return dataset.map(_pp, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")


def evaluate_on_mnli(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    params: dict,
    device: str,
    max_eval_samples: Optional[int] = None,
) -> dict:
    raw = load_dataset("nyu-mll/glue", "mnli")
    dataset = datasets.DatasetDict({
        "validation": raw["validation_matched"],
    })
    dataset = build_contextual_dataset(dataset)
    tokenized = tokenize_dataset(dataset["validation"], tokenizer, params.get("max_length", 128))

    if max_eval_samples is not None and max_eval_samples > 0:
        tokenized = tokenized.select(range(min(max_eval_samples, len(tokenized))))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _collate(examples):
        batch = collator(examples)
        # Ensure tensors are on device
        for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch

    from torch.utils.data import DataLoader

    loader = DataLoader(tokenized, batch_size=params.get("batch_size", 16), shuffle=False, collate_fn=_collate)

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels") if "labels" in batch else None
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            if labels is not None:
                all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0) if all_labels else None

    if y_true is not None:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        }
    else:
        metrics = {}

    return metrics


def predict_single(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    premise: str,
    hypothesis: str,
    max_length: int,
    device: str,
) -> Tuple[int, list]:
    text = f"[PREMISE] {premise} [HYPOTHESIS] {hypothesis}"
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(logits, dim=-1).item())
    return pred, probs


def main():
    parser = argparse.ArgumentParser(description="Simple MNLI inference for full or LoRA checkpoints")
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint directory (full or LoRA)")
    parser.add_argument("--params_json", default=None, help="Optional path to params JSON (like in training)")
    parser.add_argument("--mode", choices=["single", "eval"], default="single", help="Inference mode")
    parser.add_argument("--premise", default=None, help="Premise text for single prediction")
    parser.add_argument("--hypothesis", default=None, help="Hypothesis text for single prediction")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit num validation samples for quick eval")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base weights for inference")
    args = parser.parse_args()

    # Load environment variables
    try:
        load_dotenv(find_dotenv())
    except Exception:
        pass

    # Optional HF login if token is available (needed for private/base models)
    try:
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            hf_login(token=hf_token, add_to_git_credential=False)
    except Exception as e:
        warnings.warn(f"Hugging Face login failed: {e}")

    # If using Databricks/MLflow artifact URIs, ensure tracking URI is set
    if is_artifact_uri(args.checkpoint_path) and (
        args.checkpoint_path.startswith("dbfs:/") or args.checkpoint_path.startswith("runs:/")
    ):
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = "databricks"

    device = detect_device()
    print(f"Using device: {device}")

    # Resolve checkpoint path (local or remote URI)
    local_ckpt_path, tmp_dir = resolve_checkpoint_path(args.checkpoint_path)
    if tmp_dir:
        print(f"Downloaded checkpoint to: {local_ckpt_path}")

    # Start measuring model load time AFTER any artifact download
    load_start_ts = time.time()

    params = try_load_params(local_ckpt_path, args.params_json)
    # If num_labels can be inferred from config.json in checkpoint, use it
    try:
        cfg_path = os.path.join(local_ckpt_path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and "num_labels" in cfg:
                params.setdefault("num_labels", cfg["num_labels"])
    except Exception:
        pass
    if "model_name" not in params:
        raise ValueError("'model_name' must be available in params (via --params_json or results/<exp>/hyperparams.json)")

    # Load tokenizer and model
    tokenizer = load_tokenizer(local_ckpt_path, params["model_name"])
    model, is_lora = load_model(
        checkpoint_path=local_ckpt_path,
        params=params,
        tokenizer=tokenizer,
        device=device,
        merge_lora=args.merge_lora,
    )
    load_end_ts = time.time()
    model_load_seconds = round(load_end_ts - load_start_ts, 4)
    print(f"Loaded {'LoRA' if is_lora else 'full'} checkpoint from: {args.checkpoint_path}")

    if args.mode == "single":
        if not args.premise or not args.hypothesis:
            raise ValueError("--premise and --hypothesis are required for single mode")
        inf_start_ts = time.time()
        pred, probs = predict_single(
            model=model,
            tokenizer=tokenizer,
            premise=args.premise,
            hypothesis=args.hypothesis,
            max_length=params.get("max_length", 128),
            device=device,
        )
        inf_end_ts = time.time()
        inference_seconds = round(inf_end_ts - inf_start_ts, 4)
        print(json.dumps({
            "prediction": pred,
            "probs": probs,
            "timings": {
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds
            }
        }, indent=2))
        return

    # Eval mode
    eval_start_ts = time.time()
    metrics = evaluate_on_mnli(
        model=model,
        tokenizer=tokenizer,
        params=params,
        device=device,
        max_eval_samples=args.max_eval_samples,
    )
    eval_end_ts = time.time()
    inference_total_seconds = round(eval_end_ts - eval_start_ts, 4)
    n_samples = None
    try:
        raw = load_dataset("nyu-mll/glue", "mnli")
        n_samples = len(raw["validation_matched"]) if args.max_eval_samples is None else min(args.max_eval_samples, len(raw["validation_matched"]))
    except Exception:
        pass
    result = {
        "metrics": metrics,
        "timings": {
            "model_load_seconds": model_load_seconds,
            "inference_total_seconds": inference_total_seconds,
        }
    }
    if n_samples:
        result["timings"]["samples"] = n_samples
        result["timings"]["samples_per_second"] = round(n_samples / inference_total_seconds, 4) if inference_total_seconds > 0 else None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
    # python model_inference.py --checkpoint_path your_checkpoint_path_or_uri --mode single --premise 'A soccer game with multiple males playing.' --hypothesis 'Some men are playing a sport.' --merge_lora
    # python model_inference.py --checkpoint_path your_checkpoint_path_or_uri --mode eval --max_eval_samples 200 --merge_lora

