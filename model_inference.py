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



def compose_text(premise: str, hypothesis: str) -> str:
    return f"[PREMISE] {premise} [HYPOTHESIS] {hypothesis}"


def ensure_tokenizer_special_tokens(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {"additional_special_tokens": ["[PREMISE]", "[HYPOTHESIS]"]}
    try:
        tokenizer.add_special_tokens(special_tokens_dict)
    except Exception:
        # If already present or not supported, ignore
        pass


def load_tokenizer(checkpoint_path: str, model_name: Optional[str]) -> AutoTokenizer:
    # Prefer tokenizer saved in checkpoint (if available)
    try:
        # For local paths, add local_files_only=True to avoid HF validation
        if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception:
        base_model_name = model_name
        # If model_name not provided or failed, try reading LoRA adapter config for base model
        if (not base_model_name) or base_model_name is None:
            adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                try:
                    with open(adapter_cfg_path, "r") as f:
                        adapter_cfg = json.load(f)
                    base_model = adapter_cfg.get("base_model_name_or_path")
                    if base_model:
                        base_model_name = base_model
                except Exception:
                    pass
        if not base_model_name:
            raise ValueError("Could not determine tokenizer source. Provide --params_json with 'model_name' or ensure checkpoint contains a tokenizer or LoRA adapter_config.json with base model.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    ensure_tokenizer_special_tokens(tokenizer)
    return tokenizer


def load_model(
    checkpoint_path: str,
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
        base_model_name = None
        adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r") as f:
                    adapter_cfg = json.load(f)
                base_model_name = adapter_cfg.get("base_model_name_or_path")
            except Exception:
                base_model_name = None
        if not base_model_name:
            raise ValueError("LoRA adapter found but base_model_name_or_path missing in adapter_config.json")
        # Prefer local base checkpoints if available
        def _map_to_local_base(name: str) -> str:
            local_dir = os.environ.get("LOCAL_CHECKPOINTS_DIR", "./checkpoints")
            slug = name.strip().rstrip("/").split("/")[-1]
            candidate = os.path.join(local_dir, slug)
            return candidate if os.path.isdir(candidate) else name

        base_source = _map_to_local_base(base_model_name)
        local_only = os.path.isdir(base_source)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_source,
            num_labels=3,
            problem_type="single_label_classification",
            local_files_only=local_only,
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
    # Ensure embedding size matches tokenizer if special tokens were added
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
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
    device: str,
    max_length: int,
    batch_size: int,
    max_eval_samples: Optional[int] = None,
) -> dict:
    raw = load_dataset("nyu-mll/glue", "mnli")
    dataset = datasets.DatasetDict({
        "validation": raw["validation_matched"],
    })
    dataset = build_contextual_dataset(dataset)
    tokenized = tokenize_dataset(dataset["validation"], tokenizer, max_length)

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

    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=_collate)

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
    preds, probs = predict_batch(
        model=model,
        tokenizer=tokenizer,
        pairs=[{"premise": premise, "hypothesis": hypothesis}],
        max_length=max_length,
        device=device,
        batch_size=1,
    )
    return int(preds[0]), probs[0]


def predict_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    pairs: list,
    max_length: int,
    device: str,
    batch_size: int,
) -> Tuple[list, list]:
    texts = [compose_text(item["premise"], item["hypothesis"]) for item in pairs]
    all_preds = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend([int(p) for p in preds])
            all_probs.extend(probs)
    return all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description="Simple MNLI inference for full or LoRA checkpoints")
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint directory (full or LoRA)")
    parser.add_argument("--mode", choices=["single", "batch", "eval"], default="single", help="Inference mode")
    parser.add_argument("--premise", default=None, help="Premise text for single prediction")
    parser.add_argument("--hypothesis", default=None, help="Hypothesis text for single prediction")
    parser.add_argument("--batch_json", default=None, help="Path to JSON/JSONL file for batch predictions")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for batch predictions")
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

    # Load tokenizer and model
    tokenizer = load_tokenizer(local_ckpt_path, None)
    model, is_lora = load_model(
        checkpoint_path=local_ckpt_path,
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
            max_length=128,
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

    if args.mode == "batch":
        if not args.batch_json or not os.path.isfile(args.batch_json):
            raise ValueError("--batch_json is required for batch mode and must point to a file")

        def _load_pairs(path: str) -> list:
            with open(path, "r") as f:
                content = f.read().strip()
            if not content:
                return []
            # Try JSON array or object with items first
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                    return data["items"]
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            # Fallback to JSONL
            pairs = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pairs.append(obj)
            return pairs

        pairs = _load_pairs(args.batch_json)
        if not pairs:
            raise ValueError("No items found in --batch_json. Expected list of {premise, hypothesis}.")
        # Validate
        for idx, it in enumerate(pairs):
            if not isinstance(it, dict) or "premise" not in it or "hypothesis" not in it:
                raise ValueError(f"Item at index {idx} is invalid. Expected keys: 'premise', 'hypothesis'.")

        batch_size = args.batch_size or 16
        inf_start_ts = time.time()
        preds, probs = predict_batch(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            max_length=128,
            device=device,
            batch_size=batch_size,
        )
        inf_end_ts = time.time()
        inference_seconds = round(inf_end_ts - inf_start_ts, 4)
        results = [
            {"prediction": int(preds[i]), "probs": probs[i]} for i in range(len(preds))
        ]
        print(json.dumps({
            "results": results,
            "timings": {
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "batch_size": batch_size,
                "num_items": len(results)
            }
        }, indent=2))
        return

    # Eval mode
    eval_start_ts = time.time()
    metrics = evaluate_on_mnli(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=128,
        batch_size=args.batch_size or 16,
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
    