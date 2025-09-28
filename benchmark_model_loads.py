import os
import json
import time
import statistics
import gc
from typing import List, Dict

import torch

from model_inference import (
    detect_device,
    resolve_checkpoint_path,
    load_tokenizer,
    load_model,
    is_lora_checkpoint,
)
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login as hf_login

# Hardcoded benchmark config
BENCHMARK_RUNS = 10
BENCHMARK_OUTPUT = "model_load_benchmark.json"

VARIANTS: List[Dict] = [
    {
        "run_name": "DistilBERT-full-false",
        "checkpoint_path": "./checkpoints/ex09_distilbert_full_ce_lr1.5e-5_wd0.02_wu8/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "DistilBERT-lora-false",
        "checkpoint_path": "./checkpoints/ex10_distilbert_lora_r32_lr2e-4/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "DistilBERT-lora-true",
        "checkpoint_path": "./checkpoints/ex10_distilbert_lora_r32_lr2e-4/best_checkpoint",
        "merge_lora": True,
    },
    {
        "run_name": "RoBERTa-full-false",
        "checkpoint_path": "./checkpoints/ex12_roberta_full_baseline/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "RoBERTa-lora-false",
        "checkpoint_path": "./checkpoints/ex11_roberta_lora_baseline/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "RoBERTa-lora-true",
        "checkpoint_path": "./checkpoints/ex11_roberta_lora_baseline/best_checkpoint",
        "merge_lora": True,
    },
    {
        "run_name": "DeBERTa_v2_xlarge-full-false",
        "checkpoint_path": "./checkpoints/ex14_deberta_v2_xl_fullft_baseline/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "DeBERTa_v2_xlarge-lora-false",
        "checkpoint_path": "./checkpoints/ex15_deberta_v2_xl_lora_stabilized_lr8e-5_wu10_r16/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "DeBERTa_v2_xlarge-lora-true",
        "checkpoint_path": "./checkpoints/ex15_deberta_v2_xl_lora_stabilized_lr8e-5_wu10_r16/best_checkpoint",
        "merge_lora": True,
    },
    {
        "run_name": "Gemma2B-full-false",
        "checkpoint_path": "google/gemma-2b",
        "merge_lora": False,
    },
    {
        "run_name": "Gemma2B-lora-false",
        "checkpoint_path": "./checkpoints/ex06_gemma2b_lora_bigger_lora/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "Gemma2B-lora-true",
        "checkpoint_path": "./checkpoints/ex06_gemma2b_lora_bigger_lora/best_checkpoint",
        "merge_lora": True,
    },
    {
        "run_name": "Gemma7B-full-false",
        "checkpoint_path": "google/gemma-7b",
        "merge_lora": False,
    },
    {
        "run_name": "Gemma7B-lora-false",
        "checkpoint_path": "./checkpoints/ex08_gemma7b_lora_bigger_stable/best_checkpoint",
        "merge_lora": False,
    },
    {
        "run_name": "Gemma7B-lora-true",
        "checkpoint_path": "./checkpoints/ex08_gemma7b_lora_bigger_stable/best_checkpoint",
        "merge_lora": True,
    },
]


def _clear_gpu_memory():
    try:
        # Ensure Python references are released and GC runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Reclaim inter-process cached blocks if any
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()
    except Exception:
        pass


def _load_once(local_ckpt_path: str, merge_lora: bool, device: str) -> float:
    """Load tokenizer and model once; return load time in seconds.

    LoRA merge time is explicitly excluded by performing merge after timing when requested.
    """
    # Load tokenizer outside of the timed region? Keep it inside to mirror real usage
    start_ts = time.time()
    tokenizer = load_tokenizer(local_ckpt_path, None)
    model, is_lora = load_model(
        checkpoint_path=local_ckpt_path,
        tokenizer=tokenizer,
        device=device,
        merge_lora=False,  # Always load without merge inside timed region
    )
    end_ts = time.time()
    load_seconds = round(end_ts - start_ts, 6)

    # Cleanup to avoid accumulation across runs
    try:
        del model
        del tokenizer
    except Exception:
        pass
    _clear_gpu_memory()

    return load_seconds


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (name or "merged"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _merge_lora_to_dir(local_ckpt_path: str, device: str, save_dir: str) -> float:
    """Merge LoRA adapter into base weights and save to save_dir.

    Returns merge time in seconds (merge only, excludes save time).
    """
    tokenizer = load_tokenizer(local_ckpt_path, None)
    # Load as LoRA-attached model (no merge yet)
    model, is_lora = load_model(
        checkpoint_path=local_ckpt_path,
        tokenizer=tokenizer,
        device=device,
        merge_lora=False,
    )
    if not is_lora:
        # Nothing to merge
        # Save tokenizer to ensure subsequent loads find it
        _ensure_dir(save_dir)
        try:
            tokenizer.save_pretrained(save_dir)
        except Exception:
            pass
        # Save model as-is
        try:
            model.save_pretrained(save_dir)
        except Exception:
            pass
        try:
            del model
            del tokenizer
        except Exception:
            pass
        _clear_gpu_memory()
        return 0.0

    # Measure merge only
    start_ts = time.time()
    merged_model = model.merge_and_unload()
    end_ts = time.time()
    merge_seconds = round(end_ts - start_ts, 6)

    # Save merged model and tokenizer
    _ensure_dir(save_dir)
    try:
        merged_model.save_pretrained(save_dir)
    except Exception:
        pass
    try:
        tokenizer.save_pretrained(save_dir)
    except Exception:
        pass

    try:
        del merged_model
        del model
        del tokenizer
    except Exception:
        pass
    _clear_gpu_memory()
    return merge_seconds


def run_benchmark():
    # Load environment and HF auth for remote models (e.g., google/*)
    try:
        load_dotenv(find_dotenv())
    except Exception:
        pass
    try:
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            hf_login(token=hf_token, add_to_git_credential=False)
    except Exception:
        pass

    device = detect_device()
    print(f"Device: {device}")
    if device != "cuda":
        print("Warning: CUDA not available; results may not reflect GPU load times.")

    results: List[Dict] = []

    for variant in VARIANTS:
        print(f"\n=== Running: {variant['run_name']} ===")
        checkpoint_path = variant["checkpoint_path"]
        merge_lora = bool(variant.get("merge_lora", False))

        # Resolve path once per variant to avoid repeated downloads if using artifact URIs
        local_ckpt_path, tmp_dir = resolve_checkpoint_path(checkpoint_path)

        # Pre-flight: print what type of checkpoint it is
        ckpt_type = "LoRA" if is_lora_checkpoint(local_ckpt_path) else "full"
        print(f"Checkpoint type detected: {ckpt_type}")

        times: List[float] = []
        lora_merge_time: float = 0.0
        merged_ckpt_for_load: str = None

        if ckpt_type == "LoRA" and merge_lora:
            # Merge once (measure merge time only), save merged, then measure loads of merged model
            run_dir_name = _sanitize_name(variant.get("run_name") or os.path.basename(local_ckpt_path))
            merged_dir = os.path.join("tmp", "merged", run_dir_name)
            lora_merge_time = _merge_lora_to_dir(local_ckpt_path=local_ckpt_path, device=device, save_dir=merged_dir)
            merged_ckpt_for_load = merged_dir
            load_path = merged_ckpt_for_load
        else:
            # Load the provided checkpoint as-is (either LoRA without merge, or full)
            load_path = local_ckpt_path

        for i in range(BENCHMARK_RUNS):
            secs = _load_once(local_ckpt_path=load_path, merge_lora=False, device=device)
            times.append(secs)
            print(f"Run {i+1}/{BENCHMARK_RUNS}: {secs:.6f}s")

        avg_time = round(statistics.mean(times), 6) if times else None

        # Build output record preserving input fields
        record = {
            "run_name": variant.get("run_name"),
            "checkpoint_path": checkpoint_path,
            "merge_lora": merge_lora,
            "times": times,
            "avg_time": avg_time,
            "device": device,
        }
        if ckpt_type == "LoRA" and merge_lora:
            record["lora_merge_time"] = lora_merge_time
            record["merged_checkpoint_path"] = merged_ckpt_for_load

        results.append(record)

        # Cleanup any temporary directory from artifact resolution
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                # Best-effort cleanup; ignore failures
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # Post-process: approximate adapter swap time for LoRA (merge_lora == False)
    # Strategy: use adapter_config.json to find base model name and compare against a base/full record.
    # Fallback: match by run_name prefix against a '*-full-false' record.
    def _run_prefix(name: str) -> str:
        return (name or "").split("-", 1)[0].strip().lower()

    # Build quick lookups
    avg_time_by_checkpoint_path = {}
    full_false_time_by_prefix = {}
    for rec in results:
        cp = rec.get("checkpoint_path")
        if cp is not None and rec.get("avg_time") is not None and rec.get("merge_lora") is False:
            avg_time_by_checkpoint_path[cp] = rec["avg_time"]
        rn = rec.get("run_name", "").lower()
        if "full-false" in rn and rec.get("avg_time") is not None:
            full_false_time_by_prefix[_run_prefix(rec.get("run_name", ""))] = rec["avg_time"]

    def _adapter_base_name(local_path: str) -> str:
        try:
            cfg_path = os.path.join(local_path, "adapter_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                return cfg.get("base_model_name_or_path")
        except Exception:
            return None
        return None

    for rec in results:
        try:
            if rec.get("merge_lora") is False:
                # Resolve and verify if it's a LoRA adapter directory
                cp = rec.get("checkpoint_path")
                local_path, _ = resolve_checkpoint_path(cp)
                if is_lora_checkpoint(local_path):
                    base_name = _adapter_base_name(local_path)
                    base_time = None
                    if base_name and base_name in avg_time_by_checkpoint_path:
                        base_time = avg_time_by_checkpoint_path[base_name]
                    if base_time is None:
                        # Fallback by run prefix vs full-false
                        base_time = full_false_time_by_prefix.get(_run_prefix(rec.get("run_name", "")))
                    if base_time is not None and rec.get("avg_time") is not None:
                        rec["adapter_swap_time_approx"] = max(0.0, round(rec["avg_time"] - base_time, 6))
        except Exception:
            # Best-effort; ignore failures
            pass

    # Save results to hardcoded output file
    with open(BENCHMARK_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark to {BENCHMARK_OUTPUT}")


if __name__ == "__main__":
    run_benchmark()


