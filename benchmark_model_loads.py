import os
import sys
import json
import time
import statistics
import gc
import argparse
import subprocess
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
BENCHMARK_RUNS = 11  # default; when scripting, you can run single-run mode and loop externally
BENCHMARK_OUTPUT = "model_load_benchmark.json"


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


# removed unused direct in-process merge helper; merging is done in a separate subprocess


def _child_once(checkpoint_path: str) -> None:
    """Child mode: perform a single cold-start load and print JSON with the time.

    This function should not print anything else to stdout.
    """
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

    # Force device detection per process to include CUDA context init cost
    _ = detect_device()

    # Resolve checkpoint path in the child (for artifact URIs/HF/local)
    local_path, _ = resolve_checkpoint_path(checkpoint_path)

    # Isolation knobs: point HF caches to per-process temp dirs to avoid cross-run reuse
    # (Can be expensive; ensures more comparable IO.)
    os.environ["TRANSFORMERS_CACHE"] = os.path.join("tmp", "hf_cache_child")
    os.environ["HF_HOME"] = os.path.join("tmp", "hf_home_child")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("tmp", "hf_hub_child")
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)

    # Prefer local checkpoints for base models when loading LoRA
    os.environ["LOCAL_CHECKPOINTS_DIR"] = os.path.abspath("./checkpoints")

    # Measure a single load
    seconds = _load_once(local_ckpt_path=local_path, merge_lora=False, device=detect_device())
    # Print strictly JSON for parent to parse
    print(json.dumps({"time": seconds}))


def _run_child_process_once(checkpoint_path: str) -> float:
    """Spawn a fresh Python process to perform one load and return the measured seconds."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--checkpoint_path", checkpoint_path]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"Child process failed: {result.stderr.strip()}")
    # Expect a single JSON object on stdout
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "{}"
    try:
        obj = json.loads(line)
        return float(obj.get("time"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse child output: '{line}'. Error: {e}")


def _merge_child_once(checkpoint_path: str, save_dir: str) -> None:
    """Child mode: perform LoRA merge and save to save_dir. Print JSON with merge_time.

    Ensures the process exits after merging to avoid keeping models resident.
    """
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

    _ = detect_device()

    local_path, _ = resolve_checkpoint_path(checkpoint_path)

    # Isolate caches
    os.environ["TRANSFORMERS_CACHE"] = os.path.join("tmp", "hf_cache_merge_child")
    os.environ["HF_HOME"] = os.path.join("tmp", "hf_home_merge_child")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("tmp", "hf_hub_merge_child")
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)

    os.environ["LOCAL_CHECKPOINTS_DIR"] = os.path.abspath("./checkpoints")

    # Load tokenizer and model (LoRA attached if adapter checkpoint)
    tokenizer = load_tokenizer(local_path, None)
    model, is_lora = load_model(
        checkpoint_path=local_path,
        tokenizer=tokenizer,
        device=detect_device(),
        merge_lora=False,
    )

    # If not LoRA, just save as-is
    if not is_lora:
        try:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)
            print(json.dumps({"merge_time": 0.0}))
            return
        finally:
            try:
                del model
                del tokenizer
            except Exception:
                pass

    start_ts = time.time()
    merged_model = model.merge_and_unload()
    end_ts = time.time()
    merge_seconds = round(end_ts - start_ts, 6)

    # Save merged
    os.makedirs(save_dir, exist_ok=True)
    merged_model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    try:
        del merged_model
        del model
        del tokenizer
    except Exception:
        pass
    _clear_gpu_memory()
    print(json.dumps({"merge_time": merge_seconds}))


def _run_merge_child_process(checkpoint_path: str, save_dir: str) -> float:
    cmd = [sys.executable, os.path.abspath(__file__), "--merge-child", "--checkpoint_path", checkpoint_path, "--save_dir", save_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"Merge child process failed: {result.stderr.strip()}")
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "{}"
    try:
        obj = json.loads(line)
        return float(obj.get("merge_time", 0.0))
    except Exception as e:
        raise RuntimeError(f"Failed to parse merge child output: '{line}'. Error: {e}")


def run_benchmark_single(run_name: str, checkpoint_path: str, merge_lora: bool):
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

    # Prepare variants once (resolve paths, do single merges), then run per-iteration across models
    prepared_variants: List[Dict] = []
    results: List[Dict] = []

    # Prepare the single variant
    local_ckpt_path, tmp_dir = resolve_checkpoint_path(checkpoint_path)
        ckpt_type = "LoRA" if is_lora_checkpoint(local_ckpt_path) else "full"

        lora_merge_time: float = 0.0
        merged_ckpt_for_load: str = None
    if ckpt_type == "LoRA" and merge_lora:
        run_dir_name = _sanitize_name(run_name or os.path.basename(local_ckpt_path))
        merged_dir = os.path.join("tmp", "merged", run_dir_name)
            # Merge in a separate process to avoid keeping model in memory
            lora_merge_time = _run_merge_child_process(checkpoint_path=local_ckpt_path, save_dir=merged_dir)
            merged_ckpt_for_load = merged_dir
            load_path = merged_ckpt_for_load
        else:
            load_path = local_ckpt_path

        prepared_variants.append({
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "merge_lora": bool(merge_lora),
            "ckpt_type": ckpt_type,
            "load_path": load_path,
            "lora_merge_time": lora_merge_time,
            "merged_checkpoint_path": merged_ckpt_for_load,
    })

        rec = {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "merge_lora": bool(merge_lora),
            "times": [],
            "avg_time": None,
            "device": device,
        }
        if ckpt_type == "LoRA" and merge_lora:
            rec["lora_merge_time"] = lora_merge_time
            rec["merged_checkpoint_path"] = merged_ckpt_for_load
        results.append(rec)

        # Cleanup any temporary directory from artifact resolution
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # Run the single variant BENCHMARK_RUNS times
    for i in range(BENCHMARK_RUNS):
        print(f"\n=== Iteration {i+1}/{BENCHMARK_RUNS} ===")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join("tmp", f"hf_cache_{i}")
        os.environ["HF_HOME"] = os.path.join("tmp", f"hf_home_{i}")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join("tmp", f"hf_hub_{i}")
        v = prepared_variants[0]
        print(f"Running: {v['run_name']}")
        secs = _run_child_process_once(checkpoint_path=v["load_path"])
        results[0]["times"].append(secs)
        print(f"  Time (cold): {secs:.6f}s")

    # Compute averages (exclude first cold start from avg; record it separately)
    rec = results[0]
    if rec["times"]:
        rec["cold_start_time"] = rec["times"][0]
        rest = rec["times"][1:] if len(rec["times"]) > 1 else []
        rec["avg_time"] = round(statistics.mean(rest), 6) if rest else rec["cold_start_time"]

    # Post-process: approximate adapter swap time for LoRA (merge_lora == False)
    # Strategy: use adapter_config.json to find base model name and compare against a base/full record.
    # Fallback: match by run_name prefix against a '*-full-false' record.
    def _run_prefix(name: str) -> str:
        return (name or "").split("-", 1)[0].strip().lower()

    # Build quick lookups
    avg_time_by_checkpoint_path = {}
    full_false_time_by_prefix = {}
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

    try:
        if rec.get("merge_lora") is False:
            local_path, _ = resolve_checkpoint_path(rec.get("checkpoint_path"))
            if is_lora_checkpoint(local_path):
                base_name = _adapter_base_name(local_path)
                base_time = None
                if base_name and base_name in avg_time_by_checkpoint_path:
                    base_time = avg_time_by_checkpoint_path[base_name]
                if base_time is None:
                    base_time = full_false_time_by_prefix.get(_run_prefix(rec.get("run_name", "")))
                if base_time is not None and rec.get("avg_time") is not None:
                    rec["adapter_swap_time_approx"] = max(0.0, round(rec["avg_time"] - base_time, 6))
    except Exception:
        pass

    # Save results to hardcoded output file
    with open(BENCHMARK_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark to {BENCHMARK_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help="Run a single cold-start load and print time as JSON")
    parser.add_argument("--merge-child", action="store_true", help="Run a single merge of LoRA and print merge_time as JSON")
    parser.add_argument("--single", action="store_true", help="Run benchmark for a single variant (use with --run_name, --checkpoint_path, --merge_lora)")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for single mode")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for child/merge/single mode")
    parser.add_argument("--merge_lora", type=str, default="false", help="Whether to merge LoRA in single mode (true/false)")
    parser.add_argument("--save_dir", type=str, default=None, help="Save dir for merged model in merge-child mode")
    args = parser.parse_args()

    if args.child:
        if not args.checkpoint_path:
            raise SystemExit("--checkpoint_path is required in --child mode")
        _child_once(checkpoint_path=args.checkpoint_path)
    elif args.merge_child:
        if not args.checkpoint_path or not args.save_dir:
            raise SystemExit("--checkpoint_path and --save_dir are required in --merge-child mode")
        _merge_child_once(checkpoint_path=args.checkpoint_path, save_dir=args.save_dir)
    elif args.single:
        if not args.run_name or not args.checkpoint_path:
            raise SystemExit("--run_name and --checkpoint_path are required in --single mode")
        ml = args.merge_lora.lower() in ("1", "true", "yes")
        run_benchmark_single(run_name=args.run_name, checkpoint_path=args.checkpoint_path, merge_lora=ml)
    else:
        # Backward compatibility: expect env/VARIANTS external wrapper to call --single repeatedly
        raise SystemExit("Specify --single with --run_name, --checkpoint_path, --merge_lora; or use child/merge-child modes.")

