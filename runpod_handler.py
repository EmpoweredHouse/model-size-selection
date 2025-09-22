"""
RunPod Serverless Handler
This module provides a RunPod-compatible handler that wraps your FastAPI app
for use with RunPod's queue-based serverless endpoints.
"""

import runpod
from typing import Dict, Any, List
import time

# Import your existing model components
from model_inference import (
    detect_device,
    resolve_checkpoint_path,
    load_tokenizer,
    load_model,
    predict_single,
    predict_batch,
)
from api import ModelContext

# Global model context
ctx = None


def initialize_model():
    """Initialize the model context - called once when the worker starts"""
    global ctx
    if ctx is None:
        ctx = ModelContext()
        ctx.load()
    return ctx


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function that processes jobs in the format:
    {
        "input": {
            "action": "single" | "batch" | "health" | "info",
            "item": {"premise": "...", "hypothesis": "..."},  # for single
            "items": [{"premise": "...", "hypothesis": "..."}, ...],  # for batch
            "max_length": 128,  # optional
            "batch_size": 16   # optional, for batch only
        }
    }
    """
    try:
        print(f"🔍 HANDLER START - Received job: {job}")
        print(f"🔍 Job type: {type(job)}")
        
        # Initialize model if not already done
        print("🔍 Initializing model...")
        model_ctx = initialize_model()
        print(f"🔍 Model context loaded: model={model_ctx.model is not None}, tokenizer={model_ctx.tokenizer is not None}")
        
        if model_ctx.model is None or model_ctx.tokenizer is None or model_ctx.params is None:
            error_msg = f"Model not properly initialized - model: {model_ctx.model is not None}, tokenizer: {model_ctx.tokenizer is not None}, params: {model_ctx.params is not None}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Extract input data
        job_input = job.get("input", {})
        print(f"🔍 Job input: {job_input}")
        
        if not job_input:
            error_msg = "Missing 'input' field in job"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
            
        action = job_input.get("action", "single")
        print(f"🔍 Processing action: {action}")
        
        # Handle different actions
        if action == "health":
            return {
                "status": "ok",
                "device": model_ctx.device,
                "model_loaded": model_ctx.model is not None
            }
        
        elif action == "info":
            return {
                "device": model_ctx.device,
                "is_lora": model_ctx.is_lora,
                "timings": {"model_load_seconds": model_ctx.model_load_seconds},
                "defaults": {
                    "max_length": model_ctx.params.get("max_length", 128) if model_ctx.params else 128,
                    "batch_size": model_ctx.params.get("batch_size", 16) if model_ctx.params else 16,
                    "num_labels": model_ctx.params.get("num_labels", 3) if model_ctx.params else 3,
                },
            }
        
        elif action == "single":
            # Single prediction
            item = job_input.get("item", {})
            if not item or "premise" not in item or "hypothesis" not in item:
                return {"error": "Missing required fields: premise, hypothesis"}
            
            max_length = job_input.get("max_length") or model_ctx.params.get("max_length", 128)
            
            start_ts = time.time()
            pred, probs = predict_single(
                model=model_ctx.model,
                tokenizer=model_ctx.tokenizer,
                premise=item["premise"],
                hypothesis=item["hypothesis"],
                max_length=max_length,
                device=model_ctx.device,
            )
            end_ts = time.time()
            
            return {
                "prediction": int(pred),
                "probs": probs,
                "timings": {
                    "model_load_seconds": model_ctx.model_load_seconds,
                    "inference_seconds": round(end_ts - start_ts, 4),
                },
            }
        
        elif action == "batch":
            # Batch prediction
            items = job_input.get("items", [])
            if not items:
                return {"error": "Missing required field: items (non-empty list)"}
            
            # Validate items format
            for i, item in enumerate(items):
                if not isinstance(item, dict) or "premise" not in item or "hypothesis" not in item:
                    return {"error": f"Item {i} missing required fields: premise, hypothesis"}
            
            max_length = job_input.get("max_length") or model_ctx.params.get("max_length", 128)
            batch_size = job_input.get("batch_size") or model_ctx.params.get("batch_size", 16)
            
            start_ts = time.time()
            preds, probs = predict_batch(
                model=model_ctx.model,
                tokenizer=model_ctx.tokenizer,
                pairs=items,
                max_length=max_length,
                device=model_ctx.device,
                batch_size=batch_size,
            )
            end_ts = time.time()
            
            results = [{"prediction": int(preds[i]), "probs": probs[i]} for i in range(len(preds))]
            return {
                "results": results,
                "timings": {
                    "model_load_seconds": model_ctx.model_load_seconds,
                    "inference_seconds": round(end_ts - start_ts, 4),
                    "batch_size": batch_size,
                    "num_items": len(results),
                },
            }
        
        else:
            return {"error": f"Unknown action: {action}. Supported: single, batch, health, info"}
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"💥 HANDLER EXCEPTION: {str(e)}")
        print(f"💥 FULL TRACEBACK:\n{error_details}")
        return {"error": f"Handler error: {str(e)}", "traceback": error_details}


if __name__ == "__main__":
    # Start the RunPod serverless worker
    # In production, RunPod will call this automatically
    # For local testing, we can use runpod's server mode
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
