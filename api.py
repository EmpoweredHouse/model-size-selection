import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login as hf_login

from model_inference import (
    detect_device,
    resolve_checkpoint_path,
    load_tokenizer,
    load_model,
    predict_single,
    predict_batch,
)


class Item(BaseModel):
    premise: str = Field(..., description="Premise text")
    hypothesis: str = Field(..., description="Hypothesis text")


class SinglePredictRequest(BaseModel):
    item: Item
    max_length: Optional[int] = None


class SinglePredictResponse(BaseModel):
    prediction: int
    probs: List[float]
    timings: dict


class BatchPredictRequest(BaseModel):
    items: List[Item]
    max_length: Optional[int] = None
    batch_size: Optional[int] = None


class BatchPredictItemResult(BaseModel):
    prediction: int
    probs: List[float]


class BatchPredictResponse(BaseModel):
    results: List[BatchPredictItemResult]
    timings: dict


class ModelContext:
    def __init__(self):
        self.device = detect_device()
        self.local_ckpt_path = None
        self.tmp_dir = None
        self.params = None
        self.tokenizer = None
        self.model = None
        self.is_lora = False
        self.model_load_seconds = None

    def load(self):
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if not checkpoint_path:
            raise RuntimeError("Environment variable CHECKPOINT_PATH is required")
        merge_lora = os.environ.get("MERGE_LORA", "false").lower() in ("1", "true", "yes")

        # Load .env and optional HF auth
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

        # Ensure MLflow tracking URI if using databricks artifact schemes
        if (checkpoint_path.startswith("dbfs:/") or checkpoint_path.startswith("runs:/")) and not os.environ.get("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = "databricks"

        local_ckpt_path, tmp_dir = resolve_checkpoint_path(checkpoint_path)

        load_start_ts = time.time()
        tokenizer = load_tokenizer(local_ckpt_path, None)
        model, is_lora = load_model(
            checkpoint_path=local_ckpt_path,
            tokenizer=tokenizer,
            device=self.device,
            merge_lora=merge_lora,
        )
        load_end_ts = time.time()

        self.local_ckpt_path = local_ckpt_path
        self.tmp_dir = tmp_dir
        self.params = {"max_length": 128, "batch_size": int(os.environ.get("BATCH_SIZE", 16)), "num_labels": 3}
        self.tokenizer = tokenizer
        self.model = model
        self.is_lora = is_lora
        self.model_load_seconds = round(load_end_ts - load_start_ts, 4)


ctx = ModelContext()

app = FastAPI(title="MNLI Inference API", version="1.0.0")


@app.on_event("startup")
def on_startup():
    try:
        ctx.load()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": ctx.device}


@app.get("/info")
def info():
    return {
        "device": ctx.device,
        "is_lora": ctx.is_lora,
        "timings": {"model_load_seconds": ctx.model_load_seconds},
        "defaults": {
            "max_length": ctx.params.get("max_length", 128) if ctx.params else 128,
            "batch_size": ctx.params.get("batch_size", 16) if ctx.params else 16,
            "num_labels": ctx.params.get("num_labels", 3) if ctx.params else 3,
        },
    }


@app.post("/predict", response_model=SinglePredictResponse)
def predict(req: SinglePredictRequest):
    if ctx.model is None or ctx.tokenizer is None or ctx.params is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    max_length = req.max_length or ctx.params.get("max_length", 128)
    start_ts = time.time()
    pred, probs = predict_single(
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        premise=req.item.premise,
        hypothesis=req.item.hypothesis,
        max_length=max_length,
        device=ctx.device,
    )
    end_ts = time.time()
    return SinglePredictResponse(
        prediction=int(pred),
        probs=probs,
        timings={
            "model_load_seconds": ctx.model_load_seconds,
            "inference_seconds": round(end_ts - start_ts, 4),
        },
    )


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(req: BatchPredictRequest):
    if ctx.model is None or ctx.tokenizer is None or ctx.params is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    if not req.items:
        raise HTTPException(status_code=400, detail="'items' must be a non-empty list")

    max_length = req.max_length or ctx.params.get("max_length", 128)
    batch_size = req.batch_size or ctx.params.get("batch_size", 16)

    start_ts = time.time()
    preds, probs = predict_batch(
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        pairs=[item.model_dump() for item in req.items],
        max_length=max_length,
        device=ctx.device,
        batch_size=batch_size,
    )
    end_ts = time.time()

    results = [BatchPredictItemResult(prediction=int(preds[i]), probs=probs[i]) for i in range(len(preds))]
    return BatchPredictResponse(
        results=results,
        timings={
            "model_load_seconds": ctx.model_load_seconds,
            "inference_seconds": round(end_ts - start_ts, 4),
            "batch_size": batch_size,
            "num_items": len(results),
        },
    )


