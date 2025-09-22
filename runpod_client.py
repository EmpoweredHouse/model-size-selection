"""
RunPod Client
Client for communicating with RunPod serverless endpoints using the queue-based format.
"""

import argparse
import json
import os
import requests
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv


def build_headers(api_key: str) -> Dict[str, str]:
    """Build headers for RunPod API requests."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


def call_runpod_health(endpoint_url: str, api_key: str) -> Dict[str, Any]:
    """Check health via RunPod serverless."""
    headers = build_headers(api_key)
    payload = {"input": {"action": "health"}}
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_runpod_info(endpoint_url: str, api_key: str) -> Dict[str, Any]:
    """Get info via RunPod serverless."""
    headers = build_headers(api_key)
    payload = {"input": {"action": "info"}}
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_runpod_single(
    endpoint_url: str,
    api_key: str,
    premise: str,
    hypothesis: str,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Single prediction via RunPod serverless."""
    headers = build_headers(api_key)
    payload = {
        "input": {
            "action": "single",
            "item": {"premise": premise, "hypothesis": hypothesis}
        }
    }
    if max_length is not None:
        payload["input"]["max_length"] = max_length
    
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_runpod_batch(
    endpoint_url: str,
    api_key: str,
    items: List[Dict[str, str]],
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Batch prediction via RunPod serverless."""
    headers = build_headers(api_key)
    payload = {
        "input": {
            "action": "batch",
            "items": items
        }
    }
    if max_length is not None:
        payload["input"]["max_length"] = max_length
    if batch_size is not None:
        payload["input"]["batch_size"] = batch_size
    
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def read_jsonl(path: str) -> List[Dict[str, str]]:
    """Read JSONL file and extract premise/hypothesis pairs."""
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure only required keys are sent
            items.append({"premise": obj["premise"], "hypothesis": obj["hypothesis"]})
    return items


def main() -> None:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(description="RunPod Serverless Client for MNLI Inference")
    
    # Default to RunPod format URL
    default_url = os.environ.get("RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run")
    parser.add_argument("--endpoint_url", default=default_url,
                        help="RunPod endpoint URL (e.g., https://api.runpod.ai/v2/YOUR_ID/run)")
    
    # Require API key
    default_key = os.environ.get("RUNPOD_API_KEY")
    parser.add_argument("--api_key", required=(default_key is None), default=default_key,
                        help="RunPod API key (or set RUNPOD_API_KEY env)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Health check
    sp_health = subparsers.add_parser("health", help="Check health endpoint")

    # Info
    sp_info = subparsers.add_parser("info", help="Get server info and defaults")

    # Single prediction
    sp_single = subparsers.add_parser("single", help="Single prediction")
    sp_single.add_argument("--premise", required=True)
    sp_single.add_argument("--hypothesis", required=True)
    sp_single.add_argument("--max_length", type=int)

    # Batch prediction
    sp_batch = subparsers.add_parser("batch", help="Batch prediction from JSONL")
    sp_batch.add_argument("--jsonl", required=True, help="Path to JSONL with fields premise,hypothesis")
    sp_batch.add_argument("--max_length", type=int)
    sp_batch.add_argument("--batch_size", type=int)

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api_key or set RUNPOD_API_KEY env.")

    if args.command == "health":
        data = call_runpod_health(args.endpoint_url, args.api_key)
        print(json.dumps(data, indent=2))
        return

    if args.command == "info":
        data = call_runpod_info(args.endpoint_url, args.api_key)
        print(json.dumps(data, indent=2))
        return

    if args.command == "single":
        data = call_runpod_single(
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            premise=args.premise,
            hypothesis=args.hypothesis,
            max_length=args.max_length,
        )
        print(json.dumps(data, indent=2))
        return

    if args.command == "batch":
        items = read_jsonl(args.jsonl)
        data = call_runpod_batch(
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            items=items,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        print(json.dumps(data, indent=2))
        return


if __name__ == "__main__":
    main()
    # Example usage:
    # python runpod_client.py --endpoint_url https://api.runpod.ai/v2/43spixl766m91d/run --api_key $RUNPOD_API_KEY health
    # python runpod_client.py single --premise "A soccer game with multiple males playing." --hypothesis "Some men are playing a sport."
    # python runpod_client.py batch --jsonl batch.jsonl --batch_size 16
