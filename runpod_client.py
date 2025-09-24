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
    import time
    start_time = time.time()
    
    headers = build_headers(api_key)
    payload = {"input": {"action": "health"}}
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    
    end_time = time.time()
    result = resp.json()
    result["client_timing"] = {
        "request_duration_seconds": round(end_time - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    }
    return result


def call_runpod_info(endpoint_url: str, api_key: str) -> Dict[str, Any]:
    """Get info via RunPod serverless."""
    import time
    start_time = time.time()
    
    headers = build_headers(api_key)
    payload = {"input": {"action": "info"}}
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    
    end_time = time.time()
    result = resp.json()
    result["client_timing"] = {
        "request_duration_seconds": round(end_time - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    }
    return result


def call_runpod_single(
    endpoint_url: str,
    api_key: str,
    premise: str,
    hypothesis: str,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Single prediction via RunPod serverless."""
    import time
    start_time = time.time()
    
    headers = build_headers(api_key)
    payload = {
        "input": {
            "action": "single",
            "item": {"premise": premise, "hypothesis": hypothesis}
        }
    }
    if max_length is not None:
        payload["input"]["max_length"] = max_length
    
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    
    end_time = time.time()
    result = resp.json()
    result["client_timing"] = {
        "request_duration_seconds": round(end_time - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    }
    return result


def call_runpod_batch(
    endpoint_url: str,
    api_key: str,
    items: List[Dict[str, str]],
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Batch prediction via RunPod serverless."""
    import time
    start_time = time.time()
    
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
    
    end_time = time.time()
    result = resp.json()
    result["client_timing"] = {
        "request_duration_seconds": round(end_time - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "num_items": len(items)
    }
    return result


def check_job_status(endpoint_url: str, api_key: str, job_id: str) -> Dict[str, Any]:
    """Check the status of a RunPod job by ID."""
    import time
    start_time = time.time()
    
    # Convert runsync URL to status URL
    status_url = endpoint_url.replace("/runsync", f"/status/{job_id}")
    headers = build_headers(api_key)
    
    resp = requests.get(status_url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    end_time = time.time()
    result = resp.json()
    result["client_timing"] = {
        "request_duration_seconds": round(end_time - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    }
    return result


def poll_for_result(endpoint_url: str, api_key: str, job_id: str, max_wait_time: int = 600, poll_interval: int = 1, original_timing: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Poll RunPod job status until completion or timeout.
    
    Args:
        endpoint_url: RunPod endpoint URL
        api_key: RunPod API key
        job_id: Job ID to poll
        max_wait_time: Maximum time to wait in seconds (default: 10 minutes)
        poll_interval: Time between polls in seconds (default: 1 second)
        original_timing: Original timing data from initial request
    
    Returns:
        Final job result when completed, with updated timing information
    """
    import time
    polling_start_time = time.time()
    
    print(f"🔄 Polling for job result (ID: {job_id})")
    print(f"   Max wait time: {max_wait_time}s, Poll interval: {poll_interval}s")
    
    while time.time() - polling_start_time < max_wait_time:
        try:
            status_result = check_job_status(endpoint_url, api_key, job_id)
            
            job_status = status_result.get("status", "UNKNOWN")
            print(f"   Status: {job_status} (elapsed: {time.time() - polling_start_time:.1f}s)", end="\r")
            
            if job_status == "COMPLETED":
                polling_end_time = time.time()
                polling_duration = polling_end_time - polling_start_time
                print(f"\n✅ Job completed after {polling_duration:.1f}s of polling")
                
                # Update timing information to include total end-to-end time
                if original_timing:
                    total_duration = original_timing["request_duration_seconds"] + polling_duration
                    status_result["client_timing"] = {
                        "request_duration_seconds": round(total_duration, 4),
                        "initial_request_seconds": original_timing["request_duration_seconds"],
                        "polling_duration_seconds": round(polling_duration, 4),
                        "timestamp": original_timing["timestamp"]
                    }
                else:
                    # If no original timing, just use polling time
                    status_result["client_timing"] = {
                        "request_duration_seconds": round(polling_duration, 4),
                        "polling_duration_seconds": round(polling_duration, 4),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(polling_start_time))
                    }
                
                return status_result
                
            elif job_status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                polling_end_time = time.time()
                polling_duration = polling_end_time - polling_start_time
                print(f"\n❌ Job failed with status: {job_status} after {polling_duration:.1f}s")
                
                # Update timing even for failed jobs
                if original_timing:
                    total_duration = original_timing["request_duration_seconds"] + polling_duration
                    status_result["client_timing"] = {
                        "request_duration_seconds": round(total_duration, 4),
                        "initial_request_seconds": original_timing["request_duration_seconds"],
                        "polling_duration_seconds": round(polling_duration, 4),
                        "timestamp": original_timing["timestamp"]
                    }
                
                return status_result
                
            elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                # Continue polling
                time.sleep(poll_interval)
            else:
                print(f"\n⚠️ Unknown status: {job_status}, continuing to poll...")
                time.sleep(poll_interval)
                
        except Exception as e:
            print(f"\n❌ Error polling job status: {e}")
            time.sleep(poll_interval)
    
    polling_end_time = time.time()
    polling_duration = polling_end_time - polling_start_time
    print(f"\n⏰ Polling timed out after {max_wait_time}s")
    
    # Return last known status with timing info
    try:
        final_result = check_job_status(endpoint_url, api_key, job_id)
        if original_timing:
            total_duration = original_timing["request_duration_seconds"] + polling_duration
            final_result["client_timing"] = {
                "request_duration_seconds": round(total_duration, 4),
                "initial_request_seconds": original_timing["request_duration_seconds"],
                "polling_duration_seconds": round(polling_duration, 4),
                "timestamp": original_timing["timestamp"]
            }
        return final_result
    except:
        return {
            "status": "TIMEOUT", 
            "error": "Polling timed out",
            "client_timing": {
                "request_duration_seconds": round(polling_duration, 4),
                "polling_duration_seconds": round(polling_duration, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(polling_start_time))
            }
        }


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


def run_inference(
    endpoint_url: str,
    command: str,
    premise: str | None = None,
    hypothesis: str | None = None,
    max_length: int | None = None,
    jsonl_file: str | None = None,
    batch_size: int | None = None,
) -> None:
    load_dotenv(find_dotenv())
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("❌ Please set RUNPOD_API_KEY in .env file or edit API_KEY in this script")
        return

    if command == "health":
        data = call_runpod_health(endpoint_url, api_key)
        print("✅ Health Check Result:")
        print(json.dumps(data, indent=2))
        return

    elif command == "info":
        data = call_runpod_info(endpoint_url, api_key)
        print("ℹ️ Server Info:")
        print(json.dumps(data, indent=2))
        return

    elif command == "single":
        print(f"📝 Testing single prediction:")
        print(f"   Premise: {premise}")
        print(f"   Hypothesis: {hypothesis}")
        print()
        
        data = call_runpod_single(
            endpoint_url=endpoint_url,
            api_key=api_key,
            premise=premise,
            hypothesis=hypothesis,
            max_length=max_length,
        )
        
        # Check if job is queued/in progress and poll for result
        if data.get("status") in ["IN_QUEUE", "IN_PROGRESS"]:
            job_id = data.get("id")
            if job_id:
                print(f"⏳ Job queued (ID: {job_id}), polling for result...")
                original_timing = data.get("client_timing", {})
                data = poll_for_result(endpoint_url, api_key, job_id, original_timing=original_timing)
        
        print("🎯 Prediction Result:")
        print(json.dumps(data, indent=2))
        
        # Only print timing info if we have complete results
        if data.get("status") == "COMPLETED" and "executionTime" in data:
            timing_info = [
                f"Delay Time: {data['delayTime'] / 1000:.4g}s",
                f"Execution Time: {data['executionTime'] / 1000:.4g}s",
                f"Model Load Time: {data['output']['timings']['model_load_seconds']:.4g}s",
                f"Inference Time: {data['output']['timings']['inference_seconds']:.4g}s"
            ]
            
            client_timing = data.get('client_timing', {})
            if 'initial_request_seconds' in client_timing:
                # Job was queued and polled
                timing_info.extend([
                    f"Initial Request Time: {client_timing['initial_request_seconds']:.4g}s",
                    f"Polling Time: {client_timing['polling_duration_seconds']:.4g}s",
                    f"Total End-to-End Time: {client_timing['request_duration_seconds']:.4g}s"
                ])
            else:
                # Direct response (no polling)
                timing_info.append(f"Total Time: {client_timing.get('request_duration_seconds', 0):.4g}s")
            
            print("\n".join(timing_info))
        else:
            print(f"Status: {data.get('status', 'UNKNOWN')}")
            client_timing = data.get("client_timing", {})
            if client_timing.get("request_duration_seconds"):
                if 'initial_request_seconds' in client_timing:
                    print(f"Initial Request Time: {client_timing['initial_request_seconds']:.4g}s")
                    print(f"Polling Time: {client_timing['polling_duration_seconds']:.4g}s")
                    print(f"Total Time: {client_timing['request_duration_seconds']:.4g}s")
                else:
                    print(f"Total Time: {client_timing['request_duration_seconds']:.4g}s")
        
        return data

    elif command == "batch":
        print(f"📋 Testing batch prediction from: {jsonl_file}")
        try:
            items = read_jsonl(jsonl_file)
            print(f"   Found {len(items)} items")
            print()
            
            data = call_runpod_batch(
                endpoint_url=endpoint_url,
                api_key=api_key,
                items=items,
                max_length=max_length,
                batch_size=batch_size,
            )
            
            # Check if job is queued/in progress and poll for result
            if data.get("status") in ["IN_QUEUE", "IN_PROGRESS"]:
                job_id = data.get("id")
                if job_id:
                    print(f"⏳ Batch job queued (ID: {job_id}), polling for result...")
                    original_timing = data.get("client_timing", {})
                    data = poll_for_result(endpoint_url, api_key, job_id, original_timing=original_timing)
            
            print("📊 Batch Results:")
            print(json.dumps(data, indent=2))
            
            # Only print timing info if we have complete results
            if data.get("status") == "COMPLETED" and "executionTime" in data:
                timing_info = [
                    f"Delay Time: {data['delayTime'] / 1000:.4g}s",
                    f"Execution Time: {data['executionTime'] / 1000:.4g}s",
                    f"Model Load Time: {data['output']['timings']['model_load_seconds']:.4g}s",
                    f"Inference Time: {data['output']['timings']['inference_seconds']:.4g}s"
                ]
                
                client_timing = data.get('client_timing', {})
                if 'initial_request_seconds' in client_timing:
                    # Job was queued and polled
                    timing_info.extend([
                        f"Initial Request Time: {client_timing['initial_request_seconds']:.4g}s",
                        f"Polling Time: {client_timing['polling_duration_seconds']:.4g}s",
                        f"Total End-to-End Time: {client_timing['request_duration_seconds']:.4g}s"
                    ])
                else:
                    # Direct response (no polling)
                    timing_info.append(f"Total Time: {client_timing.get('request_duration_seconds', 0):.4g}s")
                
                # Add per-item timing
                num_items = client_timing.get('num_items') or len(items)
                if num_items > 0:
                    timing_info.append(f"Per Item Time: {client_timing.get('request_duration_seconds', 0) / num_items:.4g}s")
                
                print("\n".join(timing_info))
            else:
                print(f"Status: {data.get('status', 'UNKNOWN')}")
                client_timing = data.get("client_timing", {})
                if client_timing.get("request_duration_seconds"):
                    if 'initial_request_seconds' in client_timing:
                        print(f"Initial Request Time: {client_timing['initial_request_seconds']:.4g}s")
                        print(f"Polling Time: {client_timing['polling_duration_seconds']:.4g}s")
                        print(f"Total Time: {client_timing['request_duration_seconds']:.4g}s")
                    else:
                        print(f"Total Time: {client_timing['request_duration_seconds']:.4g}s")
                    
                    # Add per-item timing for incomplete jobs too
                    num_items = client_timing.get('num_items') or len(items)
                    if num_items > 0:
                        print(f"Per Item Time: {client_timing['request_duration_seconds'] / num_items:.4g}s")
            
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {jsonl_file}")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
            return

    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: health, info, single, batch")


def calculate_metrics(data_list):
    """Calculate average metrics from a list of response data."""
    # Filter out incomplete/failed jobs
    completed_data = [data for data in data_list if data.get("status") == "COMPLETED" and "executionTime" in data]
    
    if not completed_data:
        # Return empty metrics if no completed jobs
        return {
            "delay_time": 0.0,
            "execution_time": 0.0,
            "model_load_time": 0.0,
            "inference_time": 0.0,
            "total_time": 0.0
        }
    
    delay_times = [data['delayTime'] / 1000 for data in completed_data]
    execution_times = [data['executionTime'] / 1000 for data in completed_data]
    model_load_times = [data['output']['timings']['model_load_seconds'] for data in completed_data]
    inference_times = [data['output']['timings']['inference_seconds'] for data in completed_data]
    total_times = [data['client_timing']['request_duration_seconds'] for data in completed_data]
    
    return {
        "delay_time": float(f"{sum(delay_times) / len(delay_times):.4g}"),
        "execution_time": float(f"{sum(execution_times) / len(execution_times):.4g}"),
        "model_load_time": float(f"{sum(model_load_times) / len(model_load_times):.4g}"),
        "inference_time": float(f"{sum(inference_times) / len(inference_times):.4g}"),
        "total_time": float(f"{sum(total_times) / len(total_times):.4g}")
    }


def run_benchmark(endpoint_url: str, file_name: str = ""):
    """Run complete benchmark: single cold, single warm avg, batch cold, batch warm avg."""
    print("🚀 Starting RunPod Serverless Benchmark")
    print("=" * 50)
    
    # Initialize metrics structure
    all_metrics = {
        "single_cold": {},
        "single_50_avg": {},
        "batch100_cold": {},
        "batch100_50_avg": {}
    }
    
    # 1. Single inference - Cold start (first call)
    print("\n📊 1/4: Single inference (Cold start)")
    data = run_inference(
        endpoint_url=endpoint_url,
        command="single",
        premise="A soccer game with multiple males playing.",
        hypothesis="Some men are playing a sport.",
        max_length=128,
        jsonl_file="examples.jsonl",
        batch_size=16,
    )
    all_metrics["single_cold"] = calculate_metrics([data])
    print(f"✅ Cold start completed - Total time: {data['client_timing']['request_duration_seconds']:.3f}s")
    
    # 2. Single inference - 50 warm calls (average)
    print("\n📊 2/4: Single inference x50 (Warm calls average)")
    warm_single_data = []
    for i in range(50):
        print(f"   Progress: {i+1}/50", end="\r")
        data = run_inference(
            endpoint_url=endpoint_url,
            command="single",
            premise="A soccer game with multiple males playing.",
            hypothesis="Some men are playing a sport.",
            max_length=128,
            jsonl_file="examples.jsonl",
            batch_size=16,
        )
        warm_single_data.append(data)
    
    all_metrics["single_50_avg"] = calculate_metrics(warm_single_data)
    avg_time = all_metrics["single_50_avg"]["total_time"]
    print(f"\n✅ Warm single calls completed - Average time: {avg_time:.3f}s")
    
    # 3. Sleep 20 seconds to potentially trigger cold start
    print("\n⏳ Sleeping 20 seconds to trigger potential cold start...")
    time.sleep(20)
    
    # 4. Batch inference - Cold start (first call after sleep)
    print("\n📊 3/4: Batch inference x100 (Cold start)")
    items = read_jsonl("examples.jsonl")[:100]  # Take first 100 items
    data = run_inference(
        endpoint_url=endpoint_url,
        command="batch",
        premise=None,
        hypothesis=None,
        max_length=128,
        jsonl_file="examples.jsonl",
        batch_size=16,
    )
    all_metrics["batch100_cold"] = calculate_metrics([data])
    print(f"✅ Batch cold start completed - Total time: {data['client_timing']['request_duration_seconds']:.3f}s")
    
    # 5. Batch inference - 50 warm calls (average)
    print("\n📊 4/4: Batch inference x100, repeated 50 times (Warm calls average)")
    warm_batch_data = []
    for i in range(50):
        print(f"   Progress: {i+1}/50", end="\r")
        data = run_inference(
            endpoint_url=endpoint_url,
            command="batch",
            premise=None,
            hypothesis=None,
            max_length=128,
            jsonl_file="examples.jsonl",
            batch_size=16,
        )
        warm_batch_data.append(data)
    
    all_metrics["batch100_50_avg"] = calculate_metrics(warm_batch_data)
    avg_batch_time = all_metrics["batch100_50_avg"]["total_time"]
    print(f"\n✅ Warm batch calls completed - Average time: {avg_batch_time:.3f}s")
    
    # Save metrics to file
    output_file = file_name if file_name else "metrics.json"
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n📈 Summary:")
    print(f"Single Cold:     {all_metrics['single_cold']['total_time']:.3f}s")
    print(f"Single Warm Avg: {all_metrics['single_50_avg']['total_time']:.3f}s")
    print(f"Batch Cold:      {all_metrics['batch100_cold']['total_time']:.3f}s")
    print(f"Batch Warm Avg:  {all_metrics['batch100_50_avg']['total_time']:.3f}s")
    
    return all_metrics


if __name__ == "__main__":
    run_benchmark(
        endpoint_url="https://api.runpod.ai/v2/ura3uhimh3wrmc/runsync",
        file_name="latency/distilbert_full.json"
    )