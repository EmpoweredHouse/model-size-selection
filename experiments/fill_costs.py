import json
from typing import Dict, Any


HARD_CODED_PATH = "./experiments/costs.json"


def compute_costs_for_inference_entry(inference: Dict[str, Any]) -> None:
    cost_per_second_str = inference.get("gpu_cost_s_$", "0")
    time_1_seconds_str = inference.get("warm_1_in_single_avg", "0")
    time_100_seconds_str = inference.get("warm_100_in_batch_avg", "0")

    try:
        cost_per_second = float(cost_per_second_str)
        time_1_seconds = float(time_1_seconds_str)
        time_100_seconds = float(time_100_seconds_str)
    except ValueError:
        return

    cost_1 = cost_per_second * time_1_seconds
    cost_100 = cost_per_second * time_100_seconds

    inference["cost_1"] = f"{cost_1:.4f}"
    inference["cost_100"] = f"{cost_100:.4f}"


def main() -> None:
    with open(HARD_CODED_PATH, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    for model_obj in data.values():
        inference = model_obj.get("inference")
        if isinstance(inference, dict):
            compute_costs_for_inference_entry(inference)

    with open(HARD_CODED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Updated costs written to: {HARD_CODED_PATH}")


if __name__ == "__main__":
    main()

