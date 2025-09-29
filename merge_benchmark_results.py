import os
import re
import json
import argparse
from collections import defaultdict


def extract_iter_index(filename: str) -> int:
    m = re.search(r"_iter_(\d+)\.json$", filename)
    if not m:
        return -1
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser(description="Merge per-iteration benchmark JSONs into a single file")
    parser.add_argument("--results_dir", default="results", help="Directory with per-iteration JSONs")
    parser.add_argument("--output", default="merged_benchmark.json", help="Output JSON file")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    files = [f for f in os.listdir(args.results_dir) if f.endswith(".json")]
    if not files:
        raise SystemExit(f"No JSON files found in {args.results_dir}")

    # Group by run_name; keep (iter_index, record) to preserve order
    grouped = defaultdict(list)

    for fname in files:
        path = os.path.join(args.results_dir, fname)
        try:
            data = json.load(open(path, "r"))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
        if not isinstance(data, list) or not data:
            print(f"Skipping {fname}: not a list with a record")
            continue
        rec = data[0]
        run_name = rec.get("run_name")
        if not run_name:
            print(f"Skipping {fname}: missing run_name")
            continue
        iter_idx = extract_iter_index(fname)
        grouped[run_name].append((iter_idx, rec))

    # Build merged records per run_name
    merged = []

    for run_name, items in grouped.items():
        # Sort by iter index; unknown (-1) go last in filename order
        items.sort(key=lambda x: (x[0],))
        times = []
        times_total = []

        # Use the first record as a template for static fields
        template = items[0][1]
        out = {
            "run_name": run_name,
            "checkpoint_path": template.get("checkpoint_path"),
            "merge_lora": template.get("merge_lora"),
        }
        if template.get("lora_merge_time") is not None:
            out["lora_merge_time"] = template.get("lora_merge_time")
        if template.get("merged_checkpoint_path") is not None:
            out["merged_checkpoint_path"] = template.get("merged_checkpoint_path")
        if template.get("device") is not None:
            out["device"] = template.get("device")

        for _, rec in items:
            # Prefer single-value fields if present; fallback to arrays
            if isinstance(rec.get("time"), (int, float)):
                times.append(float(rec["time"]))
            elif rec.get("times") and isinstance(rec["times"], list) and len(rec["times"]) > 0:
                times.append(float(rec["times"][0]))

            if isinstance(rec.get("total_time"), (int, float)):
                times_total.append(float(rec["total_time"]))
            elif rec.get("times_total") and isinstance(rec["times_total"], list) and len(rec["times_total"]) > 0:
                times_total.append(float(rec["times_total"][0]))

        out["times"] = times
        out["times_total"] = times_total
        out["avg_time"] = round(sum(times) / len(times), 6) if times else None
        out["avg_total_time"] = round(sum(times_total) / len(times_total), 6) if times_total else None
        merged.append(out)

    # Keep deterministic order by run_name
    merged.sort(key=lambda x: x["run_name"])

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote merged results to {args.output}")


if __name__ == "__main__":
    main()


