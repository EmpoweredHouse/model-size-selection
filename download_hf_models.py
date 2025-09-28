import os
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv, find_dotenv
from huggingface_hub import snapshot_download, login as hf_login


def sanitize_slug(repo_id: str) -> str:
    # Prefer last path component (e.g., google/gemma-2b -> gemma-2b)
    slug = repo_id.strip().rstrip("/").split("/")[-1]
    return slug or repo_id.replace("/", "_")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def download_model(repo_id: str, target_dir: str, token: str | None) -> str:
    slug = sanitize_slug(repo_id)
    out_dir = os.path.join(target_dir, slug)
    ensure_dir(out_dir)
    # Download entire repository snapshot into out_dir (no symlinks for portability)
    snapshot_download(
        repo_id=repo_id,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        token=token,
        revision=None,
        ignore_patterns=None,
    )
    return out_dir


def main():
    try:
        load_dotenv(find_dotenv())
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Download HF models to local checkpoints directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemma-2b", "google/gemma-7b"],
        help="List of HF repo_ids to download",
    )
    parser.add_argument(
        "--target-dir",
        default="./checkpoints",
        help="Directory to place downloaded models (one subdir per model)",
    )
    args = parser.parse_args()

    ensure_dir(args.target_dir)

    # Optional HF login for gated/private models
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        try:
            hf_login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    for repo_id in args.models:
        try:
            out = download_model(repo_id=repo_id, target_dir=args.target_dir, token=token)
            print(f"Downloaded {repo_id} -> {out}")
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")


if __name__ == "__main__":
    main()


