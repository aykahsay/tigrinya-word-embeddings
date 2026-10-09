"""
Hugging Face Publishing Script for Tigrinya Skip-gram Word Embeddings.
Used to upload finished model artifacts and model card to the Hugging Face Hub.

Prerequisites:
    Run `huggingface-cli login` in your terminal or set the HF_TOKEN environment variable.
"""

import os
import argparse


def publish_to_hub(repo_id="aykgeh/tigrinya-skipgram-embeddings", private=False):
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return False

    api = HfApi()

    print(f"Connecting to Hugging Face Hub...")
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info.get('name')}")
    except Exception as e:
        print(f"Authentication error: {e}")
        print("Please authenticate securely by running: huggingface-cli login")
        return False

    print(f"Ensuring repository exists: {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Could not create repository {repo_id}: {e}")
        return False

    files_to_upload = [
        ("docs/model_card.md", "README.md"),
        ("artifacts/tigrinya_skipgram_model.json", "tigrinya_skipgram_model.json"),
        ("reports/training_metrics.json", "training_metrics.json"),
        ("reports/baseline_comparison.json", "baseline_comparison.json"),
        ("data/LICENSE_DATA.txt", "LICENSE"),
    ]

    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            print(f"Uploading {local_path} -> {repo_path}...")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded {repo_path}.")
        else:
            print(f"Warning: {local_path} not found.")

    print(f"\nModel publication complete! View at: https://huggingface.co/{repo_id}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="aykgeh/tigrinya-skipgram-embeddings",
        help="Target Hugging Face repository ID (e.g. aykgeh/tigrinya-skipgram-embeddings)",
    )
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    publish_to_hub(repo_id=args.repo_id, private=args.private)
