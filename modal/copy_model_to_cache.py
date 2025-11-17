"""
Modal script to copy a model from distillation-runs volume to sdar-models-cache volume
and fill missing files from HuggingFace.

Usage:
    # Copy slerp model with defaults (auto-fills missing files from JetLM/SDAR-4B-Chat)
    modal run copy_model_to_cache.py

    # Copy with custom paths
    modal run copy_model_to_cache.py --source-path /distillation-runs/workspaces/mergekit/results/your-model --dest-name your-model

    # List all cached models
    modal run copy_model_to_cache.py --list-models true

    # Force overwrite existing cache
    modal run copy_model_to_cache.py --force true
"""
import modal
import os
import shutil
from pathlib import Path

app = modal.App("copy-model-to-cache")

# Source volume (distillation runs)
source_volume = modal.Volume.from_name("distillation-runs", create_if_missing=False)

# Destination volume (model cache for benchmarks)
dest_volume = modal.Volume.from_name("sdar-models-cache", create_if_missing=True)

# Simple image with just huggingface_hub
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub",
    "tqdm",
)

SOURCE_DIR = "/distillation-runs"
MODELS_DIR = "/models"


@app.function(
    image=image,
    volumes={
        SOURCE_DIR: source_volume,
        MODELS_DIR: dest_volume,
    },
    timeout=3600,  # 1 hour
    memory=16384,  # 16GB RAM
)
def copy_and_fill_model(
    source_path: str,
    dest_name: str,
    hf_model: str = "JetLM/SDAR-4B-Chat",
    force: bool = False,
):
    """
    Copy a model from source_path to /models/dest_name and fill missing files from HuggingFace.

    Args:
        source_path: Path to the source model directory (on Modal)
        dest_name: Name for the destination directory in /models
        hf_model: HuggingFace model ID to use for filling missing files
        force: If True, overwrite existing destination directory
    """
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    print("=" * 80)
    print("Model Copy & Fill Utility")
    print("=" * 80)
    print(f"Source: {source_path}")
    print(f"Destination: {MODELS_DIR}/{dest_name}")
    print(f"HuggingFace fallback: {hf_model}")
    print("=" * 80)

    dest_path = f"{MODELS_DIR}/{dest_name}"

    # Check if destination exists
    if os.path.exists(dest_path):
        if not force:
            print(f"\n⚠️  Destination already exists: {dest_path}")
            print("Use --force to overwrite")
            return {"status": "skipped", "reason": "destination exists"}
        else:
            print(f"\n🗑️  Removing existing destination: {dest_path}")
            shutil.rmtree(dest_path)

    # Check if source exists
    if not os.path.exists(source_path):
        print(f"\n❌ Source path does not exist: {source_path}")
        return {"status": "error", "reason": "source not found"}

    # Step 1: Copy source model to destination
    print(f"\n[1/4] Copying model from {source_path}...")
    os.makedirs(dest_path, exist_ok=True)

    source_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.relpath(src_file, source_path)
            dest_file = os.path.join(dest_path, rel_path)

            # Create directory if needed
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)

            # Copy file
            shutil.copy2(src_file, dest_file)
            source_files.append(rel_path)
            print(f"  ✓ Copied: {rel_path}")

    print(f"\n📦 Copied {len(source_files)} files from source model")

    # Step 2: Download HuggingFace model to temp location
    print(f"\n[2/4] Downloading {hf_model} from HuggingFace...")
    temp_hf_path = "/tmp/hf_model"
    if os.path.exists(temp_hf_path):
        shutil.rmtree(temp_hf_path)

    snapshot_download(
        repo_id=hf_model,
        local_dir=temp_hf_path,
        local_dir_use_symlinks=False,
    )
    print(f"  ✓ Downloaded to {temp_hf_path}")

    # Step 3: Find missing files
    print(f"\n[3/4] Comparing file lists...")
    hf_files = []
    for root, dirs, files in os.walk(temp_hf_path):
        for file in files:
            hf_file = os.path.join(root, file)
            rel_path = os.path.relpath(hf_file, temp_hf_path)
            hf_files.append(rel_path)

    missing_files = set(hf_files) - set(source_files)

    # Filter out model weight files (we don't want to overwrite those)
    # Keep: config files, tokenizer files, custom modeling files
    important_missing = []
    for f in missing_files:
        # Skip large weight files
        if any(pattern in f for pattern in ['.safetensors', '.bin', '.pth', '.pt']):
            if not any(keyword in f for keyword in ['config', 'tokenizer', 'generation']):
                continue

        # Skip cache directories and metadata files
        if any(skip in f for skip in ['.cache/', '.metadata', '.lock', '.gitattributes', '.gitignore']):
            continue

        important_missing.append(f)

    if not important_missing:
        print(f"\n✅ No missing files! Model is complete.")
    else:
        print(f"\n📋 Found {len(important_missing)} missing files:")
        for f in sorted(important_missing):
            print(f"  - {f}")

    # Step 4: Copy missing files
    if important_missing:
        print(f"\n[4/4] Copying {len(important_missing)} missing files from HuggingFace model...")
        copied_count = 0
        for rel_path in important_missing:
            src_file = os.path.join(temp_hf_path, rel_path)
            dest_file = os.path.join(dest_path, rel_path)

            # Create directory if needed
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)

            # Copy file
            shutil.copy2(src_file, dest_file)
            print(f"  ✓ Copied: {rel_path}")
            copied_count += 1

        print(f"\n📦 Copied {copied_count} missing files")

    # Commit volume changes
    print(f"\n💾 Committing changes to sdar-models-cache volume...")
    dest_volume.commit()
    print(f"✅ Volume committed successfully")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Model copied to: {dest_path}")
    print(f"📁 Total files in source: {len(source_files)}")
    print(f"📥 Files filled from HF: {len(important_missing)}")
    print(f"📦 Total files in destination: {len(source_files) + len(important_missing)}")
    print("=" * 80)

    return {
        "status": "success",
        "source_files": len(source_files),
        "filled_files": len(important_missing),
        "total_files": len(source_files) + len(important_missing),
        "destination": dest_path,
    }


@app.function(
    image=image,
    volumes={MODELS_DIR: dest_volume},
)
def list_cached_models():
    """List all models in the cache."""
    print("=" * 80)
    print("Cached Models in /models")
    print("=" * 80)

    if not os.path.exists(MODELS_DIR):
        print("No models cache found")
        return []

    models = []
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path):
            # Count files
            file_count = sum(1 for _, _, files in os.walk(item_path) for _ in files)
            size_mb = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(item_path)
                for file in files
            ) / (1024 * 1024)

            models.append({
                "name": item,
                "files": file_count,
                "size_mb": round(size_mb, 2),
            })
            print(f"\n📦 {item}")
            print(f"   Files: {file_count}")
            print(f"   Size: {size_mb:.2f} MB")

    print("\n" + "=" * 80)
    print(f"Total: {len(models)} models")
    print("=" * 80)

    return models


@app.local_entrypoint()
def main(
    source_path: str = "/distillation-runs/workspaces/mergekit/results/slerp_sdar-4b_qwen3-4b",
    dest_name: str = "slerp_sdar-4b_qwen3-4b",
    hf_model: str = "JetLM/SDAR-4B-Chat",
    force: bool = False,
    list_models: bool = False,
):
    """
    Copy a model to the cache and fill missing files.

    Args:
        source_path: Path to source model in distillation-runs volume (default: slerp model)
        dest_name: Destination name in /models (sdar-models-cache volume)
        hf_model: HuggingFace model for missing files
        force: Overwrite existing destination
        list_models: Just list cached models
    """
    if list_models:
        list_cached_models.remote()
    else:
        result = copy_and_fill_model.remote(
            source_path=source_path,
            dest_name=dest_name,
            hf_model=hf_model,
            force=force,
        )
        print(f"\n📊 Result: {result}")
