"""
Quick script to copy config.json from slerp model to passthrough model in Modal cache.
"""
import modal
import shutil

app = modal.App("fix-passthrough-config")

# Use the same volume as benchmarks
models_volume = modal.Volume.from_name("sdar-models-cache", create_if_missing=True)

MODELS_DIR = "/models"

@app.function(
    volumes={MODELS_DIR: models_volume},
    timeout=300,
)
def copy_config():
    import os

    source_model = f"{MODELS_DIR}/slerp_sdar-4b_qwen3-4b"
    target_model = f"{MODELS_DIR}/passthrough_sdar-4b_qwen3-4b-v2"

    source_config = f"{source_model}/config.json"
    target_config = f"{target_model}/config.json"

    print(f"Source model dir exists: {os.path.exists(source_model)}")
    print(f"Target model dir exists: {os.path.exists(target_model)}")
    print(f"Source config exists: {os.path.exists(source_config)}")

    if os.path.exists(source_config):
        # Backup original if it exists
        if os.path.exists(target_config):
            backup = f"{target_config}.backup"
            print(f"Backing up original config to {backup}")
            shutil.copy2(target_config, backup)

        # Copy config
        print(f"Copying {source_config} -> {target_config}")
        shutil.copy2(source_config, target_config)

        # Verify
        print(f"✓ Config copied successfully")

        # Show the model_type from new config
        import json
        with open(target_config, 'r') as f:
            config = json.load(f)
            print(f"Model type in new config: {config.get('model_type', 'NOT FOUND')}")

        # Commit volume changes
        models_volume.commit()
        print("✓ Volume changes committed")
    else:
        print(f"✗ Source config not found: {source_config}")
        print("\nAvailable models:")
        if os.path.exists(MODELS_DIR):
            for item in os.listdir(MODELS_DIR):
                print(f"  - {item}")

@app.local_entrypoint()
def main():
    copy_config.remote()
    print("\nDone! Config copied from slerp to passthrough model.")
