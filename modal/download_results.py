"""
Download GSM8K benchmark results from Modal volume to local.

Usage:
    modal run download_results.py
"""
import modal

app = modal.App("download-gsm8k-results")

# Results volume
results_volume = modal.Volume.from_name("sdar-results", create_if_missing=False)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: results_volume},
)
def download_file(model_folder: str, results_filename: str) -> dict:
    """
    Download results file from Modal volume.
    """
    import os
    import json

    results_path = f"{RESULTS_DIR}/{model_folder}/{results_filename}"

    # List available files if not found
    if not os.path.exists(results_path):
        print(f"Error: File not found: {results_path}")
        print(f"\nLooking for files in {RESULTS_DIR}...")

        if os.path.exists(RESULTS_DIR):
            for root, dirs, files in os.walk(RESULTS_DIR):
                level = root.replace(RESULTS_DIR, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f'{subindent}{file}')
        return None

    # Read and return the file
    with open(results_path, 'r') as f:
        data = json.load(f)

    print(f"Successfully loaded: {results_path}")
    print(f"Keys in data: {list(data.keys())}")
    print(f"Number of results: {len(data.get('results', []))}")

    return data


@app.local_entrypoint()
def main(
    model_folder: str = "JetLM_SDAR-4B-Chat",
    results_filename: str = "gsm8k_JetLM_SDAR-4B-Chat_20251117_151516.json",
    output_file: str = "downloaded_results.json",
):
    """
    Download results from Modal volume to local file.
    """
    print(f"Downloading: {model_folder}/{results_filename}")
    print()

    data = download_file.remote(model_folder, results_filename)

    if data:
        # Save locally
        import json
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ File downloaded to: {output_file}")
        print(f"  Config keys: {list(data.get('config', {}).keys())}")
        print(f"  Metrics: {data.get('metrics', {})}")
        print(f"  Results count: {len(data.get('results', []))}")

        # Show first result structure
        if data.get('results'):
            print(f"\n  First result keys: {list(data['results'][0].keys())}")
    else:
        print("\n✗ Failed to download file")
