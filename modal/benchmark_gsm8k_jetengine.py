"""
Modal script for benchmarking SDAR-4B model on GSM8K dataset using JetEngine.

Usage:
    modal run benchmark_gsm8k_jetengine.py --model-name JetLM/SDAR-4B-Chat --num-samples 100
    modal run benchmark_gsm8k_jetengine.py --help
"""
import modal
import json
from pathlib import Path

# Define Modal app
app = modal.App("sdar-4b-gsm8k-jetengine")

CUDA_TAG = "12.9.1-devel-ubuntu22.04"  # CUDA 12.9 devel image

# Create Modal image with NVIDIA CUDA devel base for flash-attn compilation
image = (
    modal.Image
    # 1) Start from NVIDIA CUDA "devel" image so we have nvcc + full toolkit
    .from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .entrypoint([])  # optional: silence base image entrypoint
    # 2) Basic build tooling + git
    .apt_install("git", "build-essential")
    # Make sure CUDA_HOME is explicitly set (usually already correct, but safe)
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # 3) Install PyTorch 2.8.0 compiled for CUDA 12.9 (cu129 wheels)
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
        extra_options="--index-strategy unsafe-best-match",
    )
    # 4) Your other Python deps
    .uv_pip_install(
        "transformers==4.52.4",
        "datasets>=2.16.0",
        "accelerate>=1.3.0",
        "huggingface_hub",
        "einops",
        "numpy<2.0.0",
        "tqdm",
        "packaging",
        "ninja",
        "wheel",
    )
    # 5) Install flash-attn AFTER torch is in place
    .uv_pip_install(
        "flash-attn==2.8.3",       # pin to a recent, torch2.8-compatible version
        extra_options="--no-build-isolation",
    )
    # Optional: quick sanity check during build
    .run_commands(
        "python -c \"import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda); import flash_attn; print('flash_attn OK')\""
    )
    # 6) Install JetEngine from local directory (copy=True to allow subsequent build steps)
    .add_local_dir("thirdparty/JetEngine", remote_path="/root/JetEngine", copy=True)
    .run_commands(
        "pip install /root/JetEngine",
    )
    # 7) Your local helper file (added last for fast redeployment without rebuild)
    .add_local_file("benchmark_utils.py", remote_path="/root/benchmark_utils.py")
)

# Volumes for caching models and storing results
models_volume = modal.Volume.from_name("sdar-models-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("sdar-results", create_if_missing=True)

# Constants
MODELS_DIR = "/models"
RESULTS_DIR = "/results"


@app.function(
    image=image,
    gpu="H100",
    volumes={
        MODELS_DIR: models_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=7200,  # 2 hours
    memory=65536,  # 64GB RAM
)
def run_gsm8k_benchmark(
    model_name: str = "JetLM/SDAR-4B-Chat",
    num_samples: int | None = None,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.9,
    max_gen_length: int = 512,
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run GSM8K benchmark on SDAR model using JetEngine.

    Args:
        model_name: HuggingFace model ID or local path
        num_samples: Number of samples to evaluate (None = full test set)
        block_length: SDAR block length (default: 4)
        denoising_steps: Number of denoising steps (default: 4)
        temperature: Sampling temperature (default: 1.0)
        remasking_strategy: Token remasking strategy (default: low_confidence_dynamic)
        confidence_threshold: Threshold for dynamic remasking (default: 0.9)
        max_gen_length: Maximum generation length in tokens
        save_results: Whether to save detailed results to JSON
        verbose: Show detailed generation logs

    Returns:
        Dict with evaluation metrics and results
    """
    from jetengine import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    import time
    import os
    from datetime import datetime

    # Import helper functions
    import sys
    sys.path.append("/root")
    from benchmark_utils import (
        extract_answer_gsm8k,
        evaluate_answer,
        compute_accuracy,
        format_prompt_gsm8k,
    )

    # Capture start time
    run_datetime = datetime.now()
    timestamp_str = run_datetime.strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print("=" * 80)
    print("SDAR-4B GSM8K Benchmark (JetEngine)")
    print("=" * 80)
    print(f"Run datetime: {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_name}")
    print(f"Block length: {block_length}")
    print(f"Denoising steps: {denoising_steps}")
    print(f"Temperature: {temperature}")
    print(f"Remasking strategy: {remasking_strategy}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 80)

    # Download model from HuggingFace to local cache or use existing cached model
    print("\n[1/5] Loading model...")

    # Check if model_name is a local cache name (e.g., "slerp_sdar-4b_qwen3-4b")
    # or a HuggingFace model ID (e.g., "JetLM/SDAR-4B-Chat")
    if "/" in model_name:
        # HuggingFace model ID - download if needed
        model_local_name = model_name.replace("/", "_")
        model_path = f"{MODELS_DIR}/{model_local_name}"

        if not os.path.exists(model_path):
            print(f"  Downloading {model_name} to {model_path}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"  Using cached model at {model_path}")
    else:
        # Local cache name - use directly
        model_path = f"{MODELS_DIR}/{model_name}"
        if os.path.exists(model_path):
            print(f"  Using local cached model: {model_name}")
        else:
            raise FileNotFoundError(
                f"Model not found in cache: {model_path}\n"
                f"Available models in {MODELS_DIR}: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'none'}"
            )

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Initialize JetEngine LLM
    print("\n[3/5] Initializing JetEngine...")
    start_time = time.time()

    llm = LLM(
        model_path,  # Use local path instead of HF model ID
        enforce_eager=True,
        tensor_parallel_size=1,
        mask_token_id=151669,  # SDAR mask token ID
        block_length=block_length
    )

    print(f"JetEngine initialized in {time.time() - start_time:.2f}s")

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        topk=0,
        topp=1.0,
        max_tokens=max_gen_length,
        remasking_strategy=remasking_strategy,
        block_length=block_length,
        denoising_steps=denoising_steps,
        dynamic_threshold=confidence_threshold
    )

    # Load GSM8K dataset
    print("\n[4/5] Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples")

    # Run evaluation with batched inference
    print("\n[5/5] Running evaluation...")
    results = []
    correct_count = 0
    total_gen_time = 0.0

    # Batching configuration
    batch_size = 32  # Process 32 questions at a time for better GPU utilization

    # Create progress bar
    pbar = tqdm(
        total=len(dataset),
        desc="GSM8K Eval",
        unit="q",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Acc: {postfix}",
    )

    # Process dataset in batches
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_examples = dataset.select(range(batch_start, batch_end))

        # Prepare batch of prompts
        batch_prompts = []
        batch_questions = []
        batch_ground_truths = []

        for example in batch_examples:
            question = example["question"]
            ground_truth = example["answer"]

            batch_questions.append(question)
            batch_ground_truths.append(ground_truth)

            # Format prompt
            prompt_text = format_prompt_gsm8k(question, include_instruction=True)
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            batch_prompts.append(formatted_prompt)

        # Generate for entire batch
        try:
            batch_gen_start = time.time()
            batch_outputs = llm.generate_streaming(
                batch_prompts,
                sampling_params,
                max_active=256,
                use_tqdm=False
            )
            batch_gen_time = time.time() - batch_gen_start
            total_gen_time += batch_gen_time

            # Process each output in the batch
            for idx_in_batch, (question, ground_truth, output) in enumerate(
                zip(batch_questions, batch_ground_truths, batch_outputs)
            ):
                idx = batch_start + idx_in_batch

                if verbose:
                    print(f"\n{'='*80}")
                    print(f"Example {idx + 1}/{len(dataset)}")
                    print(f"{'='*80}")
                    print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")

                try:
                    # Extract generated text
                    output_text = output['text']
                    cleaned_text = output_text.replace('<|MASK|>', '')

                    # Extract answer using flexible method (finds last number)
                    predicted_answer = extract_answer_gsm8k(cleaned_text, method="flexible")

                    # Extract ground truth answer for comparison
                    ground_truth_answer = extract_answer_gsm8k(ground_truth, method="strict")

                    # Evaluate
                    is_correct = evaluate_answer(predicted_answer, ground_truth)
                    if is_correct:
                        correct_count += 1

                    # Calculate token metrics
                    output_len = len(tokenizer.encode(output_text))
                    gen_time = batch_gen_time / len(batch_outputs)  # Average time per question
                    tokens_per_sec = output_len / gen_time if gen_time > 0 else 0

                    if verbose:
                        print(f"\n[Result]")
                        print(f"  Generated ({output_len} tokens in {gen_time:.2f}s = {tokens_per_sec:.1f} tok/s):")
                        print(f"  {cleaned_text[:300]}..." if len(cleaned_text) > 300 else f"  {cleaned_text}")
                        print(f"  Predicted: {predicted_answer}")
                        print(f"  Ground Truth: {ground_truth_answer}")
                        print(f"  Correct: {'✓' if is_correct else '✗'}")

                    # Store result
                    result = {
                        "eval_number": idx,
                        "question": question,
                        "ground_truth_full": ground_truth,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_text_full": cleaned_text,
                        "predicted_answer": predicted_answer,
                        "correct": is_correct,
                        "generation_time_seconds": gen_time,
                        "tokens_per_second": tokens_per_sec,
                        "output_tokens": output_len,
                    }
                    results.append(result)

                    # Update progress bar
                    pbar.update(1)
                    current_acc = correct_count / (idx + 1)
                    avg_time = total_gen_time / (idx + 1)
                    pbar.set_postfix_str(f"{current_acc:.1%} | {avg_time:.1f}s/q")

                except Exception as e:
                    print(f"\n  ✗ Error on example {idx}: {e}")
                    import traceback
                    if verbose:
                        traceback.print_exc()

                    ground_truth_answer = extract_answer_gsm8k(ground_truth, method="strict")
                    results.append({
                        "eval_number": idx,
                        "question": question,
                        "ground_truth_full": ground_truth,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_text_full": None,
                        "predicted_answer": None,
                        "correct": False,
                        "error": str(e),
                    })
                    pbar.update(1)

        except Exception as e:
            print(f"\n  ✗ Batch error at index {batch_start}: {e}")
            import traceback
            traceback.print_exc()
            # Skip this batch
            pbar.update(len(batch_examples))

    pbar.close()

    # Compute final metrics
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    metrics = compute_accuracy(results)
    print(f"Accuracy: {metrics['percentage']} ({metrics['correct']}/{metrics['total']})")
    print()
    print("Performance:")
    print(f"  Average generation time: {total_gen_time / len(results):.2f}s per question")
    avg_tokens_per_sec = sum(r.get('tokens_per_second', 0) for r in results if 'tokens_per_second' in r) / len(results)
    print(f"  Average throughput: {avg_tokens_per_sec:.1f} tokens/second")
    print(f"  Total generation time: {total_gen_time:.1f}s ({total_gen_time/60:.1f} minutes)")
    print("=" * 80)

    # Get model folder name for filename
    model_folder = os.path.basename(model_path)

    # Prepare output
    output = {
        "config": {
            "run_datetime": run_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": timestamp_str,
            "model_name": model_name,
            "model_path": model_path,
            "model_folder": model_folder,
            "num_samples": num_samples if num_samples is not None else len(dataset),
            "block_length": block_length,
            "denoising_steps": denoising_steps,
            "temperature": temperature,
            "remasking_strategy": remasking_strategy,
            "confidence_threshold": confidence_threshold,
            "max_gen_length": max_gen_length,
        },
        "metrics": metrics,
        "results": results,
    }

    # Save results if requested
    if save_results:
        # Create model-specific results directory
        model_results_dir = f"{RESULTS_DIR}/{model_folder}"
        os.makedirs(model_results_dir, exist_ok=True)

        # Create filename with model name and timestamp
        safe_model_name = model_folder.replace("/", "_").replace("\\", "_")
        filename = f"gsm8k_{safe_model_name}_{timestamp_str}.json"
        output_file = f"{model_results_dir}/{filename}"

        # Save results to file
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Commit the volume to persist changes
        results_volume.commit()
        print(f"Results committed to volume: sdar-results")

    return output


@app.local_entrypoint()
def main(
    model_name: str = "JetLM/SDAR-4B-Chat",
    num_samples: int | None = None,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.9,
    max_gen_length: int = 512,
    output_file: str = "gsm8k_jetengine_results.json",
    verbose: bool = True,
):
    """
    Run GSM8K benchmark locally via Modal.
    """
    result = run_gsm8k_benchmark.remote(
        model_name=model_name,
        num_samples=num_samples,
        block_length=block_length,
        denoising_steps=denoising_steps,
        temperature=temperature,
        remasking_strategy=remasking_strategy,
        confidence_threshold=confidence_threshold,
        max_gen_length=max_gen_length,
        save_results=True,
        verbose=verbose,
    )

    print("\n✅ Benchmark complete!")
    print(f"📊 Accuracy: {result['metrics']['percentage']}")
