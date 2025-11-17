"""
Modal script for benchmarking SDAR-4B model on GSM8K dataset.

Usage:
    modal run benchmark_gsm8k.py --model-name JetLM/SDAR-4B-Chat --num-samples 100
    modal run benchmark_gsm8k.py --help
"""
import modal
import json
from pathlib import Path

# Define Modal app
app = modal.App("sdar-4b-gsm8k-benchmark")

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
    # 6) Add local files for generation and utils
    .add_local_file("generate.py", remote_path="/root/generate.py")
    .add_local_file("benchmark_utils.py", remote_path="/root/benchmark_utils.py")
)

# Volume for caching models
volume = modal.Volume.from_name("sdar-models-cache", create_if_missing=True)

# Constants
MODELS_DIR = "/models"
RESULTS_DIR = "/results"


@app.function(
    image=image,
    gpu="H100",  # Single H100 GPU
    volumes={MODELS_DIR: volume},
    timeout=7200,  # 2 hours
    memory=65536,  # 64GB RAM
)
def run_gsm8k_benchmark(
    model_name: str = "JetLM/SDAR-4B-Chat",
    num_samples: int | None = None,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,  # Default from generate.py (use 0.0 for greedy)
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    max_gen_length: int = 512,
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run GSM8K benchmark on SDAR model.

    Args:
        model_name: HuggingFace model ID or local path
        num_samples: Number of samples to evaluate (None = full test set)
        block_length: SDAR block length (default: 4, matching generate.py)
        denoising_steps: Number of denoising steps (default: 4, matching generate.py)
        temperature: Sampling temperature (default: 1.0, matching generate.py; use 0.0 for greedy)
        remasking_strategy: Token remasking strategy (default: low_confidence_dynamic)
        confidence_threshold: Threshold for dynamic remasking (default: 0.85)
        max_gen_length: Maximum generation length in tokens
        save_results: Whether to save detailed results to JSON
        verbose: Show detailed generation logs

    Returns:
        Dict with evaluation metrics and results
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from datasets import load_dataset
    from tqdm import tqdm
    import time

    # Import helper functions
    import sys
    sys.path.append("/root")
    from benchmark_utils import (
        extract_answer_gsm8k,
        evaluate_answer,
        compute_accuracy,
        format_prompt_gsm8k,
    )
    from generate import block_diffusion_generate

    # Wrapper to add verbose logging to block_diffusion_generate
    def block_diffusion_generate_verbose(
            model,
            prompt,
            mask_id,
            gen_length=128,
            block_length=8,
            denoising_steps=8,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            remasking_strategy='low_confidence_dynamic',
            confidence_threshold=0.85,
            stopping_criteria_idx=None,
            verbose=False,
        ):
        """
        Wrapper around block_diffusion_generate from generate.py that adds verbose logging.
        """
        if not verbose:
            # If verbose is False, just call the original function
            return block_diffusion_generate(
                model=model,
                prompt=prompt,
                mask_id=mask_id,
                gen_length=gen_length,
                block_length=block_length,
                denoising_steps=denoising_steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                stopping_criteria_idx=stopping_criteria_idx,
            )

        # Verbose version with logging
        from transformers.cache_utils import DynamicCache
        from torch.nn import functional as F

        model.eval()
        input_ids = prompt['input_ids']
        prompt_length = input_ids.shape[1]
        past_key_values = DynamicCache()

        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        print(f"    [Generation] Prompt length: {prompt_length} tokens")
        print(f"    [Generation] Total blocks: {num_blocks}, Total length: {total_length}")

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
        block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                                   .repeat_interleave(block_length, dim=1).unsqueeze(0)
        position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
        x[:, :prompt_length] = input_ids
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length

        # Import helper functions from generate.py
        from generate import get_num_transfer_tokens, sample_with_temperature_topk_topp

        # Prefill stage
        if prefill_length > 0:
            print(f"    [Prefill] Processing {prefill_blocks} blocks ({prefill_length} tokens)")
            cur_x = x[:, :prefill_length]
            cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
            cur_position_ids = position_ids[:, :prefill_length]
            model(cur_x,
                  attention_mask=cur_attn_mask,
                  position_ids=cur_position_ids,
                  past_key_values=past_key_values,
                  use_cache=True,
                  store_kv=True)

        num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

        # Decode stage
        for num_block in range(prefill_blocks, num_blocks):
            print(f"    [Block {num_block + 1}/{num_blocks}] Denoising...")

            cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
            cur_attn_mask = block_diffusion_attention_mask[
                :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
            ]
            cur_position_ids = position_ids[:, num_block*block_length:(num_block+1)*block_length]

            for step in range(denoising_steps + 1):
                mask_index = (cur_x == mask_id)
                num_masked = mask_index.sum().item()

                if num_masked == 0:
                    print(f"      [Step {step}] All tokens unmasked, storing KV cache")
                    model(cur_x,
                          attention_mask=cur_attn_mask,
                          position_ids=cur_position_ids,
                          past_key_values=past_key_values,
                          use_cache=True,
                          store_kv=True)
                    break

                # Denoising
                logits = model(cur_x,
                               attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True,
                               store_kv=False).logits

                # Sampling
                x0, x0_p = sample_with_temperature_topk_topp(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # Remasking strategy
                if remasking_strategy == 'sequential':
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if mask_index[j].any():
                            first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                            transfer_index[j, first_mask_index:first_mask_index + num_transfer_tokens[step]] = True

                elif remasking_strategy == 'low_confidence_static':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True

                elif remasking_strategy == 'low_confidence_dynamic':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = high_conf_mask.sum()
                        if num_high_confidence >= num_transfer_tokens[step]:
                            transfer_index[j] = high_conf_mask
                        else:
                            _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                            transfer_index[j, idx] = True

                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

                num_transferred = transfer_index.sum().item()
                avg_conf = confidence[transfer_index].mean().item() if num_transferred > 0 else 0.0
                print(f"      [Step {step}/{denoising_steps}] Masked: {num_masked}, Transferring: {num_transferred}, Avg confidence: {avg_conf:.3f}")

                cur_x[transfer_index] = x0[transfer_index]

            x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
            if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
                print(f"    [Block {num_block + 1}] Stopping criteria met")
                break

        return x

    # ========== Main Benchmark Logic ==========

    print("=" * 80)
    print("SDAR-4B GSM8K Benchmark")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Block length: {block_length}")
    print(f"Denoising steps: {denoising_steps}")
    print(f"Temperature: {temperature}")
    print(f"Remasking strategy: {remasking_strategy}")
    print("=" * 80)

    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Get special token IDs
    mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    gen_cfg = GenerationConfig.from_pretrained(model_name)
    stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(stopping_criteria_idx, int):
        stopping_criteria_idx = [stopping_criteria_idx]

    print(f"Model loaded in {time.time() - start_time:.2f}s")
    print(f"Mask token ID: {mask_id}")
    print(f"EOS token IDs: {stopping_criteria_idx}")

    # Load GSM8K dataset
    print("\n[2/4] Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples")

    # Run evaluation
    print("\n[3/4] Running evaluation...")
    results = []
    correct_count = 0
    total_gen_time = 0.0

    # Create progress bar with custom format
    pbar = tqdm(
        dataset,
        desc="GSM8K Eval",
        unit="q",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Acc: {postfix}",
    )

    for idx, example in enumerate(pbar):
        question = example["question"]
        ground_truth = example["answer"]

        if verbose:
            print(f"\n{'='*80}")
            print(f"Example {idx + 1}/{len(dataset)}")
            print(f"{'='*80}")
            print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")

        # Format prompt
        prompt_text = format_prompt_gsm8k(question, include_instruction=True)
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenize
        tokens = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        try:
            # Generate
            gen_start = time.time()
            output_ids = block_diffusion_generate_verbose(
                model,
                prompt=tokens,
                mask_id=mask_id,
                gen_length=max_gen_length,
                block_length=block_length,
                denoising_steps=denoising_steps,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                stopping_criteria_idx=stopping_criteria_idx,
                verbose=verbose,
            )
            gen_time = time.time() - gen_start
            total_gen_time += gen_time

            # Decode output
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            cleaned_text = output_text.replace('<|MASK|>', '')

            # Extract answer from generated text
            predicted_answer = extract_answer_gsm8k(cleaned_text, method="flexible")

            # Evaluate
            is_correct = evaluate_answer(predicted_answer, ground_truth)
            if is_correct:
                correct_count += 1

            # Calculate tokens generated
            prompt_len = tokens['input_ids'].shape[1]
            output_len = output_ids.shape[1]
            tokens_generated = output_len - prompt_len
            tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

            if verbose:
                print(f"\n[Result]")
                print(f"  Generated ({tokens_generated} tokens in {gen_time:.2f}s = {tokens_per_sec:.1f} tok/s):")
                print(f"  {cleaned_text[:300]}..." if len(cleaned_text) > 300 else f"  {cleaned_text}")
                print(f"  Predicted: {predicted_answer}")
                print(f"  Ground Truth: {extract_answer_gsm8k(ground_truth)}")
                print(f"  Correct: {'✓' if is_correct else '✗'}")

            # Store result
            result = {
                "idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "generated_text": cleaned_text,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
                "generation_time": gen_time,
                "tokens_per_second": tokens_per_sec,
            }
            results.append(result)

            # Update progress bar
            current_acc = correct_count / (idx + 1)
            avg_time = total_gen_time / (idx + 1)
            pbar.set_postfix_str(f"{current_acc:.1%} | {avg_time:.1f}s/q")

        except Exception as e:
            print(f"\n  ✗ Error on example {idx}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()

            results.append({
                "idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "generated_text": None,
                "predicted_answer": None,
                "correct": False,
                "error": str(e),
            })

    pbar.close()

    # Compute final metrics
    print("\n[4/4] Computing metrics...")
    metrics = compute_accuracy(results)

    # Calculate performance statistics
    valid_results = [r for r in results if "generation_time" in r]
    if valid_results:
        avg_gen_time = sum(r["generation_time"] for r in valid_results) / len(valid_results)
        avg_tokens_per_sec = sum(r["tokens_per_second"] for r in valid_results) / len(valid_results)
        total_time = sum(r["generation_time"] for r in valid_results)
    else:
        avg_gen_time = 0
        avg_tokens_per_sec = 0
        total_time = 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {metrics['percentage']} ({metrics['correct']}/{metrics['total']})")
    print(f"")
    print(f"Performance:")
    print(f"  Average generation time: {avg_gen_time:.2f}s per question")
    print(f"  Average throughput: {avg_tokens_per_sec:.1f} tokens/second")
    print(f"  Total generation time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 80)

    # Save results if requested
    if save_results:
        output = {
            "config": {
                "model_name": model_name,
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

        # Note: In Modal, we'd typically save to a Volume or return the data
        print("\nResults prepared for export.")
        return output

    return {"metrics": metrics, "results": results}


@app.local_entrypoint()
def main(
    model_name: str = "JetLM/SDAR-4B-Chat",
    num_samples: int | None = None,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    output_file: str = "gsm8k_results.json",
    verbose: bool = True,
):
    """
    Main entry point for running GSM8K benchmark.

    Args:
        model_name: HuggingFace model ID
        num_samples: Number of samples (None = full test set)
        block_length: SDAR block length (default: 4)
        denoising_steps: Number of denoising steps (default: 4)
        temperature: Sampling temperature (default: 1.0; use 0.0 for greedy decoding)
        output_file: Path to save results JSON
        verbose: Show detailed generation logs
    """
    print(f"Starting GSM8K benchmark for {model_name}")
    print(f"Samples: {num_samples or 'all (1319)'}")
    print(f"Verbose output: {'enabled' if verbose else 'disabled'}")

    # Run benchmark on Modal
    results = run_gsm8k_benchmark.remote(
        model_name=model_name,
        num_samples=num_samples,
        block_length=block_length,
        denoising_steps=denoising_steps,
        temperature=temperature,
        verbose=verbose,
    )

    # Save results locally
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Final Accuracy: {results['metrics']['percentage']}")
