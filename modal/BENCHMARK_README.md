# GSM8K Benchmarking with SDAR-4B on Modal

This directory contains scripts to benchmark the SDAR-4B model on the GSM8K dataset using Modal's H100 GPUs.

## Files

- **`benchmark_gsm8k.py`**: Original Modal script using transformers (deprecated - slower)
- **`benchmark_gsm8k_jetengine.py`**: **Recommended** Modal script using JetEngine (production-grade, 2-4× faster)
- **`benchmark_utils.py`**: Helper functions for answer extraction and evaluation
- **`BENCHMARK_README.md`**: This file

## Architecture

### JetEngine (Recommended)

The **`benchmark_gsm8k_jetengine.py`** script uses [JetEngine](https://github.com/Labman42/JetEngine), a production-grade inference engine built on nano-vllm:

**Benefits:**
- ✅ **2-4× faster** than naive implementation (1800+ tok/s on A800, 3700+ tok/s on H100)
- ✅ Production-level throughput with batching support
- ✅ Clean API with streaming support
- ✅ Optimized with FlashAttention-2 + Triton kernels
- ✅ No custom generation code needed

**Installation:**
JetEngine is installed during Modal image build by:
1. Cloning the SDAR repo with submodules
2. Installing JetEngine from `third_party/JetEngine`
3. No code duplication - uses JetEngine's optimized implementation

### Original Implementation (Deprecated)

The **`benchmark_gsm8k.py`** script imports from `generate.py` but is slower and less optimized. Use `benchmark_gsm8k_jetengine.py` instead.

## Setup

### 1. Install Modal (already done in venv)

```bash
source venv/bin/activate
modal setup
```

Follow the prompts to authenticate with your Modal account.

### 2. Configuration

**Software Stack:**
- **Base Image**: NVIDIA CUDA 12.9.1 devel (includes nvcc + full toolkit)
- **PyTorch**: 2.8.0 with CUDA 12.9 (cu129 wheels)
- **Transformers**: 4.52.4
- **Flash Attention**: 2.8.3 (compiled with --no-build-isolation)
- **JetEngine**: Latest (from SDAR third_party submodule)
- **Python**: 3.11
- **GPU**: H100

**Generation Settings** (from `generate.py`):
- **Block length**: 4
- **Denoising steps**: 4
- **Temperature**: 1.0 (use `--temperature 0.0` for greedy decoding)
- **Remasking strategy**: low_confidence_dynamic
- **Confidence threshold**: 0.85
- **Dtype**: bfloat16 (from model config)

## Usage

### Basic Usage (Recommended - JetEngine)

Run the complete GSM8K test set (1,319 examples) with JetEngine:

```bash
modal run benchmark_gsm8k_jetengine.py
```

### Quick Test (Subset)

Test on a smaller subset for debugging:

```bash
modal run benchmark_gsm8k_jetengine.py --num-samples 10
```

### Custom Configuration

```bash
modal run benchmark_gsm8k_jetengine.py \
  --model-name JetLM/SDAR-4B-Chat \
  --num-samples 100 \
  --block-length 4 \
  --denoising-steps 4 \
  --temperature 1.0 \
  --confidence-threshold 0.9 \
  --output-file gsm8k_jetengine_results.json \
  --verbose true
```

### Greedy Decoding (for baseline comparison)

```bash
modal run benchmark_gsm8k_jetengine.py --temperature 0.0
```

### Disable Verbose Output

```bash
modal run benchmark_gsm8k_jetengine.py --verbose false
```

### Original Implementation (Deprecated)

If you need to use the slower transformers-based implementation:

```bash
modal run benchmark_gsm8k.py --num-samples 10
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | str | JetLM/SDAR-4B-Chat | HuggingFace model ID or local path |
| `--num-samples` | int | None | Number of samples (None = full test set) |
| `--block-length` | int | 4 | SDAR block length |
| `--denoising-steps` | int | 4 | Number of denoising steps |
| `--temperature` | float | 1.0 | Sampling temperature (0.0 = greedy) |
| `--output-file` | str | gsm8k_results.json | Path to save results JSON |
| `--verbose` | bool | true | Show detailed generation logs |

## Output

### Console Output

The script provides rich console output with:

1. **Configuration summary** at startup
2. **Progress bar** with live accuracy updates:
   ```
   GSM8K Eval: 45%|████▌     | 56/124 [01:23<01:47, 0.63q/s] Acc: 52.5% | 1.5s/q
   ```
3. **Verbose per-example logs** (if enabled):
   - Question text
   - Generation details (blocks, tokens, time)
   - Denoising step-by-step progress
   - Generated answer and correctness
4. **Final results summary**:
   ```
   ================================================================================
   FINAL RESULTS
   ================================================================================
   Accuracy: 52.34% (65/124)

   Performance:
     Average generation time: 1.45s per question
     Average throughput: 156.2 tokens/second
     Total generation time: 179.8s (3.0 minutes)
   ================================================================================
   ```

### Verbose Output Example

When `--verbose true` (default), you'll see detailed logs for each example:

```
================================================================================
Example 1/100
================================================================================
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast...

    [Generation] Prompt length: 142 tokens
    [Generation] Total blocks: 164, Total length: 656
    [Prefill] Processing 35 blocks (140 tokens)
    [Block 36/164] Denoising...
      [Step 0/4] Masked: 4, Transferring: 1, Avg confidence: 0.892
      [Step 1/4] Masked: 3, Transferring: 1, Avg confidence: 0.876
      [Step 2/4] Masked: 2, Transferring: 1, Avg confidence: 0.901
      [Step 3/4] Masked: 1, Transferring: 1, Avg confidence: 0.945
      [Step 4] All tokens unmasked, storing KV cache
    [Block 37/164] Denoising...
    ...

[Result]
  Generated (87 tokens in 1.23s = 70.7 tok/s):
  <|im_start|>user
  Janet's ducks lay 16 eggs per day...
  <|im_start|>assistant
  Let me solve this step by step...
  #### 18
  Predicted: 18
  Ground Truth: 18
  Correct: ✓
```

### JSON Output

Results are saved to `gsm8k_results.json` (or your specified output file):

```json
{
  "config": {
    "model_name": "JetLM/SDAR-4B-Chat",
    "block_length": 4,
    "denoising_steps": 4,
    "temperature": 1.0,
    "remasking_strategy": "low_confidence_dynamic",
    "confidence_threshold": 0.85,
    "max_gen_length": 512
  },
  "metrics": {
    "accuracy": 0.5234,
    "correct": 65,
    "total": 124,
    "percentage": "52.34%"
  },
  "results": [
    {
      "idx": 0,
      "question": "Janet's ducks lay 16 eggs per day...",
      "ground_truth": "She sells 16 - 3 - 4 = <<16-3-4=9>>9...\n#### 18",
      "generated_text": "Let me solve this step by step...\n#### 18",
      "predicted_answer": "18",
      "correct": true,
      "generation_time": 1.23,
      "tokens_per_second": 70.7
    },
    ...
  ]
}
```

## Performance Expectations

On H100 with the default settings:
- **Throughput**: ~100-200 tokens/second
- **Time per question**: ~1-3 seconds
- **Full test set**: ~30-60 minutes

Performance varies based on:
- Question/answer length
- Block length and denoising steps
- Dynamic vs static remasking

## Model Caching

The first run will download the model (~8GB for SDAR-4B) to Modal's volume. Subsequent runs will use the cached model, starting much faster.

## Troubleshooting

### Modal Authentication Issues
```bash
modal setup
```

### Out of Memory
Reduce batch processing or use a smaller model:
```bash
modal run benchmark_gsm8k.py --num-samples 10
```

### Slow Generation
The verbose output helps diagnose:
- Check if blocks are taking longer than expected
- Verify confidence thresholds are working (should see high confidence values)
- Try increasing `--confidence-threshold` for faster (but potentially less accurate) generation

## Advanced: Experimenting with Settings

### Test Different Block Lengths
```bash
modal run benchmark_gsm8k.py --block-length 8 --denoising-steps 8
```

### Test Different Remasking Strategies

You can modify the script to test other strategies:
- `low_confidence_static`: Fixed number of tokens per step
- `sequential`: Left-to-right sequential unmasking
- `entropy_bounded`: Entropy-based token selection

### Batch Processing

To process multiple configurations in parallel, you can launch multiple Modal runs.

## References

- SDAR Paper: [Link to paper if available]
- GSM8K Dataset: https://huggingface.co/datasets/openai/gsm8k
- Modal Documentation: https://modal.com/docs
