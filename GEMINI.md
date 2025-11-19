# GEMINI.md

This file provides guidance to the Gemini coding assistant when working with code in this repository.

## Project Overview

**SDAR (Synergy of Diffusion and AutoRegression)** is a large-scale diffusion language model research project that combines autoregressive and discrete diffusion modeling approaches. The project delivers competitive performance with state-of-the-art open-source AR models while providing 2-4x faster inference through parallel decoding.

- **Key Innovation**: Block diffusion approach based on modified Qwen3 architecture
- **Performance**: 2-4x faster inference compared to traditional autoregressive models
- **Model Sizes**: 1.7B, 4B, 8B (dense) and 30B (MoE)
- **Status**: Early experimental research code (MIT License)

## Essential Commands

### Environment Setup

**Training Environment (Full, conda-based, not applicable to Colab):**
```bash
conda env create -f training/llamafactory_full_env.yml
conda activate llamafactory_sdar
```

**Training Environment (Manual):**
```bash
cd training/llama_factory_sdar
pip install -e .
pip install flash-attn --no-build-isolation  # Recommended for Flash Attention 2
```

**Inference-only Environment:**
```bash
pip install transformers>=4.52.4
```

### Inference

**1. Built-in Inference Script (Quick Testing):**
```bash
python generate.py \
  --model_dir=JetLM/SDAR-1.7B-Chat \
  --trust_remote_code \
  --block_length=4 \
  --denoising_steps=4 \
  --temperature=1.0 \
  --top_k=0 \
  --top_p=1.0 \
  --remasking_strategy=low_confidence_dynamic \
  --confidence_threshold=0.85
```

**Key Parameters:**
- `--block_length`: Block size (default: 4, can scale to 8/16/32/64)
- `--denoising_steps`: Denoising iterations (default: 4)
- `--remasking_strategy`: Token selection strategy (see Technical Details)
- `--confidence_threshold`: For dynamic remasking (default: 0.85)

**2. JetEngine (Production Batch Inference):**

JetEngine delivers significant speedup (1800+ tokens/second on A800, 3700+ on H200):

```bash
# Setup
cd thirdparty/JetEngine
pip install .
```

```python
from jetengine import LLM, SamplingParams
from transformers import AutoTokenizer

llm = LLM(
    model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    mask_token_id=151669,
    block_length=4
)

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=256,
    remasking_strategy="low_confidence_dynamic",
    block_length=4,
    denoising_steps=4,
    dynamic_threshold=0.9
)

outputs = llm.generate_streaming([prompt], sampling_params)
```

**3. LMDeploy (Alternative Production Engine):**
```python
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

backend_config = PytorchEngineConfig(
    tp=1,
    dtype="float16",
    dllm_block_length=4,
    dllm_denoising_steps=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
    dllm_confidence_threshold=0.9,
)

pipe = pipeline('JetLM/SDAR-8B-Chat', backend_config=backend_config)
outputs = pipe(prompts, gen_config=GenerationConfig(top_p=0.95, temperature=1.0))
```

### Training & Fine-tuning

**Launch Training (Multi-GPU Example not applicable to a Colab-based single-GPU setup):**
```bash
torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 8 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
    training/llama_factory_sdar/src/llamafactory/launcher.py \
    training/llama_factory_sdar/examples/train_full_sdar/sdar_4b/sdar_4b_math_cot_full.yaml
```

**Or using CLI:**
```bash
cd training/llama_factory_sdar
llamafactory-cli train examples/train_full_sdar/sdar_4b/sdar_4b_math_cot_full.yaml
```

**Development Commands:**
```bash
cd training/llama_factory_sdar

make quality    # Check code quality
make style      # Auto-fix style issues
make test       # Run test suite
make build      # Build distribution package
```

## High-Level Architecture

### Core Components

**1. Block Diffusion Generation**
- Custom sampling and denoising mechanism for parallel token generation
- Uses special `<|MASK|>` token (ID: 151669) for masking
- Iteratively unmasks tokens based on confidence scores

**2. Flex Attention**
- Uses PyTorch's Flex Attention for efficient attention computation
- Requires fixed-shape inputs (hence neat packing)
- Critical for performance optimization

**3. Custom Modeling Files**
Located in model directories (e.g., `training/model/SDAR-4B-Chat/`):
- `modeling_sdar.py`: Core SDAR model implementation
- `configuration_sdar.py`: Model configuration
- `fused_linear_diffusion_cross_entropy.py`: Optimized loss computation

**4. Modified LlamaFactory Framework**
`training/llama_factory_sdar/` contains a customized version of LlamaFactory:
- `src/llamafactory/train/`: Training workflows (pt, sft, dpo, kto, ppo, rm), contains "trainer_utils.py" and "tuner.py" with further functional clarifications
- `src/llamafactory/data/`: SDAR-specific data processing with neat packing, contains plethora of key .py scripts
- `src/llamafactory/model/`: Model loading with custom file support
- `src/llamafactory/chat/`: Inference engines
- `examples/`: Training configuration templates

### Directory Structure

```
SDAR/
├── generate.py                     # Built-in inference script
├── training/
│   ├── llamafactory_full_env.yml  # Complete conda environment with required dependencies
│   ├── model/                     # Example model directories with custom files
│   └── llama_factory_sdar/        # Modified LlamaFactory framework
│       ├── src/
│       │   ├── train.py          # Training entry point
│       │   └── llamafactory/     # Main package
│       └── examples/
│           ├── train_full_sdar/  # Full parameter fine-tuning configs
│           ├── train_lora/       # LoRA configs
│           └── deepspeed/        # DeepSpeed configurations
└── thirdparty/
    └── JetEngine/                 # Git submodule for optimized inference
```

**Important**: JetEngine is a git submodule. After cloning, run:
```bash
git submodule update --init --recursive
```

## Critical Technical Details

### SDAR-Specific Requirements

**1. Neat Packing is Mandatory**
- MUST set `neat_packing: true` in all training configs
- SDAR uses Flex Attention which requires fixed-shape inputs
- Without this, training will fail

**2. Trust Remote Code Required**
- Always set `trust_remote_code: true` when loading models
- SDAR models use custom modeling files not in transformers library
- Required for both training and inference

**3. Block Length Scaling Rules**
- Default `block_length: 4` during pretraining and SFT
- Can scale to 8, 16, 32, 64 **during SFT phase only**
- Keep `block_length: 4` during continued pretraining
- Training config must match model's block size

**4. Truncation Strategy**
- `truncate_mode: drop` (recommended): Discards sequences exceeding cutoff_len
- `truncate_mode: cut`: Truncates sequences to cutoff_len
- 'drop' maintains fixed shapes better for Flex Attention

### Essential Training Config Parameters

```yaml
# Required settings
trust_remote_code: true        # CRITICAL: Must be true
block_length: 4                # Must match model's block size
neat_packing: true             # CRITICAL: Required for Flex Attention
truncate_mode: drop            # Recommended for shape consistency

# Recommended settings
dataset: lyrical_ru2eng_sft
cutoff_len: 2048             # Max sequence length
preprocessing_num_workers: 96  # Parallel data processing
template: qwen3               # Chat template format
gradient_checkpointing: true # to avoid OOM failure
stage: sft
do_train: true
finetuning_type: lora
output_dir: /content/drive/MyDrive/LyricalSDAR_4b
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]
run_name: sdar_4b_lyrical
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
```

### Remasking Strategies

Controls how tokens are unmasked during diffusion decoding:

- **`low_confidence_dynamic`** (recommended): Unmasks high-confidence tokens first, variable number per step
- **`low_confidence_static`**: Fixed number of tokens per step based on confidence
- **`sequential`**: Left-to-right sequential unmasking
- **`entropy_bounded`**: Entropy-based token selection

Dynamic strategies can achieve >2x speedup with minimal accuracy loss.

### Model Preparation Checklist

Before training a model:
1. Create model directory (e.g., `./training/model/SDAR-4B-Chat`)
2. Download model weights from HuggingFace (JetLM organization)
3. Copy custom files from repository examples:
   - `modeling_sdar.py`
   - `configuration_sdar.py`
   - `fused_linear_diffusion_cross_entropy.py`
4. Verify `block_length` in config matches your training setup

## Available Models

From HuggingFace (JetLM organization):
- `SDAR-1.7B-Chat`
- `SDAR-4B-Chat`
- `SDAR-8B-Chat`
- `SDAR-30B-A3B-Chat` (MoE)
- `SDAR-30B-A3B-Sci` (Science/reasoning specialist)

Multiple block size variants available (4, 8, 16, 32, 64).

## Important Notes

1. **This is research code** - SDAR is in "early experimental state"
2. **Flex Attention is critical** - Don't skip `neat_packing: true` or training will fail
3. **Custom modeling files are required** - Models won't load without `trust_remote_code: true`
4. **Multiple inference engines available** - Use `generate.py` for testing, JetEngine/LMDeploy for production
5. **Framework is modified LlamaFactory** - Not vanilla LlamaFactory; includes SDAR-specific modifications
6. **Git submodule setup** - JetEngine requires `git submodule update --init --recursive`
