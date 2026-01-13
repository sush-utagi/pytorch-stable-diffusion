# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a from-scratch PyTorch implementation of Stable Diffusion v1.5, part of an honours research project on text-guided diffusion models. The implementation includes custom attention visualization capabilities for analyzing cross-attention patterns between text prompts and generated images.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Core Components (sd/)

**Model Components:**
- `diffusion.py`: U-Net architecture for the diffusion model with time embedding
- `encoder.py`: VAE encoder (compresses 512×512 images to 64×64 latent space)
- `decoder.py`: VAE decoder (reconstructs images from latent space)
- `clip.py`: CLIP text encoder for processing prompts

**Attention System:**
- `attention.py`: SelfAttention and CrossAttention layer implementations
  - CrossAttention: Queries from image features, Keys/Values from text embeddings (77 tokens)
  - Used to understand which text tokens influence image generation
- `attention_capture.py`: Global context manager system for capturing attention weights during forward passes
  - Provides `capture_attention()` context manager
  - Thread-safe global storage `_ATTENTION_STORE`
  - Functions: `store_attention()`, `get_attention_maps()`, `aggregate_attention()`
- `attention_viz.py`: Visualization utilities for attention analysis

**Sampling:**
- `ddpm.py`: DDPM sampler with stochastic noise scheduling
  - Default training timesteps: 1000
  - Beta schedule: linear from 0.00085 to 0.012
  - Implements variance-based noise injection
  - Better quality but slower (50-1000 steps recommended)

**Pipeline:**
- `pipeline.py`: Main generation pipeline
  - Handles text-to-image and image-to-image generation
  - Implements Classifier-Free Guidance (CFG)
  - Device management with idle_device support for memory efficiency
  - Fixed dimensions: 512×512 output, 64×64 latent space (8× downsampling)

**Utilities:**
- `model_loader.py`: Loads pretrained weights using model_converter
- `model_converter.py`: Converts standard Stable Diffusion checkpoints to this architecture

### Key Architectural Patterns

**Latent Diffusion:**
- Images are encoded to 4-channel latent space (Batch, 4, 64, 64)
- Diffusion operates in latent space for efficiency
- VAE decoder converts final latent back to RGB

**Classifier-Free Guidance (CFG):**
- Runs both conditional (with prompt) and unconditional (empty prompt) forward passes
- Formula: `output = cfg_scale * (cond - uncond) + uncond`
- Default cfg_scale: 7.5
- Requires 2× compute per step but dramatically improves prompt adherence

**Time Embedding:**
- Sinusoidal positional encoding for timesteps (160 dimensions → 320 after concat)
- Pattern: `get_time_embedding(timestep)` in pipeline.py:164

**Cross-Attention to Text:**
- CLIP tokenizes prompts to 77 tokens (padding/truncation)
- CLIP embeds each token to 768-dimensional vectors
- U-Net cross-attention layers query these embeddings
- Attention shape: (Batch, Heads, Seq_Q, 77) where Seq_Q is spatial (H×W)

## Setup and Usage

### Prerequisites

Python 3.11.3 with dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

Core dependencies:
- torch==2.0.1
- transformers==4.33.2
- numpy==1.25.0
- tqdm==4.65.0

### Required Files (data/)

Must be downloaded manually and placed in `data/`:

1. **Tokenizer files** from [stable-diffusion-v1-5/tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer):
   - `vocab.json`
   - `merges.txt`

2. **Model weights** from [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main):
   - `v1-5-pruned-emaonly.ckpt` (primary checkpoint)

**Note:** The `data/` directory is gitignored due to file size (checkpoint is ~4GB).

### Running Generation

The standard pipeline can be imported and used programmatically:

```python
from sd.model_loader import preload_models_from_standard_weights
from sd.pipeline import generate
from transformers import CLIPTokenizer

# Load models
models = preload_models_from_standard_weights(
    "data/v1-5-pruned-emaonly.ckpt",
    device="cuda"  # or "cpu", "mps"
)

# Load tokenizer
tokenizer = CLIPTokenizer(
    "data/vocab.json",
    merges_file="data/merges.txt"
)

# Generate
image = generate(
    prompt="a cat in a hat",
    uncond_prompt="",  # negative prompt
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models=models,
    seed=42,
    device="cuda",
    tokenizer=tokenizer
)
```

### Attention Visualization

To capture and analyze cross-attention during generation, modify the pipeline's diffusion loop:

```python
from sd.attention_capture import capture_attention, get_attention_maps, get_token_attention

# Inside the timestep loop in pipeline.py
with capture_attention():
    model_output = diffusion(model_input, context, time_embedding)

# After generation
attention_maps = get_attention_maps()
token_attention = get_token_attention(attention_maps)
```

The attention capture system:
1. Uses a global context manager to enable capture mode
2. CrossAttention layers check `is_capture_enabled()` and call `store_attention()`
3. Aggregation functions process raw weights across heads, layers, and spatial dimensions

## Development Notes

### Device Handling

The pipeline supports multiple devices:
- **CUDA**: Standard GPU acceleration
- **MPS**: Apple Silicon GPU (Metal Performance Shaders)
- **CPU**: Fallback, very slow

The `idle_device` parameter allows models to be moved to CPU when not actively used, reducing VRAM usage:
```python
generate(..., device="cuda", idle_device="cpu")
```

### Image-to-Image Generation

Controlled via the `strength` parameter (0-1):
- strength=0.8: Significant transformation from input image
- strength=0.3: Subtle modifications preserving most details

The `strength` determines which timestep to start from:
```python
start_step = num_inference_steps - int(num_inference_steps * strength)
```

Input images are:
1. Resized to 512×512
2. Normalized to [-1, 1]
3. Encoded to latent space with encoder noise
4. Noise added according to starting timestep

### Sampler Implementation

**DDPM Sampler (ddpm.py):**
- Implements Algorithm 2 from Ho et al. (2020)
- Key methods:
  - `set_inference_timesteps(n)`: Selects n evenly-spaced timesteps from 1000
  - `step(timestep, latents, model_output)`: Single denoising step with variance
  - `add_noise(samples, timesteps)`: Forward diffusion for img2img
  - `_get_variance(timestep)`: Computes noise variance (clamped to 1e-20)

The model predicts noise (epsilon), which is then used to estimate x₀ (original sample) via formula (15) from the DDPM paper.

### Model Weight Conversion

The `model_converter.py` handles conversion from standard SD checkpoints:
- Maps state dict keys to this architecture's naming scheme
- Extracts CLIP, VAE encoder, VAE decoder, and U-Net weights
- Critical for loading official checkpoints and fine-tuned models

### Fine-tuned Model Compatibility

Compatible with any Stable Diffusion v1.x checkpoint. Tested with:
- [InkPunk Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion)
- [Illustration Diffusion](https://huggingface.co/ogkalu/Illustration-Diffusion)

Simply download the `.ckpt` file and pass its path to `preload_models_from_standard_weights()`.

## Common Issues

### Missing tokenizer files
**Error:** FileNotFoundError when loading tokenizer

**Solution:** Download `vocab.json` and `merges.txt` from HuggingFace and place in `data/`

### Missing checkpoint
**Error:** Pipeline fails silently or crashes during model loading

**Solution:** Ensure `v1-5-pruned-emaonly.ckpt` exists in `data/` directory

### CUDA Out of Memory
**Solution:** Use `idle_device="cpu"` to move inactive models to CPU, or reduce inference steps

### MPS (Apple Silicon) Issues
**Solution:** Use autocast with float16 for memory efficiency:
```python
with torch.autocast("mps", dtype=torch.float16):
    generate(...)
```

### Attention capture returns empty maps
**Issue:** CrossAttention layers not instrumented with capture calls

**Solution:** Verify `attention.py` CrossAttention.forward() calls `store_attention()` when enabled

## Research Context

This implementation serves as the foundation for research on:
- Text-guided synthetic data generation
- Cross-attention analysis for prompt interpretation
- Semantic alignment between text and image features
- Data augmentation strategies using diffusion models

The custom attention visualization system enables analysis of:
- Token importance across denoising timesteps
- Spatial attention patterns (which image regions attend to which words)
- Layer-wise attention differences in the U-Net architecture

## Git Branch Strategy

- **main**: Stable baseline implementation
- **attention-scores-from-progressive-denoising**: Current branch for attention visualization features

Changes to core generation quality should be carefully tested as they can degrade output fidelity.
