# LLaDA-Image Turbo for ComfyUI


## UPDATE! made optimizations to speed up gen time. 

Custom ComfyUI nodes for **LLaDA-Image-Turbo**, with support for **text-to-image generation** and **native image editing**.

This project is built around the official LLaDA-Image-Turbo model from inclusionAI and adds a ComfyUI pipeline designed to run the model with RebelAI's optimized weights.

## Links

- **Model weights / Hugging Face:** https://huggingface.co/realrebelai/LLaDa-Image-Turbo_ComfyUI
- **This ComfyUI node repository:** https://github.com/RealRebelAI/LLaDa-Image_ComfyUI
- **Official LLaDA-Image-Turbo:** https://huggingface.co/inclusionAI/LLaDA-Image-Turbo
- **Official LLaDA-Image source:** https://github.com/inclusionAI/LLaDA-Image

## Features

- LLaDA-Image-Turbo text-to-image generation
- **Native LLaDA image editing**
- Native INT8 transformer support
- Q4_K_M GGUF LLaDA2-MoE text encoder support
- BF16/full transformer support
- ComfyUI model dropdowns
- 4-step Turbo generation
- Seed, resolution, CFG and negative-prompt controls
- Shared pipeline loader for generation and editing

## Native Image Editing

Editing is not conventional diffusion img2img and does not emulate editing with a denoise-strength slider.

The `LLaDA Image Edit` node invokes LLaDA-Image's native editing path with:

`generation_mode="editing"`

The source image is passed through the model's image-conditioning/SigVQ path and combined with the edit instruction.

Basic workflow:

```text
LLaDA Image Loader
        |
        +--------------------> LLaDA Image Text to Image
        |
        +--------------------> LLaDA Image Edit <---- Load Image
                                      |
                                      v
                                  Save Image
```

For Turbo, start with **4 steps** and **CFG 1.0**. Editing dimensions must be divisible by 32.

Example edit prompt:

> Turn the fox into a white arctic fox while preserving the forest composition and realistic photography.

## Model Files

The optimized model files are hosted separately on Hugging Face:

https://huggingface.co/realrebelai/LLaDa-Image-Turbo_ComfyUI

Supported weights include:

| File | Purpose |
|---|---|
| `LLaDA-Image-Turbo-transformer-BF16.safetensors` | Full/BF16 transformer |
| `LLaDA-Image-Turbo-transformer-INT8.safetensors` | Native INT8 transformer |
| `LLaDA-Image-Turbo-text_encoder-Q4_K_M-v3.gguf` | Q4_K_M LLaDA2-MoE text encoder |

The INT8 file is a native Safetensors transformer, **not a transformer GGUF**. GGUF is used for the quantized LLaDA2-MoE text encoder.

The remaining LLaDA components are based on the official LLaDA-Image-Turbo release.

## ComfyUI Model Placement

Place transformer weights in:

```text
ComfyUI/models/diffusion_models/
```

Place the GGUF text encoder in:

```text
ComfyUI/models/text_encoders/
```

Place the LLaDA VAE in your normal ComfyUI VAE directory:

```text
ComfyUI/models/vae/
```

The loader scans ComfyUI's model directories rather than relying on machine-specific absolute paths.

## Nodes

### LLaDA Image Loader

Loads the selected transformer, LLaDA2 text encoder, VAE, and supporting LLaDA pipeline components.

### LLaDA Image Text to Image

Runs native LLaDA-Image-Turbo text-to-image generation.

Recommended Turbo starting point:

- Steps: `4`
- CFG: `1.0`

### LLaDA Image Edit

Runs **native LLaDA editing** using a source image and an edit instruction.

Inputs include the pipeline, source image, prompt, width, height, steps, CFG, seed, and optional negative prompt.

### LLaDA Image Unload

Releases the loaded LLaDA pipeline when you are finished with it.

## Quantization

This project supports a mixed optimized configuration:

- **Transformer:** native INT8 Safetensors
- **Text encoder:** Q4_K_M GGUF
- **Alternative transformer:** full BF16 Safetensors

The quantized weights are derivatives of:

**inclusionAI/LLaDA-Image-Turbo**

See the Hugging Face model repository for the weight files and model card:

https://huggingface.co/realrebelai/LLaDa-Image-Turbo_ComfyUI

## Upstream Project

LLaDA-Image and LLaDA-Image-Turbo are developed by inclusionAI.

Official model:

https://huggingface.co/inclusionAI/LLaDA-Image-Turbo

Official source:

https://github.com/inclusionAI/LLaDA-Image

This repository provides an independent ComfyUI integration/optimized runtime and is not the upstream LLaDA-Image repository.

## License

Use the upstream LLaDA-Image/LLaDA-Image-Turbo license and terms applicable to the original model, along with any license included in this repository for the integration code.
