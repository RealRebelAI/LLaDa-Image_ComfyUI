import gc
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths

from .llada import LLaDAImagePipeline
from .llada_gguf_adapter import build_llada_pipeline


NODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = NODE_DIR / "configs"

_PIPELINE_CACHE = {}


MODEL_EXTENSIONS = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}


def _folder_roots(*keys):
    """Return every registered Comfy model root for the requested categories."""
    roots = []
    seen = set()

    for key in keys:
        try:
            paths = folder_paths.get_folder_paths(key)
        except Exception:
            paths = []

        for p in paths or []:
            p = str(Path(p))
            if p not in seen:
                seen.add(p)
                roots.append(Path(p))

    return roots


def _scan_model_files(keys, extensions=MODEL_EXTENSIONS):
    """Scan the actual registered model directories recursively.

    This intentionally does not depend only on get_filename_list(), because
    some Comfy installs/plugins filter extensions and can omit .gguf files.
    """
    found = {}

    # First include anything Comfy itself already registered.
    for key in keys:
        try:
            for name in folder_paths.get_filename_list(key):
                try:
                    full = folder_paths.get_full_path(key, name)
                except Exception:
                    full = None
                if full:
                    found[name.replace("\\", "/")] = Path(full)
        except Exception:
            pass

    # Then scan every registered directory directly so GGUF is always visible.
    for root in _folder_roots(*keys):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in extensions:
                continue
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.name
            found.setdefault(rel, p)

    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


def _is_llada_transformer_file(path: Path) -> bool:
    """Header-only architecture check; never loads model tensors into RAM."""
    if path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
    except Exception:
        return False

    # Full/BF16/FP8 LLaDA transformer signatures.
    full_signatures = (
        "layers.0.attention.to_q.weight",
        "layers.0.attention.to_k.weight",
        "all_x_embedder.1-1.weight",
    )
    if all(k in keys for k in full_signatures):
        return True

    # Native int8_tensorwise files produced from the same LLaDA architecture.
    int8_signatures = (
        "layers.0.attention.to_q.comfy_quant",
        "layers.0.attention.to_k.comfy_quant",
        "all_x_embedder.1-1.weight",
    )
    if all(k in keys for k in int8_signatures):
        return True

    return False


def _diffusion_files():
    files = _scan_model_files(("diffusion_models", "unet"), {".safetensors"})
    # Only expose actual LLaDA transformer checkpoints. This prevents Flux/Fibo
    # checkpoints such as single_transformer_blocks.* from ever entering the loader.
    return {name: path for name, path in files.items() if _is_llada_transformer_file(path)}


def _text_encoder_files():
    # Explicitly includes GGUF by scanning the physical text-encoder/CLIP roots.
    return _scan_model_files(("text_encoders", "clip"), {".safetensors", ".gguf", ".bin", ".pt", ".pth"})


def _vae_files():
    return _scan_model_files(("vae",), {".safetensors", ".bin", ".pt", ".pth"})


def _resolve_scanned_file(mapping, name, kind):
    path = mapping.get(name)
    if path and path.is_file():
        return path
    raise FileNotFoundError(f"Could not resolve selected {kind}: {name!r}")


def _check_bundled_assets():
    """Architecture/config code belongs to the node itself; no 'support pack'."""
    expected_configs = (
        "transformer_config.json",
        "text_encoder_config.json",
        "queryformer_config.json",
        "sigvq_config.json",
        "text_projection_config.json",
        "vae_config.json",
        "model_index.json",
        "configuration_llada2uni_moe.py",
        "modeling_llada2uni_moe.py",
        "fused_moe_ops.py",
    )
    missing_cfg = [name for name in expected_configs if not (CONFIG_DIR / name).is_file()]
    if missing_cfg:
        raise RuntimeError(
            "Custom-node config bundle is incomplete. Missing: " + ", ".join(missing_cfg)
        )


def _torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _load_pipeline(diffusion_model: str, text_encoder: str, vae_name: str, dtype_name: str, offload: str, vae_tiling: str):
    _check_bundled_assets()

    diffusion_path = _resolve_scanned_file(_diffusion_files(), diffusion_model, "diffusion model")
    text_encoder_path = _resolve_scanned_file(_text_encoder_files(), text_encoder, "text encoder")
    vae_path = _resolve_scanned_file(_vae_files(), vae_name, "VAE")

    key = (str(diffusion_path), str(text_encoder_path), str(vae_path), dtype_name, offload, vae_tiling)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    if diffusion_path.suffix.lower() != ".safetensors":
        raise RuntimeError(
            "LLaDA transformer must be a .safetensors file. "
            f"Selected: {diffusion_path.name}"
        )

    if not _is_llada_transformer_file(diffusion_path):
        raise RuntimeError(
            "Wrong diffusion model selected. This loader only accepts LLaDA-Image "
            "transformer checkpoints. The selected file does not have LLaDA keys "
            "(layers.* / all_x_embedder.*). "
            f"Selected: {diffusion_path.name}"
        )

    selection = {
        "diffusion_path": str(diffusion_path),
        "text_encoder_path": str(text_encoder_path),
        "vae_path": str(vae_path),
        "dtype": dtype_name,
        "offload": offload,
        "vae_tiling": vae_tiling,
    }

    pipe = build_llada_pipeline(selection, LLaDAImagePipeline, CONFIG_DIR)
    if not callable(pipe):
        raise RuntimeError("LLaDA pipeline assembly returned a non-callable object.")

    _PIPELINE_CACHE[key] = pipe
    return pipe

def _comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().cpu().clamp(0, 1)
    arr = (image.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_to_comfy(images):
    if isinstance(images, Image.Image):
        images = [images]

    batch = []
    for image in images:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0
        batch.append(torch.from_numpy(arr))
    return torch.stack(batch, dim=0)


class LLaDAImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        diffusion = list(_diffusion_files().keys()) or ["<no LLaDA diffusion models found>"]
        encoders = list(_text_encoder_files().keys()) or ["<no text encoders found>"]
        vaes = list(_vae_files().keys()) or ["<no VAEs found>"]

        return {
            "required": {
                "diffusion_model": (diffusion,),
                "text_encoder": (encoders,),
                "vae": (vaes,),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "offload": (
                    ["sequential_cpu_offload", "model_cpu_offload", "cuda", "cpu"],
                    {"default": "sequential_cpu_offload"},
                ),
                "vae_tiling": (["On", "Auto", "Off"], {"default": "On"}),
            }
        }

    RETURN_TYPES = ("LLADA_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "LLaDA Image"

    def load(self, diffusion_model, text_encoder, vae, dtype, offload, vae_tiling):
        pipe = _load_pipeline(
            diffusion_model,
            text_encoder,
            vae,
            dtype,
            offload,
            vae_tiling,
        )
        return (pipe,)


class LLaDAImageTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LLADA_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "LLaDA Image"

    def generate(
        self,
        pipeline,
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        negative_prompt="",
    ):
        # Diffusers creates the initial latent on pipeline._execution_device.
        # Under CPU/model/sequential offload that device can be CPU even when
        # CUDA exists, so a hard-coded CUDA Generator crashes randn_tensor.
        execution_device = getattr(pipeline, "_execution_device", None)
        if execution_device is None:
            execution_device = getattr(pipeline, "device", torch.device("cpu"))
        execution_device = torch.device(execution_device)
        generator_device = "cuda" if execution_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        result = pipeline(
            prompt=prompt,
            generation_mode="text",
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            output_type="pil",
        )

        return (_pil_to_comfy(result.images),)


class LLaDAImageEdit:
    """Native LLaDA-Image editing.

    This is NOT img2img/denoise-strength emulation. The source IMAGE is sent
    through the model's official generation_mode="editing" path, which uses
    LLaDA-Image's SigVQ/image-conditioning components.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("LLADA_PIPELINE",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the edit you want to make",
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "LLaDA Image"
    DESCRIPTION = (
        "Native LLaDA-Image editing using generation_mode='editing' and the "
        "model's SigVQ image-conditioning path."
    )

    def edit(
        self,
        pipeline,
        image,
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        negative_prompt="",
    ):
        width = int(width)
        height = int(height)

        # Official editing path requires dimensions divisible by 32.
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError(
                f"LLaDA native editing requires width/height divisible by 32; "
                f"got {width}x{height}."
            )

        if image is None:
            raise ValueError("LLaDA Image Edit requires a source IMAGE.")

        # Diffusers creates the initial latent on pipeline._execution_device.
        # Under CPU/model/sequential offload that device can be CPU even when
        # CUDA exists, so a hard-coded CUDA Generator crashes randn_tensor.
        execution_device = getattr(pipeline, "_execution_device", None)
        if execution_device is None:
            execution_device = getattr(pipeline, "device", torch.device("cpu"))
        execution_device = torch.device(execution_device)
        generator_device = "cuda" if execution_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        source_image = _comfy_image_to_pil(image)

        result = pipeline(
            prompt=prompt,
            image=source_image,
            generation_mode="editing",
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            height=height,
            width=width,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            output_type="pil",
        )

        if not hasattr(result, "images") or not result.images:
            raise RuntimeError("LLaDA native editing returned no images.")

        return (_pil_to_comfy(result.images),)


class LLaDAImageUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("LLADA_PIPELINE",)}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = "LLaDA Image"

    def unload(self, pipeline):
        dead_keys = [key for key, value in _PIPELINE_CACHE.items() if value is pipeline]
        for key in dead_keys:
            del _PIPELINE_CACHE[key]

        try:
            pipeline.to("cpu")
        except Exception:
            pass

        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return ()


NODE_CLASS_MAPPINGS = {
    "LLaDAImageLoader": LLaDAImageLoader,
    "LLaDAImageTextToImage": LLaDAImageTextToImage,
    "LLaDAImageEdit": LLaDAImageEdit,
    "LLaDAImageUnload": LLaDAImageUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaDAImageLoader": "LLaDA Image Loader",
    "LLaDAImageTextToImage": "LLaDA Image Text to Image",
    "LLaDAImageEdit": "LLaDA Image Edit",
    "LLaDAImageUnload": "LLaDA Image Unload",
}
