
from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

log = logging.getLogger("ComfyUI-LLaDA-Image")


def _load_city96_modules():
    """Load City96 ComfyUI-GGUF internals without executing its node __init__ again."""
    alias = "_llada_city96_gguf"
    if alias in sys.modules:
        pkg = sys.modules[alias]
        return (
            importlib.import_module(alias + ".loader"),
            importlib.import_module(alias + ".dequant"),
            importlib.import_module(alias + ".ops"),
        )

    comfy_root = Path(folder_paths.__file__).resolve().parent
    custom_nodes = comfy_root / "custom_nodes"

    candidates = []
    for p in custom_nodes.iterdir() if custom_nodes.is_dir() else []:
        if not p.is_dir():
            continue
        if (p / "loader.py").is_file() and (p / "dequant.py").is_file() and (p / "ops.py").is_file():
            name = p.name.lower().replace("_", "-")
            if "gguf" in name:
                candidates.append(p)

    # Prefer the canonical City96 folder if present.
    candidates.sort(key=lambda p: (0 if p.name.lower() == "comfyui-gguf" else 1, p.name.lower()))
    if not candidates:
        raise RuntimeError(
            "ComfyUI-GGUF was not found. Install/enable City96 ComfyUI-GGUF first."
        )

    root = candidates[0]
    pkg = types.ModuleType(alias)
    pkg.__path__ = [str(root)]
    pkg.__package__ = alias
    sys.modules[alias] = pkg

    loader = importlib.import_module(alias + ".loader")
    dequant = importlib.import_module(alias + ".dequant")
    ops = importlib.import_module(alias + ".ops")
    log.info("LLaDA GGUF adapter using %s", root)
    return loader, dequant, ops


def _load_text_encoder_code(config_dir: Path):
    alias = "_llada_text_encoder_code"
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(config_dir)]
        pkg.__package__ = alias
        sys.modules[alias] = pkg

    cfg_mod = importlib.import_module(alias + ".configuration_llada2uni_moe")
    model_mod = importlib.import_module(alias + ".modeling_llada2uni_moe")
    return cfg_mod, model_mod


def _resolve_attr(root, dotted: str):
    cur = root
    parts = dotted.split(".")
    for part in parts:
        if part.isdigit() and isinstance(cur, (nn.ModuleList, nn.Sequential, list, tuple)):
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _resolve_parent(root, dotted: str):
    parts = dotted.split(".")
    if len(parts) == 1:
        return root, parts[0]
    return _resolve_attr(root, ".".join(parts[:-1])), parts[-1]


def _replace_child(root, dotted: str, value):
    parent, leaf = _resolve_parent(root, dotted)
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential, list)):
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _drop_registered_attr(parent: nn.Module, leaf: str):
    if leaf in getattr(parent, "_parameters", {}):
        del parent._parameters[leaf]
    if leaf in getattr(parent, "_buffers", {}):
        del parent._buffers[leaf]
    if leaf in getattr(parent, "_modules", {}):
        del parent._modules[leaf]


class LazyGGUFLinear(nn.Module):
    """Linear that keeps GGUF bytes mmap-backed and dequantizes only for the active call."""

    def __init__(self, qweight, bias, dequant_mod):
        super().__init__()
        object.__setattr__(self, "_qweight", qweight)
        object.__setattr__(self, "_bias_value", bias)
        object.__setattr__(self, "_dequant_mod", dequant_mod)

    @property
    def weight(self):
        return self._qweight

    @property
    def bias(self):
        return self._bias_value

    def forward(self, x):
        q = self._qweight
        dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16

        if self._dequant_mod.is_quantized(q):
            qdev = q.to(device=x.device)
            w = self._dequant_mod.dequantize_tensor(qdev, dtype=dtype, dequant_dtype=dtype)
        else:
            w = q.to(device=x.device, dtype=dtype)

        b = self._bias_value
        if b is not None:
            b = b.to(device=x.device, dtype=dtype)

        y = F.linear(x.to(dtype=dtype), w, b)
        del w
        return y


class LazyGGUFEmbedding(nn.Module):
    """Embedding with mmap-backed GGUF storage.

    IMPORTANT: LLaDA2 is a BF16 model. CPU offload must not silently promote
    embeddings to FP32, because that promotes attention q/k/v to FP32 while
    the pipeline attention mask stays BF16.
    """

    def __init__(self, qweight, padding_idx, dequant_mod, compute_dtype=torch.bfloat16):
        super().__init__()
        object.__setattr__(self, "_qweight", qweight)
        self.padding_idx = padding_idx
        self.compute_dtype = compute_dtype
        object.__setattr__(self, "_dequant_mod", dequant_mod)

    @property
    def weight(self):
        return self._qweight

    def forward(self, input_ids):
        q = self._qweight
        dtype = self.compute_dtype
        if self._dequant_mod.is_quantized(q):
            qdev = q.to(device=input_ids.device)
            w = self._dequant_mod.dequantize_tensor(qdev, dtype=dtype, dequant_dtype=dtype)
        else:
            w = q.to(device=input_ids.device, dtype=dtype)

        y = F.embedding(input_ids, w, padding_idx=self.padding_idx)
        del w
        return y


def _expert_slice(qweight, expert_idx: int, dtype: torch.dtype, device, dequant_mod):
    """Dequantize ONE expert from a flattened GGUF 3-D expert bank."""
    shape = tuple(int(x) for x in getattr(qweight, "tensor_shape", qweight.shape))
    if len(shape) != 3:
        raise RuntimeError(f"Expected 3-D expert bank, got {shape}")

    experts, out_features, in_features = shape
    if expert_idx < 0 or expert_idx >= experts:
        raise IndexError(expert_idx)

    if not dequant_mod.is_quantized(qweight):
        return qweight[expert_idx].to(device=device, dtype=dtype)

    raw = qweight.data
    raw_rows = raw.reshape((-1, raw.shape[-1]))
    r0 = expert_idx * out_features
    r1 = r0 + out_features
    packed = raw_rows[r0:r1].to(device=device)

    return dequant_mod.dequantize(
        packed,
        qweight.tensor_type,
        torch.Size((out_features, in_features)),
        dtype=dtype,
    )


def _make_quant_moe_forward(original_fn, dequant_mod):
    @torch.no_grad()
    def quant_moe_forward(
        module,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        quantized = any(
            dequant_mod.is_quantized(w)
            for w in (fc1_1_weight, fc1_2_weight, fc2_weight)
        )
        if not quantized:
            return original_fn(
                module,
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
            )

        if hidden_states.ndim != 2:
            raise RuntimeError(f"LLaDA GGUF MoE expected [tokens, hidden], got {tuple(hidden_states.shape)}")

        top_k = selected_experts.shape[1]
        flat_experts = selected_experts.reshape(-1).to(torch.int64)
        order = torch.argsort(flat_experts, stable=True)
        sorted_hidden = hidden_states[torch.div(order, top_k, rounding_mode="floor")].contiguous()
        sorted_routing = routing_weights.reshape(-1)[order].contiguous()

        tokens_per_expert = torch.bincount(flat_experts, minlength=num_experts)
        expert_ends = torch.cumsum(tokens_per_expert, dim=0).to("cpu", torch.int64).tolist()

        dtype = hidden_states.dtype
        device = hidden_states.device
        outputs = []
        start = 0

        for expert_idx, end in enumerate(expert_ends):
            if end <= start:
                continue

            expert_input = sorted_hidden[start:end]

            w_gate = _expert_slice(fc1_1_weight, expert_idx, dtype, device, dequant_mod)
            w_up = _expert_slice(fc1_2_weight, expert_idx, dtype, device, dequant_mod)
            w_down = _expert_slice(fc2_weight, expert_idx, dtype, device, dequant_mod)

            gate = F.linear(expert_input, w_gate)
            up = F.linear(expert_input, w_up)
            intermediate = F.silu(gate) * up
            intermediate.mul_(sorted_routing[start:end].to(dtype=dtype).unsqueeze(-1))
            out = F.linear(intermediate, w_down)
            outputs.append(out)

            del w_gate, w_up, w_down, gate, up, intermediate
            start = end

        if outputs:
            sorted_outputs = torch.cat(outputs, dim=0)
        else:
            sorted_outputs = hidden_states.new_empty((0, hidden_states.shape[-1]))

        new_x = torch.empty_like(sorted_outputs)
        new_x[order] = sorted_outputs

        return (
            new_x.view(*selected_experts.shape, -1)
            .mul_(routing_weights.to(dtype=dtype).unsqueeze(-1))
            .sum(dim=1)
            .to(dtype=hidden_states.dtype)
        )

    return quant_moe_forward


def _load_gguf_state(path: Path):
    loader_mod, dequant_mod, _ = _load_city96_modules()

    # Our encoder deliberately uses general.architecture="lumina2" so the City96
    # image GGUF path accepts it. handle_prefix=None preserves exact HF keys.
    sd, extra = loader_mod.gguf_sd_loader(
        str(path),
        handle_prefix=None,
        is_text_model=False,
    )

    metadata = extra.get("metadata", {})
    remapped = {}
    for key, tensor in sd.items():
        original = metadata.get(f"comfy.gguf.orig_name.{key}")
        remapped[original or key] = tensor

    return remapped, extra, dequant_mod


def load_llada2_gguf_encoder(gguf_path: str | Path, config_dir: str | Path, dtype=torch.bfloat16):
    """Instantiate LLaDA2 on meta and bind mmap-backed GGUF tensors to it."""
    gguf_path = Path(gguf_path)
    config_dir = Path(config_dir)

    sd, extra, dequant_mod = _load_gguf_state(gguf_path)
    cfg_mod, model_mod = _load_text_encoder_code(config_dir)

    cfg_json = json.loads((config_dir / "text_encoder_config.json").read_text(encoding="utf-8"))

    # The bundled config can carry a stale/smaller vocab_size.  The actual
    # LLaDA-Image GGUF encoder contains the authoritative embedding table, so
    # derive vocab_size from the checkpoint BEFORE constructing the HF config.
    # This keeps the real LLaDA pad_token_id intact instead of silencing the
    # Transformers warning by changing it to an arbitrary token.
    embedding_key = None
    for candidate in (
        "model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
        "embed_tokens.weight",
    ):
        if candidate in sd:
            embedding_key = candidate
            break

    if embedding_key is None:
        # Fall back to a unique embedding-like tensor name if converter/model
        # naming changes in a future release.
        matches = [k for k in sd if k.endswith("embed_tokens.weight")]
        if len(matches) == 1:
            embedding_key = matches[0]

    if embedding_key is not None:
        actual_vocab_size = int(sd[embedding_key].shape[0])
        configured_vocab_size = int(cfg_json.get("vocab_size", actual_vocab_size))
        if configured_vocab_size != actual_vocab_size:
            log.info(
                "Correcting LLaDA2 text-encoder vocab_size from %d to checkpoint value %d",
                configured_vocab_size,
                actual_vocab_size,
            )
            cfg_json["vocab_size"] = actual_vocab_size

        pad_id = cfg_json.get("pad_token_id")
        if pad_id is not None and not (0 <= int(pad_id) < actual_vocab_size):
            raise RuntimeError(
                "LLaDA2 config/checkpoint mismatch: "
                f"pad_token_id={pad_id} is outside checkpoint vocab_size={actual_vocab_size}. "
                "Refusing to substitute an arbitrary padding token."
            )

    cfg_cls = getattr(cfg_mod, "LLaDA2MoeConfig")
    model_cls = getattr(model_mod, "LLaDA2MoeModelLM")
    config = cfg_cls.from_dict(cfg_json)

    # The bundled LLaDA2 remote-code model indexes ROPE_INIT_FUNCTIONS["default"]
    # directly. Newer Transformers releases intentionally removed the default
    # implementation from that registry (their native models handle it locally).
    # Do NOT rename the checkpoint's RoPE type: "default" is the correct LLaDA2
    # semantics. Supply the missing implementation locally and leave config alone.
    def _llada_default_rope(cfg, device=None, **kwargs):
        base = float(getattr(cfg, "rope_theta", 10000.0))
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = int(cfg.hidden_size) // int(cfg.num_attention_heads)
        partial = float(getattr(cfg, "partial_rotary_factor", 1.0))
        dim = int(head_dim * partial)
        if dim <= 0 or dim % 2:
            raise RuntimeError(f"Invalid LLaDA2 RoPE dimension: {dim}")
        dev = device if device is not None else "cpu"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=dev) / dim))
        return inv_freq, 1.0

    rope_registry = getattr(model_mod, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(rope_registry, dict):
        raise RuntimeError("LLaDA2 modeling code does not expose ROPE_INIT_FUNCTIONS")
    rope_registry.setdefault("default", _llada_default_rope)

    try:
        from accelerate import init_empty_weights
    except Exception as e:
        raise RuntimeError("accelerate is required for the LLaDA GGUF meta loader") from e

    with init_empty_weights(include_buffers=True):
        model = model_cls(config)

    module_map = dict(model.named_modules())
    consumed = set()

    # Replace every ordinary Linear/Embedding weight with a lazy mmap-backed module.
    for module_name, module in list(module_map.items()):
        if module_name == "":
            continue

        wkey = module_name + ".weight"
        if wkey not in sd:
            continue

        if isinstance(module, nn.Linear):
            bias = sd.get(module_name + ".bias")
            repl = LazyGGUFLinear(sd[wkey], bias, dequant_mod)
            _replace_child(model, module_name, repl)
            consumed.add(wkey)
            if bias is not None:
                consumed.add(module_name + ".bias")

        elif isinstance(module, nn.Embedding):
            repl = LazyGGUFEmbedding(
                sd[wkey],
                getattr(module, "padding_idx", None),
                dequant_mod,
                compute_dtype=dtype,
            )
            _replace_child(model, module_name, repl)
            consumed.add(wkey)

    # Load all remaining tensors. 3-D expert banks stay quantized/mmap-backed.
    for key, tensor in sd.items():
        if key in consumed:
            continue

        try:
            parent, leaf = _resolve_parent(model, key)
        except Exception as e:
            raise RuntimeError(f"GGUF tensor does not map to encoder module: {key}") from e

        shape = tuple(int(x) for x in getattr(tensor, "tensor_shape", tensor.shape))

        if len(shape) == 3 and ".mlp.experts." in key:
            _drop_registered_attr(parent, leaf)
            object.__setattr__(parent, leaf, tensor)
            continue

        # Keep ONLY MoE routing tensors in FP32. LLaDA2 RMSNorm / q_norm / k_norm
        # weights must stay in the model compute dtype; loading every 1-D tensor as
        # FP32 promotes normalized hidden/query/key states to FP32 and breaks SDPA
        # against the BF16 attention mask.
        force_fp32 = (
            key.endswith(".mlp.gate.weight")
            or key.endswith(".mlp.gate.expert_bias")
        )
        out_dtype = torch.float32 if force_fp32 else dtype

        if dequant_mod.is_quantized(tensor):
            value = dequant_mod.dequantize_tensor(
                tensor,
                dtype=out_dtype,
                dequant_dtype=out_dtype,
            ).cpu()
        else:
            value = tensor.to(device="cpu", dtype=out_dtype)

        _drop_registered_attr(parent, leaf)
        if isinstance(parent, nn.Module):
            parent.register_parameter(leaf, nn.Parameter(value, requires_grad=False))
        else:
            setattr(parent, leaf, value)

    # Replace the modeling module's imported fused_moe_forward symbol so the
    # 3-D expert banks dequantize one selected expert at a time.
    if hasattr(model_mod, "fused_moe_forward"):
        original = model_mod.fused_moe_forward
        model_mod.fused_moe_forward = _make_quant_moe_forward(original, dequant_mod)

    # Newer PyTorch requires a floating SDPA mask to exactly match query dtype.
    # Keep this local to the LLaDA2 attention modules. This also protects against
    # future offload/device hooks promoting hidden states.
    for _m in model.modules():
        if _m.__class__.__name__ in {"LLaDA2Attention", "LLaDA2SdpaAttention"}:
            _orig_forward = _m.forward

            def _dtype_safe_forward(*args, __orig=_orig_forward, **kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                mask = kwargs.get("attention_mask")
                if mask is not None and torch.is_floating_point(mask):
                    # A boolean mask is accepted regardless of query dtype and avoids
                    # PyTorch's strict floating-mask/query dtype equality check.
                    # LLaDA uses additive masks with 0 for keep and a large negative
                    # value for blocked positions, so >= 0 preserves the same mask.
                    kwargs["attention_mask"] = mask >= 0
                return __orig(*args, **kwargs)

            _m.forward = _dtype_safe_forward

    model.eval()
    model.requires_grad_(False)

    log.info(
        "Loaded LLaDA2 GGUF encoder: %s (arch=%s, tensors=%d)",
        gguf_path.name,
        extra.get("arch_str"),
        len(sd),
    )
    return model


class LazyINT8Linear(nn.Module):
    """Optimized runtime for converter int8_tensorwise Linear weights.

    Quantized tensors are non-persistent buffers so Accelerate/Diffusers can
    move them with the transformer component. This avoids copying every INT8
    Linear weight CPU->GPU again on every denoising forward.
    """

    def __init__(self, qweight, scale, bias, compute_dtype, convrot=False, groupsize=256):
        super().__init__()
        # Keep quantized weights as lazy CPU-owned tensors. Do NOT register
        # these as module buffers: Accelerate model_cpu_offload would otherwise
        # migrate the entire INT8 transformer before step 1, stalling low-VRAM
        # cards. Each Linear stages only the weight needed for the current op.
        object.__setattr__(self, "_qweight", qweight)
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_bias_value", bias)
        self.compute_dtype = compute_dtype
        self.convrot = bool(convrot)
        self.groupsize = int(groupsize)

    @staticmethod
    def _fast_hadamard4(x, groupsize):
        """Exact normalized H4 Kronecker transform without a dense matrix."""
        g = int(groupsize)
        original_shape = x.shape
        y = x.float().reshape(*original_shape[:-1], original_shape[-1] // g, g)

        stride = 1
        while stride < g:
            shape = y.shape
            y = y.reshape(*shape[:-1], g // (4 * stride), 4, stride)
            a, b, c, d = y.unbind(dim=-2)
            y = torch.stack(
                (
                    a + b + c - d,
                    a + b - c + d,
                    a - b + c + d,
                    -a + b + c + d,
                ),
                dim=-2,
            )
            y = y.reshape(*shape)
            stride *= 4

        return y.mul_(g ** -0.5).reshape(original_shape)

    def _rotate_input(self, x):
        if not self.convrot:
            return x

        k = x.shape[-1]
        g = self.groupsize
        if k % g:
            raise RuntimeError(f"ConvRot input width {k} is not divisible by group size {g}")

        # Converter uses power-of-4 H4 Kronecker groups (normally 256).
        cur = g
        while cur > 1 and cur % 4 == 0:
            cur //= 4
        if cur != 1:
            raise RuntimeError(
                f"ConvRot group size {g} is not a power of 4; optimized transform cannot be used."
            )

        return self._fast_hadamard4(x, g).to(self.compute_dtype)

    def forward(self, x):
        dt = self.compute_dtype
        xr = self._rotate_input(x.to(dt))

        # Fast path: component-level offload has already moved these buffers.
        q = self._qweight
        s = self._scale
        b = self._bias_value

        # Fallback for unusual/manual execution modes.
        if q.device != x.device:
            q = q.to(device=x.device, non_blocking=True)
        if s.device != x.device or s.dtype != dt:
            s = s.to(device=x.device, dtype=dt, non_blocking=True)
        if b is not None and (b.device != x.device or b.dtype != dt):
            b = b.to(device=x.device, dtype=dt, non_blocking=True)

        # Weight-only INT8: only the current Linear is expanded to compute dtype.
        w = q.to(dtype=dt) * s
        return F.linear(xr, w, b)


def _decode_quant_marker(tensor):
    raw = bytes(tensor.detach().cpu().to(torch.uint8).tolist())
    return json.loads(raw.decode("utf-8").strip())


def _load_transformer_int8(sd, config_dir: Path, dtype):
    from accelerate import init_empty_weights
    from .llada.transformer_llada_image import LLaDAImageTransformer2DModel

    cfg = json.loads((config_dir / "transformer_config.json").read_text(encoding="utf-8"))
    with init_empty_weights(include_buffers=True):
        model = LLaDAImageTransformer2DModel.from_config(cfg)

    consumed = set()
    for key in list(sd.keys()):
        if not key.endswith(".comfy_quant"):
            continue
        base = key[:-len(".comfy_quant")]
        wkey = base + ".weight"
        skey = base + ".weight_scale"
        if wkey not in sd or skey not in sd:
            raise RuntimeError(f"Incomplete INT8 tensor group for {base}")
        qcfg = _decode_quant_marker(sd[key])
        if qcfg.get("format") != "int8_tensorwise":
            raise RuntimeError(f"Unsupported quant format for {base}: {qcfg}")
        try:
            old = _resolve_attr(model, base)
        except Exception as e:
            raise RuntimeError(f"INT8 tensor does not map to transformer module: {base}") from e
        if not isinstance(old, nn.Linear):
            raise RuntimeError(f"INT8 target is not Linear: {base} ({type(old).__name__})")
        bias_key = base + ".bias"
        bias = sd.get(bias_key)
        repl = LazyINT8Linear(
            sd[wkey], sd[skey], bias, dtype,
            convrot=qcfg.get("convrot", False),
            groupsize=qcfg.get("convrot_groupsize", 256),
        )
        _replace_child(model, base, repl)
        consumed.update((key, wkey, skey))
        if bias is not None:
            consumed.add(bias_key)

    # Assign every non-quantized source tensor directly into the meta model.
    remainder = {k: v for k, v in sd.items() if k not in consumed}
    missing, unexpected = model.load_state_dict(remainder, strict=False, assign=True)
    # Missing .weight entries belonging to replaced LazyINT8Linear modules are expected.
    bad_missing = []
    quant_bases = {k[:-len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant")}
    for k in missing:
        if k.endswith(".weight") and k[:-len(".weight")] in quant_bases:
            continue
        if k.endswith(".bias") and k[:-len(".bias")] in quant_bases and (k not in sd):
            continue
        bad_missing.append(k)
    if bad_missing:
        raise RuntimeError("LLaDA INT8 transformer missing keys: " + ", ".join(bad_missing[:30]))
    if unexpected:
        raise RuntimeError("LLaDA INT8 transformer unexpected keys: " + ", ".join(unexpected[:30]))
    model.eval()
    model.requires_grad_(False)
    log.info("Loaded native LLaDA INT8 transformer (%d quantized Linear layers)", len(quant_bases))
    return model


def _load_transformer(diffusion_path: Path, config_dir: Path, dtype):
    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    from .llada.transformer_llada_image import LLaDAImageTransformer2DModel

    cfg = json.loads((config_dir / "transformer_config.json").read_text(encoding="utf-8"))

    with init_empty_weights(include_buffers=True):
        model = LLaDAImageTransformer2DModel.from_config(cfg)

    sd = load_file(str(diffusion_path), device="cpu")

    # Native converter output: bind quantized Linear layers lazily instead of
    # rejecting the model or expanding the whole transformer in RAM.
    if any(k.endswith(".comfy_quant") for k in sd):
        return _load_transformer_int8(sd, config_dir, dtype)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        raise RuntimeError("LLaDA transformer missing keys: " + ", ".join(missing[:20]))
    if unexpected:
        raise RuntimeError("LLaDA transformer unexpected keys: " + ", ".join(unexpected[:20]))

    model.eval()
    model.requires_grad_(False)
    return model


def _load_vae(vae_path: Path, config_dir: Path, dtype):
    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLFlux2

    cfg = json.loads((config_dir / "vae_config.json").read_text(encoding="utf-8"))
    with init_empty_weights(include_buffers=True):
        vae = AutoencoderKLFlux2.from_config(cfg)

    sd = load_file(str(vae_path), device="cpu")
    missing, unexpected = vae.load_state_dict(sd, strict=False, assign=True)
    if missing:
        raise RuntimeError("LLaDA VAE missing keys: " + ", ".join(missing[:20]))
    if unexpected:
        raise RuntimeError("LLaDA VAE unexpected keys: " + ", ".join(unexpected[:20]))

    vae.eval()
    vae.requires_grad_(False)
    return vae


def build_llada_pipeline(selection: dict, pipeline_cls, config_dir: Path):
    """Build the real callable LLaDAImagePipeline from Comfy dropdown selections."""

    # Safe inference optimizations for NVIDIA GPUs. These do not change the
    # stored quantized weights or sampling algorithm.
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    dtype_name = selection["dtype"]
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name, torch.bfloat16)

    diffusion_path = Path(selection["diffusion_path"])
    encoder_path = Path(selection["text_encoder_path"])
    vae_path = Path(selection["vae_path"])

    if encoder_path.suffix.lower() != ".gguf":
        raise RuntimeError(
            "This build's standalone text-encoder path currently expects the LLaDA2 GGUF encoder."
        )

    text_encoder = load_llada2_gguf_encoder(encoder_path, config_dir, dtype=dtype)
    transformer = _load_transformer(diffusion_path, config_dir, dtype)
    vae = _load_vae(vae_path, config_dir, dtype)

    # Native Diffusers AutoencoderKLFlux2 tiled decoding.
    # Auto is intentionally conservative: enable tiling as a low-VRAM default,
    # since output dimensions are only known later when the pipeline is called.
    vae_tiling = str(selection.get("vae_tiling", "On")).lower()
    if vae_tiling in ("on", "auto"):
        enable_tiling = getattr(vae, "enable_tiling", None)
        if enable_tiling is None:
            raise RuntimeError(
                "This Diffusers AutoencoderKLFlux2 build does not expose enable_tiling(). "
                "Update Diffusers or set VAE tiling to Off."
            )
        enable_tiling()
        log.info("LLaDA VAE tiled decoding ENABLED (%s)", selection.get("vae_tiling", "On"))
    elif vae_tiling == "off":
        disable_tiling = getattr(vae, "disable_tiling", None)
        if disable_tiling is not None:
            disable_tiling()
        log.info("LLaDA VAE tiled decoding disabled")
    else:
        raise ValueError(f"Unknown VAE tiling mode: {selection.get('vae_tiling')}")

    # LLaDAImagePipeline overrides DiffusionPipeline.from_pretrained and only
    # accepts (repo/path, torch_dtype, device). Therefore component overrides
    # such as text_encoder=... are NOT valid kwargs. Load only the small
    # official auxiliary components, then construct the pipeline directly with
    # our selected GGUF encoder, transformer and VAE.
    from huggingface_hub import snapshot_download

    # Select auxiliary components from the SAME upstream variant as the loaded
    # transformer/text encoder. Base must not be paired with Turbo scheduler /
    # QueryFormer / text_projection / SigVQ assets.
    names = f"{diffusion_path.name} {encoder_path.name}".lower()
    has_base = "base" in names
    has_turbo = "turbo" in names

    if has_base and has_turbo:
        raise RuntimeError(
            "LLaDA model mismatch: Base and Turbo were selected together. "
            f"Transformer={diffusion_path.name!r}, text_encoder={encoder_path.name!r}. "
            "Use matching Base+Base or Turbo+Turbo weights."
        )

    if has_base:
        variant = "Base"
        aux_repo_id = "inclusionAI/LLaDA-Image"
    elif has_turbo:
        variant = "Turbo"
        aux_repo_id = "inclusionAI/LLaDA-Image-Turbo"
    else:
        # Preserve compatibility with older/custom filenames that do not carry
        # an explicit variant marker. Existing releases were Turbo-first, so
        # retain Turbo as the fallback and make the choice visible in the log.
        variant = "Turbo (fallback: filenames did not identify Base/Turbo)"
        aux_repo_id = "inclusionAI/LLaDA-Image-Turbo"

    log.info(
        "LLaDA variant: %s | auxiliary components: %s",
        variant,
        aux_repo_id,
    )

    aux_root = Path(
        snapshot_download(
            repo_id=aux_repo_id,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "queryformer/*",
                "text_projection/*",
                "sigvq/*",
            ],
        )
    )

    # Reuse the exact classes imported by the bundled official pipeline module,
    # avoiding assumptions about package layout/version.
    g = pipeline_cls.from_pretrained.__func__.__globals__
    SchedulerCls = g["FlowMatchEulerDiscreteScheduler"]
    TokenizerCls = g["AutoTokenizer"]
    QueryFormerCls = g["LLaDAImageQueryFormerModel"]
    TextProjectionCls = g["LLaDAImageTextProjectionModel"]
    SigVQCls = g["LLaDAImageSigVQModel"]

    scheduler = SchedulerCls.from_pretrained(aux_root / "scheduler")
    tokenizer = TokenizerCls.from_pretrained(aux_root / "tokenizer")
    queryformer = QueryFormerCls.from_pretrained(
        aux_root / "queryformer", torch_dtype=dtype
    )
    text_projection = TextProjectionCls.from_pretrained(
        aux_root / "text_projection", torch_dtype=dtype
    )
    sigvq = SigVQCls.from_pretrained(
        aux_root / "sigvq", torch_dtype=dtype
    )

    pipe = pipeline_cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        queryformer=queryformer,
        text_projection=text_projection,
        sigvq=sigvq,
        transformer=transformer,
    )

    offload = selection.get("offload", "sequential_cpu_offload")
    if offload == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model_cpu_offload":
        pipe.enable_model_cpu_offload()
    elif offload == "cuda":
        # Custom INT8 CUDA mode:
        # - Keep LazyINT8Linear qweights/scales CPU-resident and stage them on demand.
        # - Move the normal transformer parameters AND the auxiliary neural modules
        #   used before denoising to CUDA.
        #
        # This matters especially for native editing. Before the progress bar is
        # created, the official pipeline runs QueryFormer/text projection, SigVQ
        # reference-image encoding, and VAE reference-latent encoding. Leaving
        # SigVQ/VAE on CPU can make an edit appear hung for hours.
        if not torch.cuda.is_available():
            raise RuntimeError("LLaDA CUDA mode selected but CUDA is not available.")

        cuda_device = torch.device("cuda:0")

        log.info("LLaDA CUDA: moving transformer non-quantized parameters to cuda:0")
        transformer.to(device=cuda_device, dtype=dtype)
        log.info("LLaDA CUDA: transformer core ready; INT8 matrices remain lazy on CPU")

        # Match the official all-CUDA pipeline for the auxiliary compute modules,
        # but do it explicitly so our unregistered INT8 matrices are NOT eagerly
        # migrated by DiffusionPipeline.to().
        log.info("LLaDA CUDA: moving QueryFormer to cuda:0")
        queryformer.to(device=cuda_device, dtype=dtype)

        log.info("LLaDA CUDA: moving text projection to cuda:0")
        text_projection.to(device=cuda_device, dtype=dtype)

        log.info("LLaDA CUDA: moving SigVQ to cuda:0 (required for fast native editing)")
        sigvq.to(device=cuda_device, dtype=dtype)

        log.info("LLaDA CUDA: moving VAE to cuda:0 (required for fast edit image encoding/decoding)")
        vae.to(device=cuda_device, dtype=dtype)

        # IMPORTANT: the auxiliary modules are only needed before denoising.
        # Keeping QueryFormer/TextProjection/SigVQ/VAE resident on an 8 GB GPU
        # while the first INT8 transformer layer stages/dequantizes is enough to
        # make the progress bar sit at 0/N while CUDA allocator thrashes.
        #
        # A transformer forward-pre-hook is the exact phase boundary we need:
        # prompt/edit preprocessing has already completed, but no denoising
        # matrix has been staged yet. Release the auxiliary modules there.
        aux_release_state = {"done": False}

        def _release_aux_before_denoise(_module, _args):
            if aux_release_state["done"]:
                return
            aux_release_state["done"] = True
            log.info("LLaDA CUDA: preprocessing complete; releasing aux modules before denoising")
            for _name, _component in (
                ("queryformer", queryformer),
                ("text_projection", text_projection),
                ("sigvq", sigvq),
                ("vae", vae),
            ):
                try:
                    _component.to("cpu")
                except Exception as _exc:
                    log.warning("LLaDA CUDA: could not release %s to CPU: %s", _name, _exc)
            torch.cuda.empty_cache()
            log.info("LLaDA CUDA: aux VRAM released; starting transformer steps")

        transformer.register_forward_pre_hook(_release_aux_before_denoise)

        log.info(
            "LLaDA CUDA: pipeline ready; aux preprocessing uses CUDA and will be released immediately before step 1"
        )
    elif offload == "cpu":
        pipe.to("cpu")
    else:
        raise ValueError(f"Unknown offload mode: {offload}")

    return pipe
