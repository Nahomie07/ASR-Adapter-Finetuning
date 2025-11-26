import torch
import torch.nn as nn
from typing import Optional

class Adapter(nn.Module):
    """
    Adapter bottleneck module (Houlsby-style).
    Input dim = hidden_dim (d_model). Reduction to bottleneck_dim.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64, activation: str = "relu", scale: float = 1.0):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.scale = scale
        # initialize small: zero init for up to make initial behavior close to identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x: (batch, seq_len, d_model) or (seq_len, batch, d_model)
        h = self.down(x)
        h = self.act(h)
        h = self.up(h)
        return x + self.scale * h

import inspect

def inject_adapters_whisper(model, bottleneck_dim=64, add_to_encoder=True, add_to_decoder=True, scale=1.0):
    """
    Parcours les modules du modèle Whisper et ajoute des adaptateurs aux layers.
    - model: WhisperForConditionalGeneration
    Returns list of (module_path, adapter) inserted for saving.
    """
    inserted = []
    base = getattr(model, "model", model)
    for part_name in ["encoder", "decoder"]:
        part = getattr(base, part_name, None)
        if part is None:
            continue
        layers = getattr(part, "layers", None) or getattr(part, "block", None) or getattr(part, "layer", None)
        if layers is None:
            # try to inspect children
            children = list(part.children())
            for name, obj in part.__dict__.items():
                if isinstance(obj, (list, tuple)) and len(obj)>0:
                    layers = obj
                    break
        if layers is None:
            continue
        for i, layer in enumerate(layers):
            d_model = layer.self_attn.embed_dim if hasattr(layer, "self_attn") else getattr(layer, "embed_dim", None)
            if d_model is None and hasattr(model.config, "d_model"):
                d_model = model.config.d_model
            if d_model is None:
                d_model = 384
            adapter = Adapter(hidden_dim=d_model, bottleneck_dim=bottleneck_dim, scale=scale)
            setattr(layer, "adapter", adapter)
            if not hasattr(layer, "_adapter_patched"):
                _patch_layer_forward_with_adapter(layer)
            inserted.append((f"{part_name}.layers.{i}.adapter", adapter))
    return inserted

def _patch_layer_forward_with_adapter(layer):
    """
    Monkeypatch the layer.forward to apply adapter after original forward output.
    Keeps original forward as attr _orig_forward.
    """
    if hasattr(layer, "_orig_forward"):
        return
    layer._orig_forward = layer.forward

    def new_forward(*args, **kwargs):
        out = layer._orig_forward(*args, **kwargs)
        if isinstance(out, tuple):
            main = out[0]
            rest = out[1:]
        else:
            main = out
            rest = None
        if hasattr(layer, "adapter"):
            main = layer.adapter(main)
        if rest is None:
            return main
        if isinstance(out, tuple):
            return (main, *rest)
        return main

    layer.forward = new_forward
    layer._adapter_patched = True
