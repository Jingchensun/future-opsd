"""Replace the Qwen-VL vision patch-embedding Conv3d with the equivalent matmul.

torch 2.9.x has a severe cuDNN Conv3d performance regression on these shapes (vLLM ships the
same workaround in vllm/model_executor/layers/conv.py for torch 2.9.0/2.9.1). On B200 it made
the vision tower dominate every SFT step and every eval sample.

The patch embedding uses kernel_size == stride and its input is exactly one patch per row, so
    Conv3d(x).view(-1, D) == F.linear(x.view(N, -1), W.view(D, -1), b)
bit-for-bit up to bf16 accumulation order. Weights are untouched, so checkpoints are unchanged.
"""
import importlib

import torch.nn.functional as F

_TARGETS = [
    ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionPatchEmbed"),
    ("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", "Qwen3VLMoeVisionPatchEmbed"),
    ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VisionPatchEmbed"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5VisionPatchEmbed"),
]


def _forward(self, hidden_states):
    w = self.proj.weight
    x = hidden_states.reshape(-1, w[0].numel()).to(dtype=w.dtype)
    return F.linear(x, w.reshape(w.shape[0], -1), self.proj.bias)


def apply():
    patched = []
    for mod_name, cls_name in _TARGETS:
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except (ImportError, AttributeError):
            continue
        if getattr(cls, "_fl1_linear_patch", False):
            continue
        cls.forward = _forward
        cls._fl1_linear_patch = True
        patched.append(cls_name)
    return patched
