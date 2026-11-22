"""E14 CVAE model (cvae/, exported by cvae/export.py): HF Qwen3-VL whose answer is generated after K latent slots.

Per batch: one prefill of the plain prompt gives the prior p(z|x) from the last hidden state; z = prior mean
(latent=mean, the deterministic plug-in prediction) or a prior sample (latent=sample); P(z) is written into K slots
placed at the end of the user turn, then the answer is generated as usual. Prompts / preprocessing / parsing are those
of `qwen3_vl` (system prompt from QWEN3VL_SYSTEM_PROMPT).

reasoning_budget (with extract_answer): budget forcing as in qwen3_vl (BudgetForcing, in-generation, so the latents
and KV cache are reused).
"""
import json
import os
import sys

import torch
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_vl import Qwen3_VL


@register_model("qwen3_vl_cvae")
class Qwen3_VL_CVAE(Qwen3_VL):
    def __init__(self, pretrained: str, latent: str = "mean", **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        root = os.environ.get("FUTURE_L1_CODE_ROOT")
        if root and root not in sys.path:
            sys.path.append(root)
        from cvae.model import CVAEHeads, LatentSlots, insert_slots, last_hidden

        self._insert_slots, self._last_hidden = insert_slots, last_hidden
        cfg = json.load(open(os.path.join(pretrained, "cvae_config.json")))
        self.K = cfg["K"]
        d = self.model.config.text_config.hidden_size
        self.heads = CVAEHeads(d, cfg["K"], cfg["dz"]).to(self.device)
        self.heads.load_state_dict(torch.load(os.path.join(pretrained, "heads.pt"), map_location=self.device))
        self.heads.eval()
        self.slots = LatentSlots(self.model.get_input_embeddings())
        assert latent in ("mean", "sample"), latent
        self.latent = latent
        eval_logger.info(f"qwen3_vl_cvae: K={self.K} dz={cfg['dz']} latent={latent} budget={self.reasoning_budget} "
                         f"(stage {cfg.get('stage')} step {cfg.get('step')})")

    @torch.no_grad()
    def _generate(self, inputs, generate_kwargs):
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        assert self.processor.tokenizer.padding_side == "left" or ids.shape[0] == 1 or bool(mask[:, -1].all())
        h = self._last_hidden(self.model.model, inputs)
        mu, lv = self.heads.prior(h)
        z = mu if self.latent == "mean" else mu + (0.5 * lv).exp() * torch.randn_like(mu)
        ids2 = self._insert_slots(ids, self.K)
        mask2 = self._insert_slots(mask, self.K)
        mask2[..., -5 - self.K: -5] = 1
        kw, procs = self._budget(generate_kwargs, ids2.shape[1])
        model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        self.slots.ez = self.heads.project(z)
        try:
            out = self.model.generate(input_ids=ids2, attention_mask=mask2, **model_inputs, **kw)
        finally:
            self.slots.ez = None
        self._log_forced(procs, ids.shape[0])
        return torch.cat([ids, out[:, ids2.shape[1]:]], dim=1)  # the caller strips len(prompt) tokens
