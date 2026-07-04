"""The Probe&Prefill wrapper — the core of the framework."""
import torch
from .probe import ToolProbe


class ToolGate:
    def __init__(self, probe: ToolProbe):
        self.probe = probe
        self.model = probe.model
        self.tokenizer = probe.tokenizer
        self.cfg = probe.cfg

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int | None = None, **gen_kwargs) -> dict:
        prob = self.probe.predict_proba(prompt)
        use_tool = prob >= self.cfg.tau
        prefill = self.cfg.tool_prefill if use_tool else self.cfg.direct_prefill

        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ) + prefill

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            **gen_kwargs,
        )
        response = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return {
            "prompt": prompt,
            "tool_needed": use_tool,
            "prob": prob,
            "prefill_used": prefill,
            "response": response,
        }