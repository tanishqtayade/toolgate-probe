"""Extract all-layer hidden states and train a linear probe for tool necessity."""
import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ToolGateConfig

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class ToolProbe:
    def __init__(self, config: ToolGateConfig | None = None):
        self.cfg = config or ToolGateConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        load_kwargs = dict(
            torch_dtype=_DTYPES[self.cfg.dtype],
            device_map=self.cfg.device,
            output_hidden_states=True,
        )
        if self.cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
        self.model.eval()

        self.scaler = StandardScaler()
        self.clf: LogisticRegression | None = None

    @torch.no_grad()
    def _extract(self, prompt: str) -> np.ndarray:
        """Last-token hidden state from ALL layers, concatenated (the paper's recipe)."""
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs)
        feats = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
        return np.concatenate(feats)

    def build_features(self, prompts: list[str]) -> np.ndarray:
        return np.stack([self._extract(p) for p in prompts])

    def train(self, prompts: list[str], labels: list[int]) -> "ToolProbe":
        """labels: 1 = tool needed, 0 = answer directly."""
        X = self.scaler.fit_transform(self.build_features(prompts))
        self.clf = LogisticRegression(C=1.0, penalty="l2", max_iter=2000)
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, prompt: str) -> float:
        assert self.clf is not None, "Probe not trained. Call .train() first."
        X = self.scaler.transform(self._extract(prompt).reshape(1, -1))
        return float(self.clf.predict_proba(X)[0, 1])

    def needs_tool(self, prompt: str) -> bool:
        return self.predict_proba(prompt) >= self.cfg.tau

    def save(self, path="artifacts/toolprobe.joblib"):
        joblib.dump({"clf": self.clf, "scaler": self.scaler, "cfg": self.cfg}, path)

    def load(self, path="artifacts/toolprobe.joblib") -> "ToolProbe":
        d = joblib.load(path)
        self.clf, self.scaler, self.cfg = d["clf"], d["scaler"], d["cfg"]
        return self