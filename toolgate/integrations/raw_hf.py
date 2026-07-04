"""Convenience: build a gate directly from a dataset."""
from ..config import ToolGateConfig
from ..probe import ToolProbe
from ..gate import ToolGate
from ..data import load_json_dataset


def build_gate_from_dataset(dataset_path: str, config: ToolGateConfig | None = None) -> ToolGate:
    probe = ToolProbe(config)
    prompts, labels = load_json_dataset(dataset_path)
    probe.train(prompts, labels)
    return ToolGate(probe)