from toolgate import ToolGateConfig, ToolProbe, ToolGate
from toolgate.data import load_json_dataset


def test_gate_generates():
    cfg = ToolGateConfig(max_new_tokens=20)
    probe = ToolProbe(cfg)
    prompts, labels = load_json_dataset("data/toy_when2tool.json")
    probe.train(prompts, labels)
    gate = ToolGate(probe)
    r = gate.generate("What is the capital of France?", max_new_tokens=20)
    assert "response" in r and isinstance(r["tool_needed"], bool)