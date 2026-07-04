from toolgate import ToolGateConfig, ToolProbe
from toolgate.data import load_json_dataset


def test_probe_trains_and_predicts():
    cfg = ToolGateConfig()
    probe = ToolProbe(cfg)
    prompts, labels = load_json_dataset("data/toy_when2tool.json")
    probe.train(prompts, labels)
    p = probe.predict_proba("What is 5 * 5?")
    assert 0.0 <= p <= 1.0