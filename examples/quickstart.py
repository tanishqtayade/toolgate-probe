from toolgate import ToolGateConfig, ToolProbe, ToolGate
from toolgate.data import load_json_dataset

cfg = ToolGateConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", tau=0.5)

probe = ToolProbe(cfg)
prompts, labels = load_json_dataset("data/toy_when2tool.json")
probe.train(prompts, labels)
probe.save()

gate = ToolGate(probe)

tests = ["What is 918273 * 33?", "What is the capital of Japan?"]
for q in tests:
    r = gate.generate(q, max_new_tokens=60)
    print(f"\nQ: {q}")
    print(f"  tool_needed={r['tool_needed']}  (p={r['prob']:.2f})")
    print(f"  -> {r['response'][:150]}")