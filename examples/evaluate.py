"""The money shot: sweep tau, plot tool-call rate vs accuracy."""
import matplotlib.pyplot as plt
from toolgate import ToolGateConfig, ToolProbe
from toolgate.data import load_json_dataset

cfg = ToolGateConfig()
probe = ToolProbe(cfg)
prompts, labels = load_json_dataset("data/toy_when2tool.json")
probe.train(prompts, labels)

probs = [probe.predict_proba(p) for p in prompts]

taus, call_rates, accs = [], [], []
for tau in [i / 20 for i in range(1, 20)]:
    preds = [1 if p >= tau else 0 for p in probs]
    call_rate = sum(preds) / len(preds)
    acc = sum(int(pr == lb) for pr, lb in zip(preds, labels)) / len(labels)
    taus.append(tau); call_rates.append(call_rate); accs.append(acc)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(taus, call_rates, "b-o", label="Tool-call rate")
ax1.set_xlabel("tau (threshold)"); ax1.set_ylabel("Tool-call rate", color="b")
ax2 = ax1.twinx()
ax2.plot(taus, accs, "r-s", label="Accuracy")
ax2.set_ylabel("Accuracy", color="r")
plt.title("ToolGate: cost vs accuracy trade-off")
plt.tight_layout()
plt.savefig("artifacts/tradeoff.png", dpi=150)
print("Saved artifacts/tradeoff.png")