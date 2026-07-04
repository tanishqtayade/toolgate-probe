"""
ToolGate Benchmark
Evaluates routing accuracy, latency, and cost savings.

Usage:
    python benchmark.py                      # uses toy dataset
    python benchmark.py --dataset my_data.json
"""

import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from toolgate import ToolGateConfig, ToolProbe, ToolGate
from toolgate.data import load_json_dataset

TOOL_CALL_COST  = 0.020   # USD per call (e.g. search API)
DIRECT_COST     = 0.001   # USD per direct LLM answer

def run_benchmark(dataset_path: str, tau: float = 0.5, output_dir: str = "benchmark_results"):
    Path(output_dir).mkdir(exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    prompts, labels = load_json_dataset(dataset_path)
    print(f"  {len(prompts)} examples  ({sum(labels)} tool-needed, {len(labels)-sum(labels)} direct)")

    print(f"\nTraining probe (tau={tau})...")
    cfg   = ToolGateConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", tau=tau)
    probe = ToolProbe(cfg)

    # Train on 80%, test on 20%
    split   = int(len(prompts) * 0.8)
    train_p, test_p = prompts[:split], prompts[split:]
    train_l, test_l = labels[:split],  labels[split:]

    probe.train(train_p, train_l)
    gate = ToolGate(probe)

    print(f"\nRunning inference on {len(test_p)} test examples...")
    preds, probs, latencies = [], [], []

    for prompt in test_p:
        t0     = time.perf_counter()
        result = gate.generate(prompt, max_new_tokens=1)   # 1 token — we only need the routing decision
        lat    = (time.perf_counter() - t0) * 1000
        preds.append(int(result["tool_needed"]))
        probs.append(result["prob"])
        latencies.append(lat)

    # ── Classification metrics ─────────────────────────────────────────────
    acc  = accuracy_score(test_l, preds)
    prec = precision_score(test_l, preds, zero_division=0)
    rec  = recall_score(test_l, preds, zero_division=0)
    f1   = f1_score(test_l, preds, zero_division=0)
    cm   = confusion_matrix(test_l, preds)
    fpr, tpr, _ = roc_curve(test_l, probs)
    roc_auc      = auc(fpr, tpr)

    # ── Cost analysis ──────────────────────────────────────────────────────
    n          = len(test_l)
    cost_base  = n * TOOL_CALL_COST          # baseline: always call tool
    tool_calls = sum(preds)
    direct     = n - tool_calls
    cost_gate  = tool_calls * TOOL_CALL_COST + direct * DIRECT_COST
    saved      = cost_base - cost_gate
    saved_pct  = saved / cost_base * 100 if cost_base > 0 else 0

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("TOOLGATE BENCHMARK RESULTS")
    print("="*50)
    print(f"\n  Dataset       : {dataset_path}")
    print(f"  Train / Test  : {len(train_p)} / {len(test_p)}")
    print(f"  Tau threshold : {tau}")

    print(f"\n── Classification ──────────────────────────────")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"  F1 score  : {f1*100:.1f}%")
    print(f"  ROC-AUC   : {roc_auc:.3f}")
    print(f"\n  Confusion matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\n── Latency ─────────────────────────────────────")
    print(f"  Mean   : {np.mean(latencies):.0f}ms")
    print(f"  Median : {np.median(latencies):.0f}ms")
    print(f"  P95    : {np.percentile(latencies, 95):.0f}ms")

    print(f"\n── Cost (per {n} queries) ──────────────────────")
    print(f"  Without ToolGate : ${cost_base:.4f}")
    print(f"  With    ToolGate : ${cost_gate:.4f}")
    print(f"  Saved            : ${saved:.4f}  ({saved_pct:.1f}%)")

    # ── Save JSON results ──────────────────────────────────────────────────
    results = {
        "dataset": dataset_path, "tau": tau,
        "n_train": len(train_p), "n_test": len(test_p),
        "accuracy": round(acc, 4),   "precision": round(prec, 4),
        "recall":   round(rec, 4),   "f1":        round(f1, 4),
        "roc_auc":  round(roc_auc, 4),
        "latency_mean_ms":   round(np.mean(latencies), 1),
        "latency_median_ms": round(np.median(latencies), 1),
        "latency_p95_ms":    round(np.percentile(latencies, 95), 1),
        "cost_baseline": round(cost_base, 6),
        "cost_with_gate": round(cost_gate, 6),
        "cost_saved": round(saved, 6),
        "cost_saved_pct": round(saved_pct, 1),
    }
    out_json = f"{output_dir}/results.json"
    Path(out_json).write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {out_json}")

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("ToolGate Benchmark", fontsize=13, fontweight="normal")

    # 1. ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, color="#2a78d6", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], "--", color="#aaa", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(fontsize=10)

    # 2. Cost comparison
    ax = axes[1]
    bars = ax.bar(["Without ToolGate", "With ToolGate"],
                  [cost_base, cost_gate],
                  color=["#f09595", "#97c459"], width=0.5)
    ax.bar_label(bars, fmt="$%.4f", padding=3, fontsize=9)
    ax.set_ylabel("Cost (USD)"); ax.set_title("Cost comparison")

    # 3. Latency distribution
    ax = axes[2]
    ax.hist(latencies, bins=15, color="#2a78d6", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(latencies), color="#e34948", lw=1.5, linestyle="--",
               label=f"mean {np.mean(latencies):.0f}ms")
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
    ax.set_title("Latency distribution"); ax.legend(fontsize=9)

    plt.tight_layout()
    out_png = f"{output_dir}/benchmark.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Charts saved  → {out_png}")
    plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/toy_when2tool.json")
    parser.add_argument("--tau",     type=float, default=0.5)
    parser.add_argument("--output",  default="benchmark_results")
    args = parser.parse_args()
    run_benchmark(args.dataset, args.tau, args.output)