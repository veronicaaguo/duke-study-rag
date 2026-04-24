# Evaluation & Error Analysis
# Run this notebook after `scripts/evaluate.py` and `scripts/run_ablation.py`

# ── Cell 1: Imports ────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ── Cell 2: Load ablation results ─────────────────────────────────────────────
df = pd.read_csv("../data/ablation_results.csv")
print(df.to_string(index=False))

# ── Cell 3: Ablation heatmap (chunking vs retrieval mode) ─────────────────────
# Pivot: rows = chunking strategy, cols = hybrid+rerank combo, values = ROUGE-L
pivot = df.pivot_table(
    index="chunking",
    columns=["hybrid_search", "reranker"],
    values="rouge_l"
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
ax.set_title("ROUGE-L by chunking strategy and retrieval mode")
plt.tight_layout()
plt.savefig("../data/figures/ablation_heatmap.png", dpi=150)
plt.show()

# ── Cell 4: Latency vs quality tradeoff ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for strategy in df["chunking"].unique():
    sub = df[df["chunking"] == strategy]
    ax.scatter(sub["latency_s"], sub["rouge_l"], label=strategy, s=80)
ax.set_xlabel("Latency (s)")
ax.set_ylabel("ROUGE-L")
ax.set_title("Quality vs latency tradeoff across pipeline configurations")
ax.legend()
plt.tight_layout()
plt.savefig("../data/figures/latency_quality.png", dpi=150)
plt.show()

# ── Cell 5: Error analysis ─────────────────────────────────────────────────────
# Load per-question eval results (output of evaluate.py)
with open("../data/eval_results_best_pipeline.json") as f:
    results = json.load(f)

# Find failures: faithfulness=0 or rouge_l < 0.2
failures = [r for r in results if r["faithfulness"] == 0 or r["rouge_l"] < 0.2]
print(f"\nFailure cases: {len(failures)} / {len(results)}")

# Categorize failures manually:
# - retrieval_miss: relevant source not in retrieved chunks
# - hallucination: answer not grounded in context (faithfulness=0)
# - context_mismatch: chunks retrieved but answer still wrong

for i, f in enumerate(failures[:5], 1):
    print(f"\n--- Failure {i} ---")
    print(f"Q: {f['question']}")
    print(f"ROUGE-L: {f['rouge_l']} | Faithfulness: {f['faithfulness_label']}")
    print(f"Answer snippet: {f['answer'][:200]}...")

# ── Cell 6: Failure type distribution ─────────────────────────────────────────
# After manually tagging failure types, visualize distribution
# (fill in after running eval)
failure_types = {
    "Retrieval miss": 0,
    "Hallucination": 0,
    "Context mismatch": 0,
    "Correct (borderline)": 0,
}
# ... fill in counts from manual review ...

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(failure_types.keys(), failure_types.values(), color=["#e74c3c","#e67e22","#3498db","#2ecc71"])
ax.set_title("Failure case categories")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../data/figures/failure_types.png", dpi=150)
plt.show()

# ── Cell 7: Iteration 2 improvement summary ────────────────────────────────────
# After tuning based on error analysis, re-run evaluate.py and compare
# Round 1: baseline custom pipeline
# Round 2: after adjusting chunk size / reranker threshold based on failures
# Document: what changed, what you measured, what improved
