"""
Bulk version of llm_as_annotator_agreement.py: scans every *_results.csv in
this folder (from run_ollama_compare.py, run_openai_ablation.py, and
run_new_experiments.py) that has the 8 reviewer columns, finds every
pred_<variant> column in it, computes LLM-vs-human Cohen's Kappa for each,
and writes ONE combined summary instead of running the single-file script
by hand over and over.

No new model calls -- pure recomputation from CSVs you already have,
including files from earlier phases (openai_ablation_results.csv,
ollama_<model>_results.csv), not just the new D/I/J/K files.

Usage:
  python run_all_agreement_checks.py
"""

import glob

import numpy as np
import pandas as pd

from llm_as_annotator_agreement import cohen_kappa, REVIEWER_COLS


def human_human_mean(df: pd.DataFrame) -> float:
    kappas = []
    for i in range(len(REVIEWER_COLS)):
        for j in range(i + 1, len(REVIEWER_COLS)):
            kappas.append(cohen_kappa(df[REVIEWER_COLS[i]], df[REVIEWER_COLS[j]]))
    return float(np.mean(kappas))


def main():
    rows = []
    human_baseline = None

    for f in sorted(glob.glob("*_results.csv")):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue

        if not all(c in df.columns for c in REVIEWER_COLS):
            continue  # synthetic neutralization-pair files etc. -- no real reviewer labels

        if human_baseline is None:
            human_baseline = human_human_mean(df)
            print(f"Human-human baseline mean Kappa: {human_baseline:.3f} (from {f}, should be ~0.156 for everyone)")

        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        for pred_col in pred_cols:
            llm_kappas = [cohen_kappa(df[pred_col], df[r]) for r in REVIEWER_COLS]
            mean_k = float(np.mean(llm_kappas))
            rows.append({
                "file": f,
                "pred_col": pred_col,
                "llm_human_mean_kappa": round(mean_k, 3),
                "delta_vs_human_human": round(mean_k - human_baseline, 3),
                "n_reviewer_pairs": len(REVIEWER_COLS),
            })
            print(f"  {f} / {pred_col}: kappa={mean_k:.3f} delta={mean_k - human_baseline:+.3f}")

    if not rows:
        print("No *_results.csv files with reviewer columns found.")
        return

    out = pd.DataFrame(rows).sort_values("delta_vs_human_human", ascending=False)
    out.to_csv("agreement_summary_all.csv", index=False)

    lines = [
        "# LLM-as-annotator agreement -- summary across all models/variants",
        "",
        f"Human-human baseline mean Kappa: {human_baseline:.3f} (28 pairs among the 8 reviewers, from Phase 1 EDA)",
        "",
        f"{(out['delta_vs_human_human'] > 0).sum()} of {len(out)} model/variant runs EXCEED human-human agreement.",
        "",
        "| File | Prediction column | LLM-human mean Kappa | Delta vs. human baseline |",
        "|---|---|---|---|",
    ]
    for _, r in out.iterrows():
        lines.append(f"| {r['file']} | {r['pred_col']} | {r['llm_human_mean_kappa']:.3f} | {r['delta_vs_human_human']:+.3f} |")
    with open("AGREEMENT_SUMMARY.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nWrote agreement_summary_all.csv and AGREEMENT_SUMMARY.md ({len(out)} rows)")


if __name__ == "__main__":
    main()
