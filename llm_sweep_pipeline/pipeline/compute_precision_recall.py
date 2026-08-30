"""
Recompute per-class precision / recall / F1 for every completed run.

Why this exists: `score_utils.score_predictions()` has always computed
per-class precision and recall, but only accuracy / macro-F1 / unparseable
counts were ever persisted to the `*_summary.csv` files. The per-tweet
`*_results.csv` files still contain the predictions and the ground-truth
column, so precision and recall can be reconstructed offline -- no model
calls, no Ollama, no API keys, runs in seconds.

Outputs two files:
  per_class_metrics.csv  -- one row per (file, variant, model, class):
                            precision, recall, f1, support
  macro_metrics.csv      -- one row per (file, variant, model):
                            accuracy, macro precision / recall / F1,
                            unparseable count

Macro averages are taken over the classes actually present in the ground
truth (1, 2, 3, 5, 6 -- classes 4 and Other have zero support after
majority voting), matching how the existing macro-F1 numbers in
RESEARCH_LOG.md were computed. Ties (agreement_status != "majority") are
excluded, so every run is scored on the same 140 tweets.

Usage:
  python compute_precision_recall.py
  python compute_precision_recall.py --pattern "D_few_shot_*_results.csv"
"""

import argparse
import glob
import os
import re

import pandas as pd

from score_utils import score_predictions

# Human-readable names for the report, keyed by the leading label number.
CLASS_NAMES = {
    "1": "No Concern",
    "2": "Distressed Self-Disclosure",
    "3": "Crisis, Seeking Support",
    "4": "Crisis, Not Seeking Support",
    "5": "Acute Crisis / SI",
    "6": "Not Relevant",
    "Other": "Other",
}


def parse_filename(path: str, pred_col: str):
    """Recover (variant, model) from a results filename.

    'D_few_shot_llama3.1-8b_results.csv' -> ('D_few_shot', 'llama3.1-8b')
    'ollama_qwen3:8b_results.csv'        -> (from pred col, 'qwen3:8b')
    """
    stem = os.path.basename(path).replace("_results.csv", "")
    variant = pred_col.replace("pred_", "")

    if stem == "openai_ablation":
        # Legacy Phase 4 file; the only OpenAI model run in it was gpt-4o-mini.
        model = "gpt-4o-mini"
    elif stem.startswith(variant + "_"):
        model = stem[len(variant) + 1:]
    elif stem.startswith("ollama_"):
        model = stem[len("ollama_"):]
    elif stem.startswith("openai_"):
        model = stem[len("openai_"):]
    else:
        model = stem
    return variant, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*_results.csv",
                        help="Glob of results files to score (default: all).")
    parser.add_argument("--per-class-out", default="per_class_metrics.csv")
    parser.add_argument("--macro-out", default="macro_metrics.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    # The neutralization files are synthetic pairs with no ground-truth
    # column -- they are scored by a different script, skip them here.
    files = [f for f in files if not os.path.basename(f).startswith("neutralization_")]

    per_class_rows = []
    macro_rows = []
    skipped = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            skipped.append((path, f"unreadable: {exc}"))
            continue

        if "ground_truth_label" not in df.columns or "agreement_status" not in df.columns:
            skipped.append((path, "no ground-truth/agreement columns"))
            continue

        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        if not pred_cols:
            skipped.append((path, "no pred_* column"))
            continue

        for pred_col in pred_cols:
            metrics = score_predictions(df, pred_col)
            variant, model = parse_filename(path, pred_col)

            precisions, recalls = [], []
            for key, v in metrics["per_class"].items():
                per_class_rows.append({
                    "file": os.path.basename(path),
                    "variant": variant,
                    "model": model,
                    "class": key,
                    "class_name": CLASS_NAMES.get(key, key),
                    "precision": round(v["precision"], 4),
                    "recall": round(v["recall"], 4),
                    "f1": round(v["f1"], 4),
                    "support": v["support"],
                })
                precisions.append(v["precision"])
                recalls.append(v["recall"])

            # Flag runs whose numbers should not be quoted without a caveat:
            # partial runs (fewer than the full 140 scored tweets) and runs
            # where a large share of responses failed to parse.
            n = metrics["n_scored"]
            unp = metrics["n_unparseable"]
            warnings = []
            if n < 140:
                warnings.append(f"PARTIAL_RUN({n}/140)")
            if n and unp / n > 0.10:
                warnings.append(f"HIGH_UNPARSEABLE({unp}/{n})")

            n_classes = len(metrics["per_class"]) or 1
            macro_rows.append({
                "file": os.path.basename(path),
                "variant": variant,
                "model": model,
                "n_scored": n,
                "accuracy": round(metrics["accuracy"], 4),
                "macro_precision": round(sum(precisions) / n_classes, 4),
                "macro_recall": round(sum(recalls) / n_classes, 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "n_unparseable": unp,
                "warning": "; ".join(warnings),
            })

    if not macro_rows:
        print("No scoreable results files matched. Nothing written.")
        return

    per_class = pd.DataFrame(per_class_rows).sort_values(
        ["variant", "model", "class"]).reset_index(drop=True)
    macro = pd.DataFrame(macro_rows).sort_values(
        ["variant", "macro_f1"], ascending=[True, False]).reset_index(drop=True)

    per_class.to_csv(args.per_class_out, index=False)
    macro.to_csv(args.macro_out, index=False)

    print(f"Scored {len(macro)} model/variant runs from {len(files)} files.")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for path, why in skipped:
            print(f"  {os.path.basename(path)} -- {why}")

    print(f"\nWrote {args.per_class_out} ({len(per_class)} rows)")
    print(f"Wrote {args.macro_out} ({len(macro)} rows)")

    # Console preview: the class that matters most for the paper.
    print("\n--- Class 5 (Acute Crisis / SI) precision & recall, by run ---")
    c5 = per_class[per_class["class"] == "5"].sort_values("recall", ascending=False)
    if len(c5):
        print(c5[["variant", "model", "precision", "recall", "f1", "support"]]
              .to_string(index=False))
    else:
        print("(no class-5 rows found)")


if __name__ == "__main__":
    main()
