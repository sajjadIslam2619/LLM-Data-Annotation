"""
Add Micro F1 and Weighted F1 alongside the existing Macro F1 numbers.

Why this exists: Sajjad asked (email, Aug 2026) whether Micro F1 (weighted)
should be reported instead of Macro F1, since the label distribution is
imbalanced (class 1 "No Concern" dominates). This script answers that
without any new model calls -- it rescans the same per-tweet `*_results.csv`
files `compute_precision_recall.py` already uses.

Two different metrics get computed, since "Micro F1 (weighted)" in the email
actually names two distinct things:

  Micro F1    -- pool every prediction into one big TP/FP/FN count, then take
                 one F1 over the pool. For single-label multiclass tasks this
                 is very close to (but NOT always exactly equal to) accuracy;
                 it differs here because UNPARSEABLE predictions count as a
                 false negative for the true class but never as a false
                 positive for any class (UNPARSEABLE isn't one of the 5
                 label classes), so micro-precision and micro-recall aren't
                 forced equal the way they would be with no unparseable
                 responses.

  Weighted F1 -- the per-class F1 scores already in per_class_metrics.csv,
                 averaged with each class weighted by its support (how many
                 tweets actually belong to that class). This is what most
                 people mean by "using Macro F1 is unfair to imbalanced
                 data, use something support-weighted instead" -- it's the
                 metric to reach for if Sajjad wants imbalance accounted for
                 without simply reporting accuracy again.

Usage:
  python compute_micro_f1.py --variants F_refined_definitions,H_reviewer_matched
  python compute_micro_f1.py   # all variants
"""

import argparse
import glob
import os

import pandas as pd

from labels import label_key
from compute_precision_recall import parse_filename, CLASS_NAMES


def score_micro_weighted(df: pd.DataFrame, pred_col: str, truth_col: str = "ground_truth_label",
                          status_col: str = "agreement_status"):
    clean = df[df[status_col] == "majority"].copy()
    clean["truth_key"] = clean[truth_col].apply(label_key)
    clean["pred_key"] = clean[pred_col].apply(label_key)

    n = len(clean)
    if n == 0:
        return None

    sum_tp = sum_fp = sum_fn = 0
    weighted_num = 0.0
    weighted_den = 0

    for key in sorted(clean["truth_key"].unique()):
        tp = int(((clean["pred_key"] == key) & (clean["truth_key"] == key)).sum())
        fp = int(((clean["pred_key"] == key) & (clean["truth_key"] != key)).sum())
        fn = int(((clean["pred_key"] != key) & (clean["truth_key"] == key)).sum())
        support = int((clean["truth_key"] == key).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        weighted_num += f1 * support
        weighted_den += support

    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
    weighted_f1 = weighted_num / weighted_den if weighted_den else 0.0
    accuracy = sum_tp / n

    return {
        "n_scored": n,
        "accuracy": round(accuracy, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variant names to restrict to, e.g. "
                             "F_refined_definitions,H_reviewer_matched. Default: all.")
    parser.add_argument("--pattern", default="*_results.csv")
    parser.add_argument("--out", default="micro_weighted_f1.csv")
    args = parser.parse_args()

    wanted_variants = set(v.strip() for v in args.variants.split(",")) if args.variants else None

    files = sorted(glob.glob(args.pattern))
    files = [f for f in files if not os.path.basename(f).startswith("neutralization_")]

    rows = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "ground_truth_label" not in df.columns or "agreement_status" not in df.columns:
            continue
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        for pred_col in pred_cols:
            variant, model = parse_filename(path, pred_col)
            if wanted_variants and variant not in wanted_variants:
                continue
            metrics = score_micro_weighted(df, pred_col)
            if metrics is None:
                continue
            rows.append({"variant": variant, "model": model, **metrics})

    if not rows:
        print("No matching results. Check --variants / --pattern.")
        return

    out = pd.DataFrame(rows).sort_values(["variant", "weighted_f1"], ascending=[True, False])
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows)\n")
    print(out.to_string(index=False))
    print(
        "\nNote: Micro F1 for a single-label task is close to Accuracy but not "
        "identical here, since UNPARSEABLE predictions count against recall but "
        "not precision. Weighted F1 (per-class F1 weighted by support) is the "
        "metric that most directly answers 'does this look different once class "
        "imbalance is accounted for' -- compare it side-by-side with the existing "
        "Macro F1 numbers in macro_metrics.csv."
    )


if __name__ == "__main__":
    main()
