"""
Scoring helpers shared by the OpenAI and Ollama runner scripts.
"""

import pandas as pd
from labels import label_key


def score_predictions(df: pd.DataFrame, pred_col: str, truth_col: str = "ground_truth_label",
                       status_col: str = "agreement_status"):
    """Compare predicted labels to ground truth and return a metrics dict.

    Rows where the human reviewers tied (no clear majority) are reported
    separately since they aren't reliable ground truth.
    """
    clean = df[df[status_col] == "majority"].copy()

    clean["truth_key"] = clean[truth_col].apply(label_key)
    clean["pred_key"] = clean[pred_col].apply(label_key)

    n = len(clean)
    n_correct = (clean["truth_key"] == clean["pred_key"]).sum()
    accuracy = n_correct / n if n else 0.0

    # Per-class precision/recall/F1 without extra dependencies
    per_class = {}
    for key in sorted(clean["truth_key"].unique()):
        tp = ((clean["pred_key"] == key) & (clean["truth_key"] == key)).sum()
        fp = ((clean["pred_key"] == key) & (clean["truth_key"] != key)).sum()
        fn = ((clean["pred_key"] != key) & (clean["truth_key"] == key)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[key] = {"precision": precision, "recall": recall, "f1": f1, "support": int((clean["truth_key"] == key).sum())}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    n_unparseable = (clean["pred_key"] == "UNPARSEABLE").sum()

    return {
        "n_scored": n,
        "n_correct": int(n_correct),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_unparseable": int(n_unparseable),
        "per_class": per_class,
    }


def print_report(variant_name: str, model_name: str, metrics: dict):
    print(f"\n=== {variant_name} | {model_name} ===")
    print(f"Scored on {metrics['n_scored']} tweets (ties excluded)")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Macro F1:  {metrics['macro_f1']:.3f}")
    print(f"Unparseable responses: {metrics['n_unparseable']}")
    print("Per-class:")
    for key, v in metrics["per_class"].items():
        print(f"  class {key}: precision={v['precision']:.2f} recall={v['recall']:.2f} "
              f"f1={v['f1']:.2f} (n={v['support']})")
