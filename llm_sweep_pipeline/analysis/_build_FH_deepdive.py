import re, glob, json
import pandas as pd
from score_utils import score_predictions
from labels import label_key

CLASS_NAMES = {
    "1": "No Concern", "2": "Distressed Self-Disclosure", "3": "Crisis, Seeking Support",
    "4": "Crisis, Not Seeking", "5": "Acute Crisis / SI", "6": "Not Relevant", "Other": "Other",
}

def canon(m):
    return re.sub(r'-(\d+(?:\.\d+)?b)$', r':\1', m)

VARIANTS = ["F_refined_definitions", "H_reviewer_matched"]

all_rows = []
per_class_rows = []

for variant in VARIANTS:
    files = glob.glob(f"{variant}_*_results.csv")
    best_by_model = {}  # canon_model -> (unparseable, path, model_raw)
    for path in files:
        m = re.match(rf"^{re.escape(variant)}_(.+)_results\.csv$", path)
        if not m:
            continue
        model_raw = m.group(1)
        cmodel = canon(model_raw)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        pred_col = f"pred_{variant}"
        if pred_col not in df.columns:
            continue
        metrics = score_predictions(df, pred_col=pred_col)
        key = cmodel
        cur = best_by_model.get(key)
        # prefer the file with fewer unparseable, tie-break by more n_scored
        if cur is None or metrics["n_unparseable"] < cur[0]["n_unparseable"]:
            best_by_model[key] = (metrics, path, model_raw)

    for cmodel, (metrics, path, model_raw) in best_by_model.items():
        all_rows.append({
            "variant": variant, "model": cmodel, "source_file": path,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "n_scored": metrics["n_scored"],
            "unparseable": metrics["n_unparseable"],
        })
        for key, v in metrics["per_class"].items():
            per_class_rows.append({
                "variant": variant, "model": cmodel, "class_key": key,
                "class_name": CLASS_NAMES.get(key, key),
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1": round(v["f1"], 4),
                "support": v["support"],
            })

summary_df = pd.DataFrame(all_rows).sort_values(["variant", "macro_f1"], ascending=[True, False])
per_class_df = pd.DataFrame(per_class_rows)

summary_df.to_csv("_FH_summary.csv", index=False)
per_class_df.to_csv("_FH_per_class.csv", index=False)

print(summary_df.to_string(index=False))
print("\nmodel counts per variant:")
print(summary_df['variant'].value_counts())
