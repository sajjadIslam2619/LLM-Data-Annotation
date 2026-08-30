import re, glob, os
import pandas as pd
from score_utils import score_predictions
from compute_micro_f1 import score_micro_weighted

def canon(m):
    return re.sub(r'-(\d+(?:\.\d+)?b)$', r':\1', m)

VARIANTS = ["A_bare_labels","B_short_definitions","C_long_definitions","D_few_shot",
            "E_chain_of_thought","F_refined_definitions","H_reviewer_matched",
            "I_label_flipped_random","J_label_flipped_adversarial","K_complementary_explanations"]

EXCLUDE_FILES = {
    "openai_ablation_results.csv","ollama_deepseek-r1:8b_results.csv",
    "ollama_qwen3:8b_results.csv","ollama_yi:6b_results.csv",
    "anthropic_ablation_claude-sonnet-5_results.csv",
}

FLAG_THRESHOLD = 0.15  # >15% unparseable => flagged, excluded from "clean" averages

rows = []
for variant in VARIANTS:
    files = [f for f in glob.glob(f"{variant}_*_results.csv") if f not in EXCLUDE_FILES]
    best = {}
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
        macro = score_predictions(df, pred_col=pred_col)
        micro = score_micro_weighted(df, pred_col=pred_col)
        if micro is None:
            continue
        rec = {
            "variant": variant, "model": cmodel, "path": path,
            "n_scored": macro["n_scored"],
            "accuracy": round(macro["accuracy"], 4),
            "macro_f1": round(macro["macro_f1"], 4),
            "micro_f1": round(micro["micro_f1"], 4),
            "weighted_f1": round(micro["weighted_f1"], 4),
            "unparseable": macro["n_unparseable"],
        }
        rec["unparseable_rate"] = rec["unparseable"] / rec["n_scored"] if rec["n_scored"] else 1.0
        rec["flagged"] = rec["unparseable_rate"] > FLAG_THRESHOLD
        cur = best.get(cmodel)
        if cur is None or rec["unparseable"] < cur["unparseable"]:
            best[cmodel] = rec
    rows.extend(best.values())

master = pd.DataFrame(rows)
master.to_csv("_master_table.csv", index=False)
print(f"Master table: {len(master)} rows, {master['variant'].nunique()} variants, {master['model'].nunique()} models")
print("\nFlagged rows (unparseable > 15%):")
print(master[master['flagged']][['variant','model','n_scored','unparseable','unparseable_rate']].sort_values(['variant','unparseable_rate'], ascending=[True,False]).to_string(index=False))
