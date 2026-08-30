import re, glob
import pandas as pd
from score_utils import score_predictions

def canon(m):
    return re.sub(r'-(\d+(?:\.\d+)?b)$', r':\1', m)

VARIANTS = ["A_bare_labels","B_short_definitions","C_long_definitions","D_few_shot",
            "E_chain_of_thought","F_refined_definitions","H_reviewer_matched",
            "I_label_flipped_random","J_label_flipped_adversarial","K_complementary_explanations"]

SELECTED = {"llama3.1:8b","hermes3:8b","qwen2.5:7b","gpt-5.4-mini","gpt-5.6-sol","gpt-5.6-terra"}

EXCLUDE_FILES = {
    "openai_ablation_results.csv","ollama_deepseek-r1:8b_results.csv",
    "ollama_qwen3:8b_results.csv","ollama_yi:6b_results.csv",
    "anthropic_ablation_claude-sonnet-5_results.csv",
}

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
        if cmodel not in SELECTED:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        pred_col = f"pred_{variant}"
        if pred_col not in df.columns:
            continue
        metrics = score_predictions(df, pred_col=pred_col)
        cur = best.get(cmodel)
        if cur is None or metrics["n_unparseable"] < cur["n_unparseable"]:
            best[cmodel] = {**metrics, "path": path}

    for cmodel, metrics in best.items():
        rows.append({
            "variant": variant, "model": cmodel,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "n_scored": metrics["n_scored"],
            "unparseable": metrics["n_unparseable"],
        })

df_out = pd.DataFrame(rows).sort_values(["model","variant"])
df_out.to_csv("_selected_models_by_variant.csv", index=False)
print(df_out.to_string(index=False))
print("\ncoverage:")
print(df_out.groupby('model')['variant'].nunique())
