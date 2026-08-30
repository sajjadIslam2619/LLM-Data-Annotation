import glob, re, sys
sys.path.insert(0, ".")
import pandas as pd
from score_utils import score_predictions

VARIANT = "C_long_definitions"
OPENAI = {"gpt-3.5-turbo","gpt-4o","gpt-4o-mini","gpt-5.4","gpt-5.4-mini","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra"}
ANTHROPIC = {"claude-opus-5","claude-sonnet-5"}

rows = []
files = sorted(glob.glob(f"{VARIANT}_*_results.csv"))
for fp in files:
    m = re.match(rf"^{re.escape(VARIANT)}_(.+)_results\.csv$", fp)
    model = m.group(1)
    df = pd.read_csv(fp)
    pred_col = f"pred_{VARIANT}"
    if pred_col not in df.columns:
        print("MISSING pred col in", fp, df.columns.tolist())
        continue
    metrics = score_predictions(df, pred_col=pred_col)
    backend = "openai" if model in OPENAI else ("anthropic" if model in ANTHROPIC else "ollama")
    rows.append({
        "backend": backend, "model": model, "status": "ok",
        "accuracy": round(metrics["accuracy"], 3),
        "macro_f1": round(metrics["macro_f1"], 3),
        "unparseable": metrics["n_unparseable"],
    })

summary_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, na_position="last")
print(summary_df.to_string(index=False))
summary_df.to_csv(f"full_sweep_{VARIANT}_summary.csv", index=False)
print(f"\nSaved: full_sweep_{VARIANT}_summary.csv  ({len(rows)} models)")
