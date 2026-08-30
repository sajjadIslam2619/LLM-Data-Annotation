import glob, os
import pandas as pd
from compute_micro_f1 import score_micro_weighted
from compute_precision_recall import parse_filename

EXCLUDE_FILES = {
    "openai_ablation_results.csv",          # superseded gpt-4o-mini-only A-E ablation dump
    "ollama_deepseek-r1:8b_results.csv",     # superseded single-model H dump
    "ollama_qwen3:8b_results.csv",           # superseded single-model D dump
    "ollama_yi:6b_results.csv",              # superseded single-model F dump
    "anthropic_ablation_claude-sonnet-5_results.csv",  # superseded single-model D dump
}
EXCLUDE_VARIANTS = {"G_official_reasoning_format"}  # incomplete/unverified this round

files = sorted(glob.glob("*_results.csv"))
files = [f for f in files if not os.path.basename(f).startswith("neutralization_")]
files = [f for f in files if os.path.basename(f) not in EXCLUDE_FILES]

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
        if variant in EXCLUDE_VARIANTS:
            continue
        metrics = score_micro_weighted(df, pred_col)
        if metrics is None:
            continue
        rows.append({"variant": variant, "model": model, **metrics})

out = pd.DataFrame(rows).sort_values(["variant", "weighted_f1"], ascending=[True, False])

# sanity check: no duplicate (variant, model) pairs
dupes = out[out.duplicated(subset=["variant", "model"], keep=False)]
if len(dupes):
    print("WARNING -- duplicate (variant, model) rows still present:")
    print(dupes.to_string(index=False))
else:
    print("OK: no duplicate (variant, model) pairs")

out.to_csv("micro_weighted_f1_no_G.csv", index=False)
print(f"\nWrote micro_weighted_f1_no_G.csv ({len(out)} rows, {out['variant'].nunique()} variants)")
print(out['variant'].value_counts().to_string())
