"""
One entry point for the paper-inspired additions (arXiv:2307.05052, flagged
by Sajjad): label flipping (I, J), complementary explanations (K), and the
input-neutralization robustness test. Runs everything against local Ollama
models only -- no OpenAI/API key needed or used anywhere in this script.

Runs one task at a time by design (loop over --models x --steps), so if
something crashes or you want to stop partway, just rerun -- already-done
rows in the tracker are skipped unless --force is passed.

Every completed run updates experiment_tracker.csv and regenerates
EXPERIMENT_TRACKER.md (a plain markdown table) in this folder. Send
EXPERIMENT_TRACKER.md to Sajjad any time -- it always reflects the latest
state, so it doubles as "here's what's done, here's what's still pending."

Examples:
  # everything, on the recommended model set (see RECOMMENDED_MODELS below)
  python run_new_experiments.py

  # just one model, one step at a time
  python run_new_experiments.py --models qwen3:8b --steps ablation
  python run_new_experiments.py --models qwen3:8b --steps neutralization

  # a model Sajjad specifically asks about, not in the recommended set
  python run_new_experiments.py --models phi4-mini:3.8b --steps ablation,neutralization

  # re-run something already marked done in the tracker
  python run_new_experiments.py --models qwen3:8b --steps ablation --force
"""

import argparse
import csv
import os
from datetime import datetime, timezone

import pandas as pd

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output, label_key
from run_ollama_compare import call_ollama
from score_utils import score_predictions
from test_input_neutralization import NEUTRALIZATION_PAIRS

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"
TRACKER_CSV = "experiment_tracker.csv"
TRACKER_MD = "EXPERIMENT_TRACKER.md"
# D_few_shot runs alongside I/J/K on every model so there's always a clean
# same-model baseline to measure "how much did flipping/explaining change
# things" against -- comparing I/J/K to a *different* model's D score, or to
# that model's own H/F score (different scaffolding entirely), isn't a fair
# comparison.
ABLATION_VARIANTS = ["D_few_shot", "I_label_flipped_random", "J_label_flipped_adversarial", "K_complementary_explanations"]
NEUTRALIZATION_VARIANT = "D_few_shot"  # which prompt variant to test robustness under

# Prioritized model list: the best-performing local models found so far
# (RESEARCH_LOG.md Phase 8/10/11), covering two axes on purpose:
#   - size: llama3.2 / gemma3:4b / phi4-mini:3.8b (small) vs. qwen3:8b /
#     llama3.1:8b / internlm2:7b / yi:6b / llama3:8b (7-8B) -- the paper's
#     core finding is that label flipping affects large models much more
#     than small ones, so this needs a real small-model cluster to compare
#     against, not just one data point.
#   - demo-sensitivity outlier: exaone3.5:7.8b jumped from 0.283 macro F1
#     (Variant H, no examples) to 0.707 (Variant F, few-shot) in Phase 11 --
#     the single biggest variant-swing of any model tested, which makes it
#     the model most likely to show a real label-flipping effect.
# Deliberately excluded from the default set: models already scoring near
# random under every variant tested (falcon3, granite3.3, olmo2) add noise,
# not signal, when testing whether flipping "degrades" something that's
# already broken. deepseek-r1:8b (reasoning model, unpredictable chain-of-
# thought, ~40+ min/run) is worth a dedicated one-off run separately -- pass
# it via --models rather than bundling it into the default batch.
RECOMMENDED_MODELS = [
    "qwen3:8b", "llama3.1:8b", "internlm2:7b", "yi:6b", "llama3:8b",
    "exaone3.5:7.8b", "gemma3:4b", "phi4-mini:3.8b", "llama3.2",
]

TRACKER_FIELDS = ["model", "task", "status", "accuracy", "macro_f1", "flip_rate", "downgrades", "timestamp"]


def load_tracker() -> dict:
    """Returns {(model, task): row_dict}."""
    if not os.path.exists(TRACKER_CSV):
        return {}
    rows = {}
    with open(TRACKER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["model"], row["task"])] = row
    return rows


def save_tracker(rows: dict):
    with open(TRACKER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)

    # Regenerate the human-readable version every time -- this is the file
    # to actually send to Sajjad.
    pending = [r for r in rows.values() if r["status"] == "pending"]
    done = [r for r in rows.values() if r["status"] == "done"]
    lines = [
        "# Experiment tracker -- paper-inspired additions (label flipping, "
        "complementary explanations, input neutralization)",
        "",
        f"_Last updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z_",
        "",
        f"## Still needs to run ({len(pending)})",
        "",
        "| Model | Task | Notes |",
        "|---|---|---|",
    ]
    for r in sorted(pending, key=lambda r: (r["model"], r["task"])):
        lines.append(f"| {r['model']} | {r['task']} | |")
    lines += ["", f"## Done ({len(done)})", "", "| Model | Task | Accuracy | Macro F1 | Flip rate | Downgrades | When |", "|---|---|---|---|---|---|---|"]
    for r in sorted(done, key=lambda r: (r["model"], r["task"])):
        lines.append(
            f"| {r['model']} | {r['task']} | {r['accuracy']} | {r['macro_f1']} | "
            f"{r['flip_rate']} | {r['downgrades']} | {r['timestamp']} |"
        )
    with open(TRACKER_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def seed_tracker(rows: dict, models: list[str]):
    tasks = ABLATION_VARIANTS + ["neutralization"]
    for model in models:
        for task in tasks:
            key = (model, task)
            if key not in rows:
                rows[key] = {"model": model, "task": task, "status": "pending",
                             "accuracy": "", "macro_f1": "", "flip_rate": "", "downgrades": "", "timestamp": ""}
    return rows


def run_ablation(model: str, variant_name: str, sample: pd.DataFrame) -> dict:
    builder_fn = PROMPT_VARIANTS[variant_name]
    system_prompt, user_template = builder_fn()
    predictions = []
    for _, row in sample.iterrows():
        try:
            raw = call_ollama(model, system_prompt, user_template.format(tweet=row["tweet"]))
        except Exception as e:
            raw = f"__ERROR__: {e}"
        predictions.append(normalize_model_output(raw))
    out = sample.copy()
    out[f"pred_{variant_name}"] = predictions
    metrics = score_predictions(out, pred_col=f"pred_{variant_name}")

    model_tag = model.replace(":", "-")
    out.to_csv(f"{variant_name}_{model_tag}_results.csv", index=False)
    return metrics


def run_neutralization(model: str) -> dict:
    system_prompt, user_template = PROMPT_VARIANTS[NEUTRALIZATION_VARIANT]()
    rows = []
    for name, true_label, explicit_tweet, implicit_tweet in NEUTRALIZATION_PAIRS:
        for version, tweet in (("explicit", explicit_tweet), ("implicit", implicit_tweet)):
            try:
                raw = call_ollama(model, system_prompt, user_template.format(tweet=tweet))
            except Exception as e:
                raw = f"__ERROR__: {e}"
            pred = normalize_model_output(raw)
            rows.append({"pair": name, "version": version, "true_label": true_label, "predicted_label": pred})
    df = pd.DataFrame(rows)

    n_flips, n_downgrades = 0, 0
    for name, group in df.groupby("pair"):
        ek = label_key(group[group.version == "explicit"]["predicted_label"].iloc[0])
        ik = label_key(group[group.version == "implicit"]["predicted_label"].iloc[0])
        if ek != ik:
            n_flips += 1
            ei, ii = (int(ek) if ek.isdigit() else None), (int(ik) if ik.isdigit() else None)
            ti = int(label_key(group["true_label"].iloc[0]))
            if ei is not None and ii is not None and ii < ti <= ei:
                n_downgrades += 1

    model_tag = model.replace(":", "-")
    df.to_csv(f"neutralization_{NEUTRALIZATION_VARIANT}_{model_tag}_results.csv", index=False)
    return {"n_pairs": len(NEUTRALIZATION_PAIRS), "n_flips": n_flips, "n_downgrades": n_downgrades}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(RECOMMENDED_MODELS),
                         help="Comma-separated Ollama model tags")
    parser.add_argument("--steps", default="ablation,neutralization",
                         help="Comma-separated: ablation, neutralization")
    parser.add_argument("--limit", type=int, default=149, help="Tweets to test for the ablation step")
    parser.add_argument("--force", action="store_true", help="Re-run even if tracker already marks it done")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    tracker = load_tracker()
    tracker = seed_tracker(tracker, models)
    save_tracker(tracker)

    df = pd.read_csv(GROUND_TRUTH_CSV)
    sample = df.sample(n=args.limit, random_state=42).reset_index(drop=True) if args.limit < len(df) else df

    for model in models:
        if "ablation" in steps:
            for variant_name in ABLATION_VARIANTS:
                key = (model, variant_name)
                if tracker[key]["status"] == "done" and not args.force:
                    print(f"[skip] {model} / {variant_name} already done (use --force to re-run)")
                    continue
                print(f"\n[run] {model} / {variant_name} on {len(sample)} tweets...")
                metrics = run_ablation(model, variant_name, sample)
                tracker[key] = {
                    "model": model, "task": variant_name, "status": "done",
                    "accuracy": f"{metrics['accuracy']:.3f}", "macro_f1": f"{metrics['macro_f1']:.3f}",
                    "flip_rate": "", "downgrades": "",
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
                }
                save_tracker(tracker)
                print(f"  accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")

        if "neutralization" in steps:
            key = (model, "neutralization")
            if tracker[key]["status"] == "done" and not args.force:
                print(f"[skip] {model} / neutralization already done (use --force to re-run)")
                continue
            print(f"\n[run] {model} / neutralization ({NEUTRALIZATION_VARIANT})...")
            result = run_neutralization(model)
            tracker[key] = {
                "model": model, "task": "neutralization", "status": "done",
                "accuracy": "", "macro_f1": "",
                "flip_rate": f"{result['n_flips']}/{result['n_pairs']}",
                "downgrades": str(result["n_downgrades"]),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
            }
            save_tracker(tracker)
            print(f"  flips={result['n_flips']}/{result['n_pairs']} downgrades={result['n_downgrades']}")

    print(f"\nDone. See {TRACKER_MD} for the current status table -- safe to send to Sajjad as-is.")


if __name__ == "__main__":
    main()
