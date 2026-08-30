"""
Run the ablation study against OpenAI models.

What this does:
  1. Loads the ground-truth CSV (built by build_ground_truth.py).
  2. For each prompt variant in prompts.PROMPT_VARIANTS, sends every
     (sampled) tweet to the OpenAI API and records the predicted label.
  3. Scores each variant against the human majority-vote label.
  4. Prints a comparison table and saves detailed results to CSV.
  5. Optionally logs each variant as an MLflow run (--use-mlflow).

Setup:
  pip install openai pandas mlflow   # mlflow only needed if using --use-mlflow
  export OPENAI_API_KEY="sk-..."

Example:
  python run_openai_ablation.py --model gpt-4o-mini --limit 30
  python run_openai_ablation.py --model gpt-4o-mini --limit 30 --use-mlflow
  python run_openai_ablation.py --model gpt-4o-mini --variants A_bare_labels,C_long_definitions
"""

import argparse
import os
import time

import pandas as pd
from openai import OpenAI

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output
from score_utils import score_predictions, print_report

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"


def call_openai(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                 reasoning_effort: str = None) -> str:
    """Call the model, handling both standard models (gpt-4o-mini, etc., which
    support temperature) and reasoning models (gpt-5.x, o-series, etc., which
    reject the temperature parameter entirely).

    We try with temperature=0 first (best for reproducible classification
    output). If the API rejects it because this model doesn't support
    temperature, we transparently retry without it.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"model": model, "messages": messages}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        resp = client.chat.completions.create(temperature=0, **kwargs)
    except Exception as e:
        if "temperature" in str(e).lower():
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content


def run_variant(client, model, variant_name, builder_fn, tweets_df, sleep_s=0.0,
                 reasoning_effort=None):
    system_prompt, user_template = builder_fn()
    predictions = []
    for _, row in tweets_df.iterrows():
        user_prompt = user_template.format(tweet=row["tweet"])
        try:
            raw = call_openai(client, model, system_prompt, user_prompt,
                               reasoning_effort=reasoning_effort)
        except Exception as e:
            raw = f"__ERROR__: {e}"
        predictions.append(normalize_model_output(raw))
        if sleep_s:
            time.sleep(sleep_s)

    out = tweets_df.copy()
    out[f"pred_{variant_name}"] = predictions
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--limit", type=int, default=30,
                         help="Number of tweets to test on (start small, then scale up)")
    parser.add_argument("--variants", default=None,
                         help="Comma-separated subset of variant names, e.g. A_bare_labels,C_long_definitions")
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    parser.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"],
                         help="Only relevant for reasoning models (e.g. gpt-5.4-mini, o-series). "
                              "Lower effort = faster/cheaper, less internal reasoning.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY as an environment variable first.")

    client = OpenAI()

    df = pd.read_csv(GROUND_TRUTH_CSV)
    if args.limit < len(df):
        # Random (but reproducible) sample instead of the first N rows --
        # the first rows of this CSV are disproportionately edge-case/tied
        # tweets, so df.head(n) is not representative of the full dataset
        # for small test runs.
        sample = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
    else:
        sample = df.copy()

    variants_to_run = PROMPT_VARIANTS
    if args.variants:
        wanted = set(args.variants.split(","))
        variants_to_run = {k: v for k, v in PROMPT_VARIANTS.items() if k in wanted}

    if args.use_mlflow:
        import mlflow
        mlflow.set_experiment("veteran-crisis-labeling")

    all_results = sample.copy()
    summary_rows = []

    for variant_name, builder_fn in variants_to_run.items():
        print(f"\nRunning variant: {variant_name} on {len(sample)} tweets with model {args.model}...")
        result_df = run_variant(client, args.model, variant_name, builder_fn, sample, sleep_s=args.sleep,
                                 reasoning_effort=args.reasoning_effort)
        pred_col = f"pred_{variant_name}"
        all_results[pred_col] = result_df[pred_col]

        metrics = score_predictions(result_df, pred_col=pred_col)
        print_report(variant_name, args.model, metrics)

        summary_rows.append({
            "variant": variant_name,
            "model": args.model,
            "n_scored": metrics["n_scored"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "n_unparseable": metrics["n_unparseable"],
        })

        if args.use_mlflow:
            with mlflow.start_run(run_name=f"{variant_name}_{args.model}"):
                mlflow.log_param("variant", variant_name)
                mlflow.log_param("model", args.model)
                mlflow.log_param("backend", "openai")
                mlflow.log_param("n_tweets", len(sample))
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("macro_f1", metrics["macro_f1"])
                mlflow.log_metric("n_unparseable", metrics["n_unparseable"])
                # Save the exact system prompt used, so this run is reproducible
                system_prompt, _ = builder_fn()
                with open("_tmp_prompt.txt", "w") as f:
                    f.write(system_prompt)
                mlflow.log_artifact("_tmp_prompt.txt", artifact_path="prompts")

    summary_df = pd.DataFrame(summary_rows)
    print("\n\n=== ABLATION SUMMARY ===")
    print(summary_df.to_string(index=False))

    all_results.to_csv("openai_ablation_results.csv", index=False)
    summary_df.to_csv("openai_ablation_summary.csv", index=False)
    print("\nSaved: openai_ablation_results.csv, openai_ablation_summary.csv")


if __name__ == "__main__":
    main()
