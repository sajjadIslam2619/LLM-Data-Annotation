"""
Run your best prompt variant(s) against local Ollama models, so you can
compare a free/local small model against the paid OpenAI models.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model, e.g.:  ollama pull llama3.1
                           ollama pull mistral
                           ollama pull phi3
  3. Make sure the Ollama app/server is running (it runs a local server
     at http://localhost:11434 automatically once installed).
  4. pip install requests pandas

Example:
  python run_ollama_compare.py --model llama3.1 --limit 30
  python run_ollama_compare.py --model phi3 --variants C_long_definitions
"""

import argparse
import time

import pandas as pd
import requests

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output
from score_utils import score_predictions, print_report

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"
OLLAMA_URL = "http://localhost:11434/api/chat"


def call_ollama(model: str, system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def run_variant(model, variant_name, builder_fn, tweets_df, sleep_s=0.0):
    system_prompt, user_template = builder_fn()
    predictions = []
    for _, row in tweets_df.iterrows():
        user_prompt = user_template.format(tweet=row["tweet"])
        try:
            raw = call_ollama(model, system_prompt, user_prompt)
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
    parser.add_argument("--model", default="llama3.1", help="Ollama model tag, e.g. llama3.1, mistral, phi3")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--variants", default=None,
                         help="Comma-separated subset of variant names")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--use-mlflow", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(GROUND_TRUTH_CSV)
    sample = df.head(args.limit).copy()

    variants_to_run = PROMPT_VARIANTS
    if args.variants:
        wanted = set(args.variants.split(","))
        variants_to_run = {k: v for k, v in PROMPT_VARIANTS.items() if k in wanted}

    if args.use_mlflow:
        import mlflow
        # Same experiment name as run_openai_ablation.py so OpenAI and
        # Ollama runs show up side by side in one comparison table.
        mlflow.set_experiment("veteran-crisis-labeling")

    all_results = sample.copy()
    summary_rows = []

    for variant_name, builder_fn in variants_to_run.items():
        print(f"\nRunning variant: {variant_name} on {len(sample)} tweets with Ollama model {args.model}...")
        result_df = run_variant(args.model, variant_name, builder_fn, sample, sleep_s=args.sleep)
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
                mlflow.log_param("backend", "ollama")
                mlflow.log_param("n_tweets", len(sample))
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("macro_f1", metrics["macro_f1"])
                mlflow.log_metric("n_unparseable", metrics["n_unparseable"])
                system_prompt, _ = builder_fn()
                with open("_tmp_prompt.txt", "w") as f:
                    f.write(system_prompt)
                mlflow.log_artifact("_tmp_prompt.txt", artifact_path="prompts")

    summary_df = pd.DataFrame(summary_rows)
    print("\n\n=== OLLAMA SUMMARY ===")
    print(summary_df.to_string(index=False))

    all_results.to_csv(f"ollama_{args.model}_results.csv", index=False)
    summary_df.to_csv(f"ollama_{args.model}_summary.csv", index=False)
    print(f"\nSaved: ollama_{args.model}_results.csv, ollama_{args.model}_summary.csv")


if __name__ == "__main__":
    main()
