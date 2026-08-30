"""
Run the ablation study against Anthropic (Claude) models.

Mirrors run_openai_ablation.py exactly -- same CSV in, same prompts.py /
labels.py / score_utils.py, same summary/results CSV shape -- so the output
slots directly into macro_metrics.csv / per_class_metrics.csv / the
Model_Prompt_Performance.xlsx workbook alongside the OpenAI and Ollama runs.

What this does:
  1. Loads the ground-truth CSV (built by build_ground_truth.py).
  2. For each prompt variant in prompts.PROMPT_VARIANTS, sends every
     (sampled) tweet to the Claude API and records the predicted label.
  3. Scores each variant against the human majority-vote label.
  4. Prints a comparison table and saves detailed results to CSV.
  5. Optionally logs each variant as an MLflow run (--use-mlflow).

Setup:
  pip install anthropic pandas mlflow   # mlflow only needed if using --use-mlflow
  export ANTHROPIC_API_KEY="sk-ant-..."

Example:
  python run_anthropic_ablation.py --model claude-sonnet-5 --limit 30
  python run_anthropic_ablation.py --model claude-sonnet-5 --limit 149 --use-mlflow
  python run_anthropic_ablation.py --model claude-haiku-4-5-20251001 --variants A_bare_labels,D_few_shot
"""

import argparse
import os
import time

import pandas as pd
from anthropic import Anthropic

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output
from score_utils import score_predictions, print_report

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"
MAX_TOKENS = 2048  # generous headroom for a single-label answer now that
                    # thinking is force-disabled below (see call_anthropic())

# Models where the API rejects any non-default temperature/top_p/top_k on
# EVERY request, thinking or not (confirmed against Anthropic's current
# docs, Aug 2026: platform.claude.com/docs/en/build-with-claude/thinking,
# "Limits and feature compatibility" -> "Sampling parameters"). Passing
# temperature=0 to these models 400s on every single call -- this was
# silently turning into UNPARSEABLE via the except-and-normalize path
# below, on top of the separate thinking-budget bug. No workaround exists;
# these models simply cannot be pinned to temperature=0 right now.
NO_CUSTOM_TEMPERATURE = {"claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5-20251001"}
# claude-haiku-4-5 raises "unexpected keyword argument 'temperature'" (a
# client-side TypeError, not a server 400) when thinking={"type":"disabled"}
# is combined with temperature, and the same error persists on the
# no-thinking retry below -- so skip the thinking attempt for it entirely
# rather than try/catch. Correction to the docstring note below: Haiku
# turned out NOT to be unaffected by this after all, just differently
# affected (Aug 2026 debugging, run_full_sweep.py has the identical fix).
NO_THINKING_ATTEMPT = {"claude-haiku-4-5-20251001"}


def call_anthropic(client: Anthropic, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call the model. Claude models take `system` as a top-level param, not
    a message in the messages list (unlike the OpenAI chat format prompts.py
    was written for).

    Two separate bugs were stacked here before, both confirmed against
    Anthropic's current docs (Aug 2026), not just the first one found
    2026-08-14:

    1. Opus 5 / Sonnet 5 have adaptive thinking ON by default, and thinking
       tokens count toward max_tokens. The old fix (output_config effort
       "low") does NOT reliably prevent this -- Anthropic's own docs call
       effort "a behavioral signal, not a strict token budget," so it can
       still think enough to exhaust a small max_tokens on a given call.
       The real fix is `thinking={"type": "disabled"}`, a hard off-switch,
       supported on Sonnet 5 and Opus 5 at effort high (the default) or
       below. This task is single-label classification -- there's nothing
       for thinking to buy here, so fully disabling it is strictly correct,
       not just a cost optimization.
    2. Separately, Sonnet 5 / Opus 5 / Fable 5 return a 400 error on ANY
       request that sets a non-default temperature, regardless of thinking
       state. The old code passed temperature=0 unconditionally, so every
       call to these three models was erroring out before it ever reached
       the model -- caught by run_variant()'s except block and normalized
       to UNPARSEABLE, indistinguishable from a real empty response. This
       was likely the dominant cause of the 100% failures, not just the
       thinking budget. See NO_CUSTOM_TEMPERATURE above.

    Haiku 4.5 turned out to have its own, separate issue (see
    NO_THINKING_ATTEMPT above): passing temperature alongside
    thinking={"type":"disabled"} raises a client-side TypeError on this
    model specifically, so it's now also in NO_CUSTOM_TEMPERATURE and
    skips the thinking attempt entirely rather than try/catching it.
    """
    kwargs = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if model not in NO_CUSTOM_TEMPERATURE:
        kwargs["temperature"] = 0

    if model in NO_THINKING_ATTEMPT:
        resp = client.messages.create(**kwargs)
    else:
        try:
            resp = client.messages.create(thinking={"type": "disabled"}, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "thinking" in msg or "unexpected keyword argument" in msg:
                resp = client.messages.create(**kwargs)
            else:
                raise
    # Concatenate all text blocks in case of multi-block responses.
    return "".join(block.text for block in resp.content if block.type == "text")


def run_variant(client, model, variant_name, builder_fn, tweets_df, sleep_s=0.0):
    system_prompt, user_template = builder_fn()
    predictions = []
    for _, row in tweets_df.iterrows():
        user_prompt = user_template.format(tweet=row["tweet"])
        try:
            raw = call_anthropic(client, model, system_prompt, user_prompt)
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
    parser.add_argument("--model", default="claude-sonnet-5", help="Anthropic model name")
    parser.add_argument("--limit", type=int, default=30,
                         help="Number of tweets to test on (start small, then scale up)")
    parser.add_argument("--variants", default=None,
                         help="Comma-separated subset of variant names, e.g. A_bare_labels,C_long_definitions")
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY as an environment variable first.")

    client = Anthropic()

    df = pd.read_csv(GROUND_TRUTH_CSV)
    if args.limit < len(df):
        # Same reproducible sample seed as run_openai_ablation.py -- keeps
        # small test runs comparable across backends.
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
        result_df = run_variant(client, args.model, variant_name, builder_fn, sample, sleep_s=args.sleep)
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
                mlflow.log_param("backend", "anthropic")
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

    # Per-model filename (like run_new_experiments.py), not a shared
    # openai_ablation_results.csv-style name -- RESEARCH_LOG.md Phase 18.6
    # documents two summary CSVs that got silently overwritten this way.
    model_slug = args.model.replace("/", "-")
    all_results.to_csv(f"anthropic_ablation_{model_slug}_results.csv", index=False)
    summary_df.to_csv(f"anthropic_ablation_{model_slug}_summary.csv", index=False)
    print(f"\nSaved: anthropic_ablation_{model_slug}_results.csv, anthropic_ablation_{model_slug}_summary.csv")


if __name__ == "__main__":
    main()
