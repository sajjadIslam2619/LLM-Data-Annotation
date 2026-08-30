"""
Run one prompt variant across every model in the comparison sweep --
OpenAI models sequentially (to stay under rate limits) followed by
Ollama models sequentially (to avoid fighting over local CPU/GPU/RAM) --
logging every run to MLflow and printing one consolidated results table
at the end.

This is the "run everything with one command" version of what
run_openai_ablation.py / run_ollama_compare.py do individually. If a
specific model errors out (e.g. deprecated, not pulled, rate limited),
it's recorded as failed and the sweep continues with the rest rather
than stopping.

Usage:
  export OPENAI_API_KEY="sk-..."
  ollama pull llama3.2   (and any other Ollama models you want, ahead of time)
  python run_full_sweep.py --variant H_reviewer_matched --limit 149 --use-mlflow

  # Only a subset:
  python run_full_sweep.py --variant H_reviewer_matched --openai-models gpt-4o-mini,gpt-5.6-sol --ollama-models llama3.2

  # Add OpenRouter's free models too (verify working IDs first with
  # check_openrouter_free_models.py -- the free lineup rotates):
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python run_full_sweep.py --variant H_reviewer_matched --openai-models none --ollama-models none \\
      --openrouter-models tencent/hy3:free,openai/gpt-oss-20b:free,nvidia/nemotron-3-ultra-550b-a55b:free

  # Add Claude models (pip install anthropic first):
  export ANTHROPIC_API_KEY="sk-ant-..."
  python run_full_sweep.py --variant H_reviewer_matched --openai-models none --ollama-models none \\
      --anthropic-models claude-opus-5,claude-sonnet-5,claude-haiku-4-5-20251001 --use-mlflow
"""

import argparse
import os
import time

import pandas as pd
import requests
from openai import OpenAI

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output
from score_utils import score_predictions

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"

# Anthropic client is created lazily inside run_one_model (same pattern as
# the OpenAI/OpenRouter clients below) so this script still runs fine
# without the anthropic package installed if you're not using this backend.
DEFAULT_ANTHROPIC_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
]

DEFAULT_OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]
# gpt-oss:20b removed from the default list on purpose -- it overloaded this
# machine badly enough to crash it (see RESEARCH_LOG.md Phase 7). Never add
# it back to this default; if it's ever needed, pass it explicitly via
# --ollama-models with full awareness of the RAM risk.
DEFAULT_OLLAMA_MODELS = ["llama3.2"]

# Confirmed working via check_openrouter_free_models.py (2026-07-15). Excludes
# openai/gpt-oss-120b:free (404 -- paid only right now) and openrouter/free
# (a router, not a specific model -- can silently switch backends mid-sweep,
# which breaks per-model attribution for the paper).
DEFAULT_OPENROUTER_MODELS = [
    "tencent/hy3:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-m.1:free",
    "openai/gpt-oss-20b:free",
]

OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def call_openai(client, model, system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def call_openai_safe(client, model, system_prompt, user_prompt):
    """Same as call_openai, but retries once without temperature if the
    model rejects it (reasoning-tier models like gpt-5.x)."""
    try:
        return call_openai(client, model, system_prompt, user_prompt)
    except Exception as e:
        if "temperature" in str(e).lower():
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        raise


def call_openrouter(client, model, system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


NO_CUSTOM_TEMPERATURE = {"claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5-20251001"}
# claude-haiku-4-5 raises "unexpected keyword argument 'temperature'" (a
# client-side TypeError, not a server 400) on EVERY attempt to pass
# thinking={"type":"disabled"} together with temperature, including the
# no-thinking retry -- so for this model, skip the thinking attempt
# entirely rather than trying and catching. Matches HANDOFF's note that
# Haiku doesn't support the thinking param at all.
NO_THINKING_ATTEMPT = {"claude-haiku-4-5-20251001"}


def call_anthropic(client, model, system_prompt, user_prompt):
    # See run_anthropic_ablation.py's call_anthropic() for the full writeup.
    # Short version, two stacked bugs, both fixed here (Aug 2026):
    #   1. output_config effort "low" never reliably capped thinking tokens
    #      (Anthropic's own docs: "effort is a behavioral signal, not a
    #      strict token budget"). Real fix: thinking={"type": "disabled"}.
    #   2. Sonnet 5 / Opus 5 / Fable 5 400 on ANY non-default temperature,
    #      thinking or not -- the old unconditional temperature=0 was
    #      erroring every call, silently normalized to UNPARSEABLE.
    kwargs = {
        "model": model,
        "max_tokens": 2048,
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
    return "".join(block.text for block in resp.content if block.type == "text")


def call_ollama(model, system_prompt, user_prompt):
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


def run_one_model(backend, model, system_prompt, user_template, sample_df, sleep_s=0.0):
    predictions = []
    run_one_model._printed_error = False
    if backend == "openai":
        client = OpenAI()
    elif backend == "openrouter":
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    elif backend == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
    else:
        client = None
    for _, row in sample_df.iterrows():
        user_prompt = user_template.format(tweet=row["tweet"])
        try:
            if backend == "openai":
                raw = call_openai_safe(client, model, system_prompt, user_prompt)
            elif backend == "openrouter":
                raw = call_openrouter(client, model, system_prompt, user_prompt)
            elif backend == "anthropic":
                raw = call_anthropic(client, model, system_prompt, user_prompt)
            else:
                raw = call_ollama(model, system_prompt, user_prompt)
        except Exception as e:
            raw = f"__ERROR__: {e}"
            if not getattr(run_one_model, "_printed_error", False):
                print(f"    [DEBUG] first error for {model}: {type(e).__name__}: {e}")
                run_one_model._printed_error = True
        predictions.append(normalize_model_output(raw))
        if sleep_s:
            time.sleep(sleep_s)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="H_reviewer_matched",
                         help="Key from PROMPT_VARIANTS to run across all models")
    parser.add_argument("--limit", type=int, default=149)
    parser.add_argument("--openai-models", default=",".join(DEFAULT_OPENAI_MODELS),
                         help="Comma-separated OpenAI model list (or 'none' to skip)")
    parser.add_argument("--ollama-models", default=",".join(DEFAULT_OLLAMA_MODELS),
                         help="Comma-separated Ollama model list (or 'none' to skip)")
    parser.add_argument("--openrouter-models", default="none",
                         help="Comma-separated OpenRouter model list (or 'none' to skip). "
                              "Requires OPENROUTER_API_KEY set. Defaults to 'none' since it's "
                              "opt-in -- pass explicit model IDs, e.g. "
                              f"{','.join(DEFAULT_OPENROUTER_MODELS)}")
    parser.add_argument("--anthropic-models", default="none",
                         help="Comma-separated Anthropic model list (or 'none' to skip). "
                              "Requires ANTHROPIC_API_KEY and `pip install anthropic`. "
                              "Defaults to 'none' since it's opt-in -- pass explicit model IDs, e.g. "
                              f"{','.join(DEFAULT_ANTHROPIC_MODELS)}")
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if args.variant not in PROMPT_VARIANTS:
        raise SystemExit(f"Unknown variant '{args.variant}'. Options: {list(PROMPT_VARIANTS.keys())}")

    builder_fn = PROMPT_VARIANTS[args.variant]
    system_prompt, user_template = builder_fn()

    df = pd.read_csv(GROUND_TRUTH_CSV)
    sample = df.head(args.limit).copy()

    openai_models = [] if args.openai_models.lower() == "none" else args.openai_models.split(",")
    ollama_models = [] if args.ollama_models.lower() == "none" else args.ollama_models.split(",")
    openrouter_models = [] if args.openrouter_models.lower() == "none" else args.openrouter_models.split(",")
    anthropic_models = [] if args.anthropic_models.lower() == "none" else args.anthropic_models.split(",")

    if openrouter_models and "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("--openrouter-models was set but OPENROUTER_API_KEY is not exported.")
    if anthropic_models and "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("--anthropic-models was set but ANTHROPIC_API_KEY is not exported.")

    if args.use_mlflow:
        import mlflow
        mlflow.set_experiment("veteran-crisis-labeling")

    results = []  # (backend, model, status, metrics_dict or error string)

    def run_and_score(backend, model):
        print(f"\n{'='*60}\nRunning {args.variant} on {model} ({backend}), {len(sample)} tweets...")
        try:
            preds = run_one_model(backend, model, system_prompt, user_template, sample, sleep_s=args.sleep)
        except Exception as e:
            print(f"  FAILED to run {model}: {e}")
            results.append((backend, model, "failed", str(e)))
            return

        scored = sample.copy()
        scored[f"pred_{args.variant}"] = preds
        metrics = score_predictions(scored, pred_col=f"pred_{args.variant}")
        print(f"  accuracy={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}  "
              f"unparseable={metrics['n_unparseable']}")
        results.append((backend, model, "ok", metrics))

        # Save per-tweet predictions, not just the summary metrics. Without
        # this, per-class precision/recall can never be recomputed later --
        # this is exactly the gap that lost the original local Variant F
        # sweep (RESEARCH_LOG.md Phase 18.6). Filename matches the existing
        # "{variant}_{model}_results.csv" convention (see D/I/J/K files)
        # exactly, NOT "{variant}_{backend}_{model}" -- compute_precision_recall.py's
        # parse_filename() strips only the variant prefix to recover the
        # model name, so a backend segment in the middle would get parsed
        # as part of the model name (e.g. "openai_gpt-4o-mini" instead of
        # "gpt-4o-mini"). Model names are already distinct across backends
        # (gpt-4o-mini vs. claude-sonnet-5 vs. llama3.1-8b), so no collision.
        model_slug = model.strip().replace("/", "-")
        results_filename = f"{args.variant}_{model_slug}_results.csv"
        scored.to_csv(results_filename, index=False)

        if args.use_mlflow:
            with mlflow.start_run(run_name=f"{args.variant}_{model}"):
                mlflow.log_param("variant", args.variant)
                mlflow.log_param("model", model)
                mlflow.log_param("backend", backend)
                mlflow.log_param("n_tweets", len(sample))
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("macro_f1", metrics["macro_f1"])
                mlflow.log_metric("n_unparseable", metrics["n_unparseable"])
                with open("_tmp_prompt.txt", "w") as f:
                    f.write(system_prompt)
                mlflow.log_artifact("_tmp_prompt.txt", artifact_path="prompts")

    for model in openai_models:
        run_and_score("openai", model.strip())

    for model in openrouter_models:
        run_and_score("openrouter", model.strip())

    for model in anthropic_models:
        run_and_score("anthropic", model.strip())

    for model in ollama_models:
        run_and_score("ollama", model.strip())

    # --- Final consolidated table ---
    print(f"\n\n{'='*60}\nFULL SWEEP RESULTS -- variant: {args.variant}\n{'='*60}")
    rows = []
    for backend, model, status, data in results:
        if status == "ok":
            rows.append({
                "backend": backend, "model": model, "status": status,
                "accuracy": round(data["accuracy"], 3),
                "macro_f1": round(data["macro_f1"], 3),
                "unparseable": data["n_unparseable"],
            })
        else:
            rows.append({"backend": backend, "model": model, "status": f"FAILED: {data[:60]}",
                          "accuracy": None, "macro_f1": None, "unparseable": None})

    summary_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, na_position="last")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"full_sweep_{args.variant}_summary.csv", index=False)
    print(f"\nSaved: full_sweep_{args.variant}_summary.csv")


if __name__ == "__main__":
    main()
