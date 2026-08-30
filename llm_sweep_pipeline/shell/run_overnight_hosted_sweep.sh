#!/bin/bash
# Overnight batch: runs D_few_shot and F_refined_definitions (the two
# strongest variants on the record -- best-per-model peaks of 0.689 and
# 0.715 macro F1 respectively across the 17-model local sweep, see
# RESEARCH_LOG.md Phase 13/18.6) plus H_reviewer_matched (the low-info
# capability test), across BOTH OpenAI and Anthropic hosted models in one
# run. No Ollama models here -- this project already has extensive local
# coverage; this run is specifically to get hosted-model numbers for these
# 3 variants while there's API credit available on both accounts.
#
# Model lists below deliberately exclude the priciest tier on each side,
# per instruction -- same reasoning as excluding gpt-4 from the sweep:
#   - OpenAI: skips gpt-4 and gpt-4-turbo (legacy, ~6-30x priciest current
#     model on this task; gpt-4 alone would cost more than every other
#     OpenAI model on this list combined)
#   - Anthropic: skips claude-fable-5 (priced like a premium creative-writing
#     tier, 2x Opus 5, and not the right tool for a classification task anyway)
#
# Estimated cost for this whole run (8 OpenAI models + 3 Anthropic models,
# 3 variants x 149 tweets = 447 calls/model, no prompt caching): roughly
# $5-6 on the OpenAI side and $3-4 on the Anthropic side -- comfortably
# under $20 on each account, even doubled for a safety margin. The most
# expensive single model in the whole run is gpt-5.6-sol at under $2.
#
# Usage (so your Mac doesn't fall asleep partway through, and survives
# closing the terminal window/tab):
#
#   export OPENAI_API_KEY="sk-..."
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
#   nohup caffeinate -i bash run_overnight_hosted_sweep.sh > overnight_hosted.log 2>&1 &
#   disown
#
# NOTE: caffeinate -i keeps the machine awake while the lid is OPEN and
# plugged in -- it does NOT survive an actually closed lid (see HANDOFF.md).
# Leave the lid open and plugged in overnight, or this will stall partway
# through like the Ollama runs did before.
#
# Check progress any time with:  tail -f overnight_hosted.log
# When it's done, re-run compute_precision_recall.py so the new results
# flow into per_class_metrics.csv / macro_metrics.csv -- Model_Prompt_Performance.xlsx
# picks them up automatically from there, no manual editing needed.

set -uo pipefail   # NOT -e: one model failing shouldn't kill the whole overnight run

VARIANTS=("D_few_shot" "F_refined_definitions" "H_reviewer_matched")

# Cheapest to priciest is fine -- order doesn't matter for cost, but running
# cheap models first means you see *something* complete quickly if you check
# in partway through the night.
OPENAI_MODELS="gpt-4o-mini,gpt-5.6-luna,gpt-3.5-turbo,gpt-5.4-mini,gpt-5.6-terra,gpt-4o,gpt-5.4,gpt-5.6-sol"
ANTHROPIC_MODELS="claude-haiku-4-5-20251001,claude-sonnet-5,claude-opus-5"

LOGFILE="overnight_hosted_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== Overnight hosted sweep started: $(date) ==="

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not exported. Export it and re-run."
  exit 1
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is not exported. Export it and re-run."
  exit 1
fi

for VARIANT in "${VARIANTS[@]}"; do
  echo ""
  echo "=========================================================="
  echo "=== Variant: $VARIANT -- started $(date) ==="
  echo "=========================================================="
  python run_full_sweep.py \
    --variant "$VARIANT" \
    --limit 149 \
    --openai-models "$OPENAI_MODELS" \
    --anthropic-models "$ANTHROPIC_MODELS" \
    --ollama-models none \
    --use-mlflow \
    --sleep 0.5
  echo "=== Variant: $VARIANT -- finished $(date) ==="
done

echo ""
echo "=== Overnight hosted sweep finished: $(date) ==="
echo "Results: full_sweep_<variant>_summary.csv for each of the 3 variants above."
echo "Next step in the morning:"
echo "  python compute_precision_recall.py"
echo "This regenerates per_class_metrics.csv / macro_metrics.csv with the new hosted"
echo "rows, which Model_Prompt_Performance.xlsx reads from automatically."
