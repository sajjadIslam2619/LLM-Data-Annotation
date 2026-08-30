#!/bin/bash
# Completes the ablation study (Variants A-E) across all installed local
# Ollama models. These 5 variants have only ever been tested on
# gpt-4o-mini (Phase 4, single-model ablation) -- H, F, and G already
# have multi-model local data (Phases 8, 10, 11, and the G run).
#
# This is a genuinely long job: 5 variants x 17 models x 149 tweets.
# qwen3:8b and deepseek-r1:8b are "thinking" models and run much slower
# than the rest (~40-60 min each per variant vs. ~10-15 min for the
# others) -- expect this to run for many hours, likely overnight.
#
# Safe to stop anytime with Ctrl+C: each model's MLflow log is written
# as soon as that model finishes (not batched at the end), so you only
# lose the current in-progress model, not everything before it. The
# per-variant summary CSV is written after that variant's full model
# loop completes, so a variant killed mid-way won't have its CSV yet --
# query MLflow directly for partial results if needed.
#
# Usage:
#   caffeinate -i bash run_remaining_ablation.sh

set -e
cd "$(dirname "$0")"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

MODELS="llama3.1:8b,qwen3:8b,mistral:7b,gemma3:4b,phi4-mini:3.8b,deepseek-r1:8b,granite3.3:8b,hermes3:8b,olmo2:7b,falcon3:7b,command-r7b:7b,exaone3.5:7.8b,internlm2:7b,yi:6b,aya-expanse:8b,llama3:8b,qwen2.5:7b"

for variant in A_bare_labels B_short_definitions C_long_definitions D_few_shot E_chain_of_thought; do
  echo ""
  echo "=================================================="
  echo "STARTING VARIANT: $variant"
  echo "=================================================="
  python run_full_sweep.py --variant "$variant" --limit 149 --use-mlflow \
      --openai-models none --ollama-models "$MODELS"
done

echo ""
echo "ALL REMAINING VARIANTS COMPLETE."
