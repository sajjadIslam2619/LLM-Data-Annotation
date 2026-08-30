#!/bin/bash
# Runs the last 12 Ollama models for prompt variant G_official_reasoning_format
# on the veteran-crisis-tweet dataset. Nothing else is needed -- OpenAI/Anthropic
# are already done on this variant, so this only touches your local Ollama install.
set -uo pipefail
cd "$(dirname "$0")"

MODELS="gemma3:4b,granite3.3:8b,hermes3:8b,internlm2:7b,llama3:8b,llama3.1:8b,mistral:7b,olmo2:7b,phi4-mini:3.8b,qwen2.5:7b,qwen3:8b,yi:6b"

echo "Pulling models (skips any already installed)..."
for m in $(echo "$MODELS" | tr ',' '\n'); do
  ollama pull "$m"
done

echo ""
echo "Running the sweep. This can take several hours depending on your machine --"
echo "each model runs on 149 tweets one at a time, sequentially."
python3 -u run_full_sweep.py \
  --variant G_official_reasoning_format \
  --openai-models none \
  --anthropic-models none \
  --ollama-models "$MODELS" \
  2>&1 | tee -a G_remaining_run.log

echo ""
echo "Done (or check G_remaining_run.log if it stopped early)."
echo "Send back every file matching G_official_reasoning_format_*_results.csv"
echo "in this folder -- that's the actual output that needs to come back."
