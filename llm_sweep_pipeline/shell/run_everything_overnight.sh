#!/bin/bash
# Overnight batch: runs the full D/I/J/K ablation + neutralization test on
# EVERY installed Ollama model (auto-discovered via `ollama list`, excluding
# gpt-oss:20b -- that one crashed this machine before, see RESEARCH_LOG.md
# Phase 7 -- never run it again), then runs the LLM-as-annotator agreement
# check across every prediction file that exists, old and new.
#
# Safe to walk away from: models already marked "done" in
# experiment_tracker.csv are skipped automatically, so if this gets
# interrupted (sleep, crash, whatever), just run it again and it picks up
# where it left off instead of redoing everything.
#
# Usage (do this, not just `bash run_everything_overnight.sh`, so your Mac
# doesn't fall asleep partway through and stall every Ollama call):
#
#   caffeinate -i bash run_everything_overnight.sh
#
# Or, to also survive closing the terminal window/tab:
#
#   nohup caffeinate -i bash run_everything_overnight.sh > overnight.log 2>&1 &
#   disown
#
# Check progress any time with:  tail -f overnight.log
# (or the timestamped log this script writes itself, see below)

set -uo pipefail   # NOT -e: one model failing shouldn't kill the whole overnight run

LOGFILE="overnight_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== Overnight run started: $(date) ==="

echo ""
echo "--- Discovering installed Ollama models ---"
ALL_MODELS=$(ollama list | tail -n +2 | awk '{print $1}' | grep -v '^gpt-oss:20b$')
echo "$ALL_MODELS"
MODEL_LIST=$(echo "$ALL_MODELS" | paste -sd, -)
echo "Model list passed to run_new_experiments.py: $MODEL_LIST"

echo ""
echo "=== Step 1/2: D/I/J/K ablation + input-neutralization test, all models ==="
echo "(already-done model/task combos in experiment_tracker.csv are skipped automatically)"
python run_new_experiments.py --models "$MODEL_LIST" --steps ablation,neutralization

echo ""
echo "=== Step 2/2: LLM-as-annotator agreement check across every results file ==="
echo "(recomputes over every *_results.csv found -- new ones from step 1 plus every"
echo " historical one: openai_ablation_results.csv, ollama_<model>_results.csv, etc.)"
python run_all_agreement_checks.py

echo ""
echo "=== Overnight run finished: $(date) ==="
echo "Check in the morning: EXPERIMENT_TRACKER.md, AGREEMENT_SUMMARY.md, and $LOGFILE"
