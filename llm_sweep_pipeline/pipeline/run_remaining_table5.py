#!/usr/bin/env python3
"""
Resumable runner for the E_chain_of_thought / G_official_reasoning_format
sweep. Safe to re-run any time it dies, gets killed, or the machine sleeps:
it checks which {variant}_{model}_results.csv files already exist and only
asks run_full_sweep.py to do the ones that are missing, so nothing already
finished gets redone.

Usage:  python3 -u run_remaining_table5.py
Add --dry-run to just print what would run without actually calling anything.
"""
import subprocess
import sys
import os

DRY_RUN = "--dry-run" in sys.argv

OPENAI = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5.4", "gpt-5.4-mini",
          "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"]
ANTHROPIC = ["claude-opus-5", "claude-sonnet-5"]
# deepseek-r1:8b deliberately excluded (slow + caused the original crash)
OLLAMA = ["aya-expanse:8b", "command-r7b:7b", "gemma3:4b", "granite3.3:8b",
          "hermes3:8b", "internlm2:7b", "llama3:8b", "llama3.1:8b",
          "mistral:7b", "olmo2:7b", "phi4-mini:3.8b", "qwen2.5:7b",
          "qwen3:8b", "yi:6b"]

VARIANTS = ["E_chain_of_thought", "G_official_reasoning_format"]


def missing_models(variant, models):
    out = []
    for m in models:
        slug = m.strip().replace("/", "-")
        fname = f"{variant}_{slug}_results.csv"
        if not os.path.exists(fname):
            out.append(m)
    return out


def run_variant(variant):
    remaining = {
        "--openai-models": missing_models(variant, OPENAI),
        "--anthropic-models": missing_models(variant, ANTHROPIC),
        "--ollama-models": missing_models(variant, OLLAMA),
    }

    if not any(remaining.values()):
        print(f"\n=== {variant}: already complete, nothing to do ===", flush=True)
        return

    cmd = ["python3", "-u", "run_full_sweep.py", "--variant", variant, "--use-mlflow"]
    for flag, models in remaining.items():
        cmd += [flag, ",".join(models) if models else "none"]

    print(f"\n=== {variant}: {sum(len(v) for v in remaining.values())} models remaining ===", flush=True)
    for flag, models in remaining.items():
        if models:
            print(f"    {flag}: {','.join(models)}", flush=True)
    print(f"    command: {' '.join(cmd)}", flush=True)

    if DRY_RUN:
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n!!! {variant} exited with code {result.returncode} -- "
              f"just re-run this script, it will pick up whatever's still missing.", flush=True)


def main():
    for variant in VARIANTS:
        try:
            run_variant(variant)
        except Exception as e:
            print(f"\n!!! {variant} crashed with {type(e).__name__}: {e} -- "
                  f"just re-run this script, it will pick up whatever's still missing.", flush=True)

    print("\n=== all variants processed. Re-run this script again if anything above shows non-zero exit codes or crashes. ===", flush=True)


if __name__ == "__main__":
    main()
