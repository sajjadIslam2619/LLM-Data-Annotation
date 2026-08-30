"""
Prints the FULL raw response from an Ollama model for a handful of tweets,
so you can see exactly what it's outputting instead of just the pass/fail
unparseable count. Use this when a model's unparseable rate looks
suspiciously high and you want to know why before assuming a cause.

Usage:
  python debug_raw_response.py --model yi:6b --variant F_refined_definitions --limit 8
"""

import argparse
import requests
import pandas as pd

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output

OLLAMA_URL = "http://localhost:11434/api/chat"
GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="F_refined_definitions")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    system_prompt, user_template = PROMPT_VARIANTS[args.variant]()
    df = pd.read_csv(GROUND_TRUTH_CSV).head(args.limit)

    for i, row in df.iterrows():
        user_prompt = user_template.format(tweet=row["tweet"])
        try:
            raw = call_ollama(args.model, system_prompt, user_prompt)
        except Exception as e:
            raw = f"__ERROR__: {e}"

        parsed = normalize_model_output(raw)
        flag = "  <-- UNPARSEABLE" if parsed == "UNPARSEABLE" else ""

        print(f"\n{'='*70}")
        print(f"Tweet #{i}: {row['tweet']!r}")
        print(f"True label: {row.get('ground_truth', row.get('majority_label', '?'))}")
        print(f"Parsed as: {parsed}{flag}")
        print(f"--- RAW RESPONSE ({len(raw)} chars) ---")
        print(raw)


if __name__ == "__main__":
    main()
