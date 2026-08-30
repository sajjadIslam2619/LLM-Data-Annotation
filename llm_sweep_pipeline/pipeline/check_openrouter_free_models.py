"""
Smoke test: check which OpenRouter free-tier model IDs are actually
callable with your API key, before wiring OpenRouter into
run_full_sweep.py as a new backend.

Setup:
  export OPENROUTER_API_KEY="sk-or-v1-..."   (you already did this)
  pip install openai

Usage:
  python check_openrouter_free_models.py
"""

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

CANDIDATE_MODELS = [
    "tencent/hy3:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-m.1:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "openrouter/free",
]

print(f"Testing {len(CANDIDATE_MODELS)} model IDs against OpenRouter...\n")

working = []
for model in CANDIDATE_MODELS:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
            timeout=30,
        )
        content = resp.choices[0].message.content
        print(f"  WORKS  {model:55} -> {content!r}")
        working.append(model)
    except Exception as e:
        print(f"  FAILS  {model:55} -> {e}")

print(f"\n{len(working)}/{len(CANDIDATE_MODELS)} working:")
for m in working:
    print(f"  - {m}")
