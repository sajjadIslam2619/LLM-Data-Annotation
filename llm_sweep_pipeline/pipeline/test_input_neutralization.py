"""
Input-neutralization robustness test (adapted from arXiv:2307.05052, Liu/Xu/Li
et al. -- "Towards Understanding In-Context Learning with Contrastive
Demonstrations and Saliency Maps", flagged by Sajjad).

What the paper actually does: neutralize sentiment-indicative words INSIDE the
few-shot demonstrations and see how much that (vs. label flipping) affects
saliency/accuracy. That doesn't map cleanly onto this project's few-shot set --
only the Label-3 demo in Variant D has any indicative language to strip, so a
literal replication would only touch 1 of 3 examples and not tell us much.

What this script does instead: operationalizes the same underlying question
-- "does the model need explicit indicative language, or does it understand
the underlying situation?" -- against this project's own most concerning
documented failure mode (RESEARCH_LOG.md, Phase 6): "5 -> 4 undershooting on
implicit language... the model appears to require more explicit language than
what's actually present."

Method: hand-built pairs of tweets, one severity level apart in explicitness
only (same underlying situation and same true label), one version with overt
crisis/distress keywords, one with the keywords stripped/implied instead. Both
versions of a pair get the SAME ground-truth label -- if the model's
prediction differs between the two, that's evidence it's leaning on keyword
presence rather than situational understanding, and (for the label-5 pairs
especially) a false negative risk.

These are synthetic, hand-written pairs -- NOT pulled from the evaluation CSV
(same no-leakage precedent as Variant F's synthetic examples) -- so there is
no overlap with anything used for scoring elsewhere.

Usage (Ollama, local, default -- make sure `ollama serve` is running):
  python test_input_neutralization.py --model qwen3:8b --variant D_few_shot
  python test_input_neutralization.py --model llama3.1:8b --variant H_reviewer_matched

Usage (OpenAI, if/when you have API access again):
  export OPENAI_API_KEY="sk-..."
  python test_input_neutralization.py --backend openai --model gpt-4o-mini --variant D_few_shot
"""

import argparse
import os

import pandas as pd

from prompts import PROMPT_VARIANTS
from labels import normalize_model_output, label_key

# Each pair: (severity_name, true_label, explicit_version, implicit_version)
# True label is identical across both versions of a pair -- only the
# explicitness of the language differs, not the underlying situation.
NEUTRALIZATION_PAIRS = [
    (
        "acute_crisis_1",
        "5- Acute Crisis with Suicidal Ideation",
        "I want to kill myself tonight, I've got the pills right here.",
        "Everything's fading now. I don't think I'll see tomorrow.",
    ),
    (
        "acute_crisis_2",
        "5- Acute Crisis with Suicidal Ideation",
        "I'm going to end it. This is my last night here.",
        "This is it for me, guys. Just wanted to say goodbye.",
    ),
    (
        "crisis_seeking_support",
        "3-Self-Identified Crisis Symptoms and Seeking Support",
        "I'm in a mental health crisis and I really need someone to talk to right now, please.",
        "Is anyone even out there tonight? Kind of struggling over here.",
    ),
    (
        "crisis_not_seeking_support",
        "4- Self-Identified Crisis Symptoms but Not Seeking Support",
        "I'm dealing with some dark thoughts but I don't want to talk about it or get help.",
        "Yeah I'm not okay. Not looking for advice though.",
    ),
    (
        "distressed_not_crisis",
        "2- Distressed Self-Disclosure",
        "I've been so anxious and stressed out all week, it's honestly been rough.",
        "This week has just been... a lot, honestly.",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                         help="ollama (default, local/free) or openai (needs OPENAI_API_KEY)")
    parser.add_argument("--model", default="qwen3:8b",
                         help="Model tag. Ollama e.g. qwen3:8b, llama3.1:8b. OpenAI e.g. gpt-4o-mini.")
    parser.add_argument("--variant", default="D_few_shot",
                         choices=list(PROMPT_VARIANTS.keys()),
                         help="Which prompt variant to test robustness under")
    parser.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"],
                         help="OpenAI reasoning models only")
    args = parser.parse_args()

    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY as an environment variable first.")
        from openai import OpenAI
        from run_openai_ablation import call_openai
        client = OpenAI()

        def call_model(system_prompt, user_prompt):
            return call_openai(client, args.model, system_prompt, user_prompt,
                                reasoning_effort=args.reasoning_effort)
    else:
        from run_ollama_compare import call_ollama

        def call_model(system_prompt, user_prompt):
            return call_ollama(args.model, system_prompt, user_prompt)

    system_prompt, user_template = PROMPT_VARIANTS[args.variant]()

    rows = []
    for name, true_label, explicit_tweet, implicit_tweet in NEUTRALIZATION_PAIRS:
        for version, tweet in (("explicit", explicit_tweet), ("implicit", implicit_tweet)):
            try:
                raw = call_model(system_prompt, user_template.format(tweet=tweet))
            except Exception as e:
                raw = f"__ERROR__: {e}"
            pred = normalize_model_output(raw)
            rows.append({
                "pair": name, "version": version, "tweet": tweet,
                "true_label": true_label, "predicted_label": pred,
                "correct": label_key(pred) == label_key(true_label),
            })

    df = pd.DataFrame(rows)

    print(f"\n=== Input neutralization test | variant={args.variant} | model={args.model} ===\n")
    print(df[["pair", "version", "true_label", "predicted_label", "correct"]].to_string(index=False))

    print("\n--- Flips (explicit vs. implicit prediction differs) ---")

    def _as_int(key):
        return int(key) if key.isdigit() else None

    flipped_any = False
    n_flips = 0
    n_downgrades = 0
    for name, group in df.groupby("pair"):
        explicit_pred = group[group.version == "explicit"]["predicted_label"].iloc[0]
        implicit_pred = group[group.version == "implicit"]["predicted_label"].iloc[0]
        ek, ik = label_key(explicit_pred), label_key(implicit_pred)
        if ek != ik:
            flipped_any = True
            n_flips += 1
            ei, ii, ti = _as_int(ek), _as_int(ik), _as_int(label_key(group["true_label"].iloc[0]))
            is_downgrade = ei is not None and ii is not None and ti is not None and ii < ti <= ei
            if is_downgrade:
                n_downgrades += 1
            direction = "DOWNGRADE (concerning)" if is_downgrade else "changed"
            print(f"  {name}: explicit -> {explicit_pred} | implicit -> {implicit_pred}  [{direction}]")
    if not flipped_any:
        print("  None -- predictions were consistent regardless of explicit vs. implicit wording.")

    model_tag = args.model.replace(":", "-").replace("/", "-")
    out_path = f"neutralization_test_{args.variant}_{model_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"flip_rate={n_flips}/{len(NEUTRALIZATION_PAIRS)} downgrades={n_downgrades}")


if __name__ == "__main__":
    main()
