"""
Prompt variants for the ablation study.

Each variant adds ONE new ingredient on top of the previous one, so that
when you compare their scores you can see exactly which ingredient moved
accuracy up or down. This is the core of the "ablation study" the task
asks for.

  A - bare label list only
  B - + short one-line definitions (what the human reviewers got)
  C - + full paragraph definitions (from Dissertation Ch. 5)
  D - + a few worked examples ("few-shot")
  E - + explicit step-by-step reasoning instruction ("chain of thought")

Usage: each build_prompt_* function returns (system_prompt, user_prompt_template)
where user_prompt_template has a {tweet} placeholder.
"""

from labels import LABELS, SHORT_DEFINITIONS, LONG_DEFINITIONS, REFINED_DEFINITIONS

LABEL_LIST_TEXT = "\n".join(LABELS)

# A small set of worked examples for the few-shot variant. These are
# deliberately NOT pulled from the evaluation set used in scoring --
# swap in your own 3-5 clear-cut examples so you don't leak eval tweets
# into the prompt. Pick unambiguous ones (high reviewer agreement).
FEW_SHOT_EXAMPLES = [
    ("Happy birthday to the beloved king of grunge #KurtCobain", "1-No Concern"),
    ("I can't make it through this one guys I'm sorry", "3-Self-Identified Crisis Symptoms and Seeking Support"),
    ("Rush Limbaugh has died rot in pieces you racist sack of crap", "6- Not Relevant to the Research"),
]

# Label-flipping variants (arXiv:2307.05052, Liu/Xu/Li et al. -- "Towards
# Understanding In-Context Learning with Contrastive Demonstrations and
# Saliency Maps"). Sajjad flagged this paper: they show that giving few-shot
# demonstrations deliberately WRONG labels significantly degrades ICL
# performance in large LLMs but barely affects small ones -- a way to test
# whether a model is actually using demonstration labels or just leaning on
# pretrained priors. Same tweets as FEW_SHOT_EXAMPLES (Variant D), labels
# altered only.

# I: each example reassigned to a different, arbitrary wrong label (rotated,
# not adjacent to the truth) -- closest match to the paper's own method.
FEW_SHOT_EXAMPLES_FLIPPED_RANDOM = [
    ("Happy birthday to the beloved king of grunge #KurtCobain", "6- Not Relevant to the Research"),
    ("I can't make it through this one guys I'm sorry", "1-No Concern"),
    ("Rush Limbaugh has died rot in pieces you racist sack of crap", "3-Self-Identified Crisis Symptoms and Seeking Support"),
]

# J: only the genuine crisis example is flipped, to "1-No Concern" -- the
# other two are left correct as controls. This isolates the effect of a
# single bad demonstration, and targets the failure mode that actually
# matters for this project: a real crisis tweet mislabeled as no concern.
FEW_SHOT_EXAMPLES_FLIPPED_ADVERSARIAL = [
    ("Happy birthday to the beloved king of grunge #KurtCobain", "1-No Concern"),
    ("I can't make it through this one guys I'm sorry", "1-No Concern"),
    ("Rush Limbaugh has died rot in pieces you racist sack of crap", "6- Not Relevant to the Research"),
]


def _base_instructions() -> str:
    return (
        "You are labeling tweets written by military veterans for a crisis-detection "
        "research study. Read the tweet and assign exactly ONE label from the list below. "
        "Respond with ONLY the label text, nothing else."
    )


# Standardized output-format instruction, applied consistently across variants
# so every response is parsed the same way. NOT added to:
#   - Variant D (Sajjad's official "First version" email -- must stay byte-for-byte
#     unchanged, he explicitly said not to discard/alter it)
#   - Variant E / G (already have their own, richer reasoning+format instructions)
#   - Variant H (intentionally kept minimal/unscaffolded -- see variant_h below)
OUTPUT_FORMAT_ADDENDUM = (
    "\n\nRespond in EXACTLY this format, with no other text:\n"
    "Label: <one of the exact label names from the list above>"
)


def variant_a_bare() -> tuple[str, str]:
    system = (
        f"{_base_instructions()}\n\nLabels:\n{LABEL_LIST_TEXT}" + OUTPUT_FORMAT_ADDENDUM
    )
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def _short_defs_text(include_other_reason_note: bool) -> str:
    """Build the short-definitions listing. The original SHORT_DEFINITIONS['Other']
    says '...include a reason if picking this' -- that only makes sense when
    the response format has a reasoning field to put the reason in. For
    label-only formats (no reasoning field), that clause is a contradiction
    (tells the model to explain itself, then the format instruction says
    respond with ONLY the label) -- so it's stripped in that case."""
    lines = []
    for label in LABELS:
        text = SHORT_DEFINITIONS[label]
        if label == "Other" and not include_other_reason_note:
            text = "Doesn't fit any category above."
        lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def variant_b_short_defs() -> tuple[str, str]:
    defs = _short_defs_text(include_other_reason_note=False)
    system = f"{_base_instructions()}\n\nLabels and definitions:\n{defs}" + OUTPUT_FORMAT_ADDENDUM
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def _long_defs_core() -> str:
    """Shared by variant_c (which adds the format addendum) and variant_d
    (which must NOT get the addendum, to stay byte-for-byte identical to
    Sajjad's official 'First version' email)."""
    defs = "\n\n".join(f"{label}:\n{LONG_DEFINITIONS[label]}" for label in LABELS)
    return f"{_base_instructions()}\n\nLabels and detailed definitions:\n\n{defs}"


def variant_c_long_defs() -> tuple[str, str]:
    system = _long_defs_core() + OUTPUT_FORMAT_ADDENDUM
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def variant_d_few_shot() -> tuple[str, str]:
    """This is Sajjad's official 'First version' prompt, sent verbatim via
    email. DO NOT add the output-format addendum or otherwise edit this --
    he explicitly asked for it to be preserved unchanged. (Variant G is
    where his requested reasoning + output format additions go.)"""
    system_base = _long_defs_core()
    examples_text = "\n".join(
        f'Tweet: "{tweet}"\nLabel: {label}\n' for tweet, label in FEW_SHOT_EXAMPLES
    )
    system = f"{system_base}\n\nExamples:\n{examples_text}"
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def _few_shot_variant(examples: list[tuple[str, str]]) -> tuple[str, str]:
    """Shared builder for Variant D and its label-flipped siblings (I, J) --
    identical structure/definitions to variant_d_few_shot, only the example
    labels differ, so score comparisons against D isolate that one change."""
    system_base = _long_defs_core()
    examples_text = "\n".join(
        f'Tweet: "{tweet}"\nLabel: {label}\n' for tweet, label in examples
    )
    system = f"{system_base}\n\nExamples:\n{examples_text}"
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def variant_i_label_flipped_random() -> tuple[str, str]:
    """Variant D's exact 3 examples, each reassigned to a different, arbitrary
    wrong label. Replicates the label-flipping experiment from arXiv:2307.05052
    (Sajjad's suggestion): compare against Variant D to see how much accuracy
    degrades when demonstration labels are wrong but plausible-looking."""
    return _few_shot_variant(FEW_SHOT_EXAMPLES_FLIPPED_RANDOM)


def variant_j_label_flipped_adversarial() -> tuple[str, str]:
    """Variant D's exact 3 examples, but only the genuine crisis example is
    flipped to '1-No Concern' (other two left correct as controls). Tests the
    project's actual worst-case failure mode: one bad demonstration teaching
    the model that crisis language is nothing to flag."""
    return _few_shot_variant(FEW_SHOT_EXAMPLES_FLIPPED_ADVERSARIAL)


def variant_k_complementary_explanations() -> tuple[str, str]:
    """Paper's third technique (arXiv:2307.05052): attach a short explanation
    to each few-shot example (WHY it got that label), rather than asking the
    model to generate its own reasoning (that's what E/G already do). Same 3
    tweets/labels as Variant D -- only the explanations are new -- so this
    isolates "does explaining the demos help" from "does the model reasoning
    out loud help". The paper found explanations didn't move accuracy for
    sentiment analysis; worth checking whether that holds for this task too."""
    system_base = _long_defs_core()
    explained_examples = [
        ("Happy birthday to the beloved king of grunge #KurtCobain", "1-No Concern",
         "This is a tribute to a public figure, not a statement about the author's own state."),
        ("I can't make it through this one guys I'm sorry", "3-Self-Identified Crisis Symptoms and Seeking Support",
         "The author expresses being overwhelmed and addresses an audience directly -- an implicit reach for support."),
        ("Rush Limbaugh has died rot in pieces you racist sack of crap", "6- Not Relevant to the Research",
         "A reaction to a public figure's death; says nothing about the author's own mental/emotional state."),
    ]
    examples_text = "\n".join(
        f'Tweet: "{tweet}"\nLabel: {label}\nExplanation: {explanation}\n'
        for tweet, label, explanation in explained_examples
    )
    system = f"{system_base}\n\nExamples:\n{examples_text}"
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def variant_h_reviewer_matched(reasoning_word_limit: int = None) -> tuple[str, str]:
    """Per Sajjad's original intent (not the accuracy-maximizing ablation
    goal of variants A-G): give the AI roughly the same information the
    human reviewers had -- short definitions matching the original SME
    instruction sheet -- with NO few-shot examples.

    reasoning_word_limit=None (default): lean, label-only response. Use
    this for broad multi-model sweeps where you just want to compare
    accuracy/macro-F1 across many models fast and cheap.

    reasoning_word_limit=<int> (e.g. 30): adds a short reasoning
    requirement so the choice is auditable/spot-checkable, similar to
    what we did by hand in the Playground testing. Use this for a
    follow-up drill-down on 1-2 specific models you want to understand
    better (e.g. the best and worst performers from the broad sweep) --
    not for the whole sweep, since it adds real runtime across many calls.
    """
    has_reasoning = reasoning_word_limit is not None
    defs = _short_defs_text(include_other_reason_note=has_reasoning)
    base = f"{_base_instructions()}\n\nLabels and definitions:\n{defs}"

    if not has_reasoning:
        system = base + OUTPUT_FORMAT_ADDENDUM
        user = "Tweet: \"{tweet}\"\n\nLabel:"
    else:
        system = (
            f"{base}\n\nRespond in EXACTLY this format, with no other text:\n"
            f"Reasoning: <no more than {reasoning_word_limit} words explaining your choice>\n"
            f"Label: <one of the exact label names from the list above>"
        )
        user = "Tweet: \"{tweet}\"\n\nReasoning and label:"
    return system, user


def variant_h_reviewer_matched_with_reasoning() -> tuple[str, str]:
    """Follow-up drill-down version of H, with a 30-word reasoning
    requirement. Not part of the broad multi-model sweep -- run this
    separately, only against models you want to understand more deeply."""
    return variant_h_reviewer_matched(reasoning_word_limit=30)


def variant_g_official_reasoning_format() -> tuple[str, str]:
    """This is Sajjad's exact request from his email: take the 'First
    version' prompt (which is identical to variant_d_few_shot -- same
    definitions, same 3 examples) and add explicit output format +
    reasoning instructions on top, WITHOUT changing the original text.

    Do not edit variant_d_few_shot() itself -- Sajjad said not to discard
    the previous prompt, so it needs to stay intact and available on its
    own for comparison.
    """
    system_base, _ = variant_d_few_shot()
    addition = (
        "\n\nBefore giving your final answer, briefly reason step-by-step "
        "(1-2 sentences) about the tweet's tone, intent, and any relevant "
        "context that influences the label.\n\n"
        "Respond in EXACTLY this format, with no other text:\n"
        "Reasoning: <your brief reasoning>\n"
        "Label: <one of the exact label names from the list above>"
    )
    system = system_base + addition
    user = "Tweet: \"{tweet}\"\n\nReasoning and label:"
    return system, user


def variant_f_refined_definitions() -> tuple[str, str]:
    """Built from the actual 149-tweet ablation results, not a guess. Targets
    three specific failure modes found in that run:
      1. Label 3 tweets undershooting to Label 2 (audience-address not
         recognized as seeking support)
      2. Label 5 tweets undershooting to Label 4 (implicit/no-keyword
         crisis language not recognized)
      3. Label 1 tweets over-escalating to Label 3 (diagnosis-for-awareness
         confused with active distress)

    Uses REFINED_DEFINITIONS (patched versions of labels 1, 3, 5) plus
    NEW synthetic few-shot examples -- deliberately NOT pulled from the
    evaluation CSV, unlike variants D/E, so this comparison has no leakage.
    """
    defs = "\n\n".join(f"{label}:\n{REFINED_DEFINITIONS[label]}" for label in LABELS)

    synthetic_examples = [
        ("Is anyone else awake right now? I really need someone to talk to.",
         "3-Self-Identified Crisis Symptoms and Seeking Support"),
        ("This is it for me. Just wanted to say goodbye everyone.",
         "5- Acute Crisis with Suicidal Ideation"),
        ("Proud to share my journey with anxiety, depression and PTSD for "
         "#MentalHealthAwarenessMonth. Recovery is possible.",
         "1-No Concern"),
    ]
    examples_text = "\n".join(
        f'Tweet: "{tweet}"\nLabel: {label}\n' for tweet, label in synthetic_examples
    )

    system = (
        f"{_base_instructions()}\n\nLabels and detailed definitions:\n\n{defs}"
        f"\n\nExamples:\n{examples_text}"
    ) + OUTPUT_FORMAT_ADDENDUM
    user = "Tweet: \"{tweet}\"\n\nLabel:"
    return system, user


def variant_e_chain_of_thought() -> tuple[str, str]:
    system_base, _ = variant_d_few_shot()
    system = (
        f"{system_base}\n\n"
        "Think through the tweet's tone and intent in 1-2 short sentences first. "
        "Then, on a NEW final line, output only the label text by itself, "
        "prefixed with 'FINAL:' e.g. 'FINAL: 1-No Concern'."
    )
    user = "Tweet: \"{tweet}\"\n\nReasoning and label:"
    return system, user


# Registry used by the runner scripts. Add/remove variants here.
PROMPT_VARIANTS = {
    "A_bare_labels": variant_a_bare,
    "B_short_definitions": variant_b_short_defs,
    "C_long_definitions": variant_c_long_defs,
    "D_few_shot": variant_d_few_shot,
    "I_label_flipped_random": variant_i_label_flipped_random,
    "J_label_flipped_adversarial": variant_j_label_flipped_adversarial,
    "K_complementary_explanations": variant_k_complementary_explanations,
    "E_chain_of_thought": variant_e_chain_of_thought,
    "F_refined_definitions": variant_f_refined_definitions,
    "G_official_reasoning_format": variant_g_official_reasoning_format,
    "H_reviewer_matched": variant_h_reviewer_matched,
    "H_reviewer_matched_with_reasoning": variant_h_reviewer_matched_with_reasoning,
}
