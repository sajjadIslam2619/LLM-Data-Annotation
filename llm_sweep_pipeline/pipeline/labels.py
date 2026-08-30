"""
Canonical label taxonomy for the veteran crisis-tweet labeling task.

This mirrors the final label scheme used by the human SME reviewers
(see "Veteran SME Labelling Exercise" instructions + Dissertation Ch. 5,
section 5.7). Keeping this in one place means every script (ground truth
builder, OpenAI ablation runner, Ollama runner) uses the exact same
label set and definitions.
"""

import re

# Canonical label strings -- these MUST match how labels appear in the
# ground-truth CSV (Veteran_Tweet_Label_PTSD_with_ground_truth.csv),
# so scoring can do exact / near-exact string matches.
LABELS = [
    "1-No Concern",
    "2- Distressed Self-Disclosure",
    "3-Self-Identified Crisis Symptoms and Seeking Support",
    "4- Self-Identified Crisis Symptoms but Not Seeking Support",
    "5- Acute Crisis with Suicidal Ideation",
    "6- Not Relevant to the Research",
    "Other",
]

# Short, one-line definitions -- roughly what the human reviewers were
# given in the SME instruction sheet.
SHORT_DEFINITIONS = {
    "1-No Concern": "Neutral, acknowledgement, accepting. No signs of distress.",
    "2- Distressed Self-Disclosure": "Avoiding, concerning, diminishing. Expresses distress/venting but not crisis.",
    "3-Self-Identified Crisis Symptoms and Seeking Support": "Disclosing, re-experiencing, expressing, guilt/shame, exhaustion, AND explicitly asking for help.",
    "4- Self-Identified Crisis Symptoms but Not Seeking Support": "Same crisis symptoms as #3, but NOT asking for help or intervention.",
    "5- Acute Crisis with Suicidal Ideation": "Crisis, suicidal, final, defeated. Explicit suicidal thoughts or plans.",
    "6- Not Relevant to the Research": "Tweet has nothing to do with veterans, mental health, or crisis.",
    "Other": "Doesn't fit any category above, but include a reason if picking this.",
}

# Longer definitions pulled from Dissertation Chapter 5 (5.7.1 - 5.7.5).
# Used for the "full definition" ablation variant.
LONG_DEFINITIONS = {
    "1-No Concern": (
        "Neutral or positive text that doesn't indicate distress. This can be tricky: "
        "a tweet might contain crisis-adjacent words (PTSD, suicide, capital letters, "
        "exclamation points) while the actual tone/intent is not distressed at all "
        "(e.g., raising awareness or making a factual/critical comment)."
    ),
    "2- Distressed Self-Disclosure": (
        "Expressions of anxiety, general distress, or worry that reflect emotional "
        "unrest, but do NOT rise to the level of crisis. The key challenge is telling "
        "distress apart from ordinary day-to-day complaints or discomfort."
    ),
    "3-Self-Identified Crisis Symptoms and Seeking Support": (
        "The person explicitly describes being in a personal crisis AND is asking "
        "for help, support, or validation -- directly or through hashtags/context "
        "that signal a request for support."
    ),
    "4- Self-Identified Crisis Symptoms but Not Seeking Support": (
        "The person explicitly describes being in a personal crisis but shows no "
        "intention of seeking help. May include jokes, sarcasm, or euphemism about "
        "their situation, or a stated unwillingness to reach out."
    ),
    "5- Acute Crisis with Suicidal Ideation": (
        "The most critical category. Explicit suicidal ideation, plans, or "
        "descriptions of a suicide attempt. Precision here matters most because "
        "false negatives can be life-threatening."
    ),
    "6- Not Relevant to the Research": (
        "The tweet is not about the author's own mental/emotional state at all "
        "(e.g., news, trivia, celebrity mentions, weather complaints)."
    ),
    "Other": (
        "Use only if the tweet genuinely does not fit any category above. "
        "Should be rare -- if you pick this, a short reason is expected."
    ),
}



# --- Refined definitions (Variant F) ---
# Built directly from the failure patterns found in the 149-tweet ablation
# run: (1) label 3 undershooting to label 2 when a tweet addresses an
# audience without a direct "please help" ask, (2) label 5 undershooting
# to label 4 when crisis language is implicit rather than using explicit
# keywords like "suicide", and (3) label 1/3 confusion when a tweet lists
# clinical/diagnostic terms for awareness purposes rather than active
# distress. Only labels 1, 3, and 5 are patched; 2, 4, 6, and Other are
# unchanged from LONG_DEFINITIONS.
REFINED_DEFINITIONS = dict(LONG_DEFINITIONS)
REFINED_DEFINITIONS["1-No Concern"] = (
    LONG_DEFINITIONS["1-No Concern"] + " IMPORTANT: sharing a past or ongoing "
    "diagnosis for awareness/advocacy purposes (e.g. hashtags like "
    "#BellLetsTalk, #MentalHealthAwareness) is No Concern, even if it lists "
    "multiple clinical terms (depression, PTSD, suicidal ideation, etc.), as "
    "long as the tweet is not expressing current distress or asking for help."
)
REFINED_DEFINITIONS["3-Self-Identified Crisis Symptoms and Seeking Support"] = (
    LONG_DEFINITIONS["3-Self-Identified Crisis Symptoms and Seeking Support"] +
    " IMPORTANT: 'seeking support' does not require a direct, explicit ask "
    "like 'please help me'. Addressing an audience while in distress "
    "('anyone out there?', 'guys...', 'is somebody awake') counts as "
    "seeking support -- the person is reaching out for a response, even "
    "indirectly."
)
REFINED_DEFINITIONS["5- Acute Crisis with Suicidal Ideation"] = (
    LONG_DEFINITIONS["5- Acute Crisis with Suicidal Ideation"] +
    " IMPORTANT: do NOT require the words 'suicide' or 'kill myself' to "
    "assign this label. Farewell/goodbye language ('bye', 'I'm gone', "
    "'this is it', 'can't do this anymore'), or imagery of fading, "
    "disappearing, or finality, can indicate acute crisis just as strongly "
    "as explicit language. When genuinely unsure between label 4 and label "
    "5, and there is any hint of finality or farewell, prefer label 5 -- "
    "missing a real crisis (false negative) is the worst possible error "
    "for this label."
)


def label_key(label: str) -> str:
    """Extract the leading number (or 'Other') from a label string.

    '2- Distressed Self-Disclosure' -> '2'
    'Other'                          -> 'Other'
    """
    m = re.match(r"\s*(\d+)", label)
    return m.group(1) if m else label.strip()


def normalize_model_output(raw_text: str) -> str:
    """Map a raw LLM response to one of the canonical LABELS.

    Handles cases where the model just outputs a number ('3'), a number
    plus text ('3 - seeking support'), or the full label string. Returns
    the closest canonical label, or 'UNPARSEABLE' if nothing matches.
    """
    if not raw_text:
        return "UNPARSEABLE"

    text = raw_text.strip()

    # If the model followed instructions and gave just a digit (or "Other")
    # look at the LAST line, since chain-of-thought variants reason first
    # and put the final answer on the last line.
    last_line = text.strip().splitlines()[-1].strip()

    for candidate in (last_line, text):
        # exact / near match on canonical label string
        for label in LABELS:
            if candidate.lower() == label.lower():
                return label
        # match on "Other"
        if candidate.strip().lower() == "other":
            return "Other"
        # match on leading digit, e.g. "3", "3.", "3 -", "Label 3"
        m = re.search(r"\b([1-6])\b", candidate)
        if m:
            num = m.group(1)
            for label in LABELS:
                if label_key(label) == num:
                    return label

    return "UNPARSEABLE"
