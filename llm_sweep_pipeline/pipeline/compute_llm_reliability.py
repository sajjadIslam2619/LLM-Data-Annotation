"""
Inter-model reliability for LLM predictions -- the model-side analog of
Phase 1 / Phase 19's human inter-rater reliability table (RESEARCH_LOG.md).

Treats each model's predictions on a given prompt variant as one "rater",
the same way the 8 human reviewers were treated as raters in the original
table. Computes the same five metrics: Majority agreement, Full (100%)
agreement, Fleiss' kappa, Krippendorff's alpha (nominal + ordinal), and a
Normalized Reviewer Vendi Score -- so results are directly comparable to
the human numbers.

Methodology validated against the known human Phase 19 numbers on
Veteran_Tweet_Label_PTSD (1).csv before being trusted on model data --
all 6 metrics reproduce exactly (see git/chat history for the check).
Key methodology notes discovered during that validation:

  - Fleiss' kappa here uses the standard generalization for a VARYING
    number of raters per item (P_i divided by n_i(n_i-1) per item, P_e
    from pooled category proportions) rather than the textbook fixed-n
    version, since UNPARSEABLE model responses are missing data, not a
    7th category, and different tweets end up with different numbers of
    valid (parseable) predictions.
  - Krippendorff's alpha (nominal) is computed over all 7 canonical
    labels including "6- Not Relevant" and "Other".
  - Krippendorff's alpha (ordinal) EXCLUDES "6- Not Relevant" and "Other"
    entirely (treated as missing, not merged into a shared code) -- only
    labels 1-5 form a real severity scale. This exact exclusion rule is
    what reproduces the human Phase 19 ordinal alpha (0.4941) exactly;
    an earlier attempt that merged "6"/"Other" into severity 0 (matching
    labels.py's severity_map, meant for a different purpose) does not.
  - Normalized Reviewer Vendi Score: Vendi Score (Friedman & Dieng 2023)
    over the set of raters, using pairwise raw observed label-agreement
    rate (on tweets both raters answered) as the similarity kernel, then
    divided by the rater count -- the same normalization convention
    compute_vendi_score.py uses for tweet-diversity (VS / n).
  - A model is EXCLUDED as a rater for a given variant if its UNPARSEABLE
    rate on that variant exceeds 50% -- otherwise a badly-broken model
    would look like a maximally-disagreeing outlier for nearly every
    tweet and mechanically crater every chance-corrected metric, which
    measures brokenness, not disagreement. Excluded models are printed
    and saved so nothing is silently dropped.

Usage:
  python compute_llm_reliability.py
  python compute_llm_reliability.py --variants F_refined_definitions,H_reviewer_matched
Outputs: llm_reliability_metrics.csv
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd
import krippendorff
from vendi_score import vendi

LABELS = [
    "1-No Concern", "2- Distressed Self-Disclosure",
    "3-Self-Identified Crisis Symptoms and Seeking Support",
    "4- Self-Identified Crisis Symptoms but Not Seeking Support",
    "5- Acute Crisis with Suicidal Ideation",
    "6- Not Relevant to the Research", "Other",
]
NOMINAL_CODE = {l: i for i, l in enumerate(LABELS)}
UNPARSEABLE_EXCLUDE_THRESHOLD = 0.5


def label_key(label):
    if pd.isna(label):
        return None
    m = re.match(r"\s*(\d+)", str(label))
    return m.group(1) if m else str(label).strip()


def ordinal_code(label):
    k = label_key(label)
    if k is None or k in ("6", "Other"):
        return None
    return int(k)


def build_matrix(variant, exclude_unparseable_over=UNPARSEABLE_EXCLUDE_THRESHOLD):
    files = sorted(glob.glob(f"{variant}_*_results.csv"))
    col = f"pred_{variant}"
    models, preds, excluded = [], {}, []
    for f in files:
        m = re.match(rf"{variant}_(.+)_results\.csv", f)
        model = m.group(1) if m else f
        df = pd.read_csv(f)
        rate = (df[col] == "UNPARSEABLE").mean()
        if rate > exclude_unparseable_over:
            excluded.append((model, round(rate, 4)))
            continue
        models.append(model)
        preds[model] = df[col].tolist()
    if not models:
        return None, [], excluded, 0
    n_tweets = len(next(iter(preds.values())))
    mat = pd.DataFrame(preds).replace("UNPARSEABLE", np.nan)
    return mat, models, excluded, n_tweets


def majority_full_agreement(mat):
    maj_count = full_count = valid_tweets = 0
    for _, row in mat.iterrows():
        vals = row.dropna().apply(label_key)
        n = len(vals)
        if n < 2:
            continue
        valid_tweets += 1
        vc = vals.value_counts()
        top = vc.iloc[0]
        if top > n / 2:
            maj_count += 1
        if top == n:
            full_count += 1
    return maj_count / valid_tweets, full_count / valid_tweets, valid_tweets


def fleiss_kappa_varying_n(mat):
    """Generalized Fleiss' kappa allowing a different number of raters
    per item (item excluded if fewer than 2 valid ratings)."""
    Pis, all_ns, cat_totals = [], [], {}
    for _, row in mat.iterrows():
        vals = row.dropna().apply(label_key)
        n = len(vals)
        if n < 2:
            continue
        vc = vals.value_counts()
        Pi = sum(c * (c - 1) for c in vc) / (n * (n - 1))
        Pis.append(Pi)
        all_ns.append(n)
        for cat, c in vc.items():
            cat_totals[cat] = cat_totals.get(cat, 0) + c
    P_bar = sum(Pis) / len(Pis)
    total = sum(all_ns)
    Pe = sum((c / total) ** 2 for c in cat_totals.values())
    return (P_bar - Pe) / (1 - Pe)


def krippendorff_alpha(mat, mode="nominal"):
    fn = (lambda l: NOMINAL_CODE.get(l) if not pd.isna(l) else None) if mode == "nominal" else ordinal_code
    coded = mat.apply(lambda col: col.map(fn))
    rel = coded.to_numpy(dtype=float).T
    return krippendorff.alpha(reliability_data=rel, level_of_measurement=mode)


def reviewer_vendi(mat, models):
    def sim(m1, m2):
        a, b = mat[m1], mat[m2]
        both = a.notna() & b.notna()
        if both.sum() == 0:
            return 0.0
        return (a[both].apply(label_key) == b[both].apply(label_key)).mean()
    vs = vendi.score(models, sim)
    return vs, vs / len(models)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="F_refined_definitions,H_reviewer_matched")
    parser.add_argument("--unparseable-threshold", type=float, default=UNPARSEABLE_EXCLUDE_THRESHOLD)
    args = parser.parse_args()

    rows = []
    for variant in args.variants.split(","):
        mat, models, excluded, n_tweets = build_matrix(variant, args.unparseable_threshold)
        if mat is None:
            print(f"{variant}: no usable model files found, skipping")
            continue
        maj, full, valid_tweets = majority_full_agreement(mat)
        kappa = fleiss_kappa_varying_n(mat)
        alpha_nom = krippendorff_alpha(mat, "nominal")
        alpha_ord = krippendorff_alpha(mat, "ordinal")
        vs, vs_norm = reviewer_vendi(mat, models)

        print("=" * 70)
        print(f"{variant} -- {len(models)} models included")
        print(f"  Excluded (>{args.unparseable_threshold:.0%} unparseable): {[e[0] for e in excluded]}")
        print(f"  Majority agreement (>50%): {maj*100:.2f}%")
        print(f"  Full agreement (100%):     {full*100:.2f}%")
        print(f"  Fleiss' kappa:             {kappa:.4f}")
        print(f"  Krippendorff alpha (nom):  {alpha_nom:.4f}")
        print(f"  Krippendorff alpha (ord):  {alpha_ord:.4f}")
        print(f"  Normalized Vendi Score:    {vs_norm*100:.2f}%  (raw {vs:.2f}/{len(models)})")

        rows.append({
            "variant": variant,
            "n_models_included": len(models),
            "models_included": ";".join(models),
            "models_excluded": ";".join(e[0] for e in excluded),
            "majority_agreement_pct": round(maj * 100, 2),
            "full_agreement_pct": round(full * 100, 2),
            "fleiss_kappa": round(kappa, 4),
            "krippendorff_alpha_nominal": round(alpha_nom, 4),
            "krippendorff_alpha_ordinal": round(alpha_ord, 4),
            "vendi_score_raw": round(vs, 2),
            "normalized_vendi_pct": round(vs_norm * 100, 2),
        })

    pd.DataFrame(rows).to_csv("llm_reliability_metrics.csv", index=False)
    print("\nSaved: llm_reliability_metrics.csv")


if __name__ == "__main__":
    main()
