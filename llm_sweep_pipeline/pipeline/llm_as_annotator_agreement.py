"""
Treats each LLM prediction as "one more annotator" and asks the question the
new paper framing (Sajjad's suggested title: "Can LLMs achieve or exceed
human annotation quality when humans themselves disagree?") actually needs
answered: is the LLM's agreement with individual human reviewers as good as,
or better than, the reviewers' agreement with EACH OTHER?

Why this matters here specifically: Phase 1 EDA already found average
pairwise human-human Cohen's Kappa = 0.156 ("slight agreement") on this
dataset. Every accuracy/macro-F1 number reported so far measures the LLM
against the majority-vote label -- but that majority vote is itself built
from reviewers who barely agreed with each other. This script instead asks:
if you swap the LLM in as an extra rater, does it agree with each human
reviewer about as much as reviewers agree with each other, more, or less?
That's the actual comparison the reading list's papers (Gilardi et al. 2023;
Törnberg 2023) make for their own tasks, and it's a fairer bar than "accuracy
against a shaky ground truth."

No new model calls -- reuses whichever *_results.csv from run_ollama_compare.py
/ run_new_experiments.py is passed in (reviewer1-8 + a pred_<variant> column
are already in those files).

Usage:
  python llm_as_annotator_agreement.py --file D_few_shot_llama3.1-8b_results.csv --pred-col pred_D_few_shot
"""

import argparse

import pandas as pd
import numpy as np

REVIEWER_COLS = [f"reviewer{i}" for i in range(1, 9)]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's Kappa between two label series over their shared label space,
    implemented directly (no sklearn dependency)."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    labels = sorted(set(df["a"]) | set(df["b"]))
    n = len(df)
    if n == 0:
        return float("nan")

    idx = {l: i for i, l in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)))
    for a_val, b_val in zip(df["a"], df["b"]):
        confusion[idx[a_val], idx[b_val]] += 1

    po = np.trace(confusion) / n
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    pe = (row_marg * col_marg).sum()
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="A *_results.csv from run_ollama_compare.py / run_new_experiments.py")
    parser.add_argument("--pred-col", required=True, help="e.g. pred_D_few_shot")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    missing = [c for c in REVIEWER_COLS if c not in df.columns]
    if missing or args.pred_col not in df.columns:
        raise SystemExit(f"Missing columns: {missing + ([args.pred_col] if args.pred_col not in df.columns else [])}")

    # Human-human pairwise kappa (all 28 pairs among the 8 reviewers)
    human_pairs = []
    for i in range(len(REVIEWER_COLS)):
        for j in range(i + 1, len(REVIEWER_COLS)):
            k = cohen_kappa(df[REVIEWER_COLS[i]], df[REVIEWER_COLS[j]])
            human_pairs.append((REVIEWER_COLS[i], REVIEWER_COLS[j], k))
    human_kappas = [k for _, _, k in human_pairs]

    # LLM-vs-each-human kappa (8 pairs)
    llm_pairs = [(r, cohen_kappa(df[args.pred_col], df[r])) for r in REVIEWER_COLS]
    llm_kappas = [k for _, k in llm_pairs]

    print(f"=== LLM-as-annotator agreement | {args.pred_col} ({args.file}) ===\n")
    print(f"Human-human pairwise Kappa: mean={np.mean(human_kappas):.3f}  "
          f"min={np.min(human_kappas):.3f}  max={np.max(human_kappas):.3f}  (n={len(human_kappas)} pairs)")
    print(f"LLM-vs-human Kappa:         mean={np.mean(llm_kappas):.3f}  "
          f"min={np.min(llm_kappas):.3f}  max={np.max(llm_kappas):.3f}  (n={len(llm_kappas)} pairs)\n")

    print("LLM vs. each reviewer:")
    for r, k in sorted(llm_pairs, key=lambda x: -x[1]):
        print(f"  {r}: kappa={k:.3f}")

    delta = np.mean(llm_kappas) - np.mean(human_kappas)
    verdict = "EXCEEDS" if delta > 0.02 else ("MATCHES" if abs(delta) <= 0.02 else "FALLS SHORT OF")
    print(f"\nLLM's average agreement with individual humans {verdict} average human-human agreement "
          f"(delta={delta:+.3f}).")

    out = pd.DataFrame(
        [{"pair_type": "human-human", "rater_a": a, "rater_b": b, "kappa": k} for a, b, k in human_pairs] +
        [{"pair_type": "llm-human", "rater_a": args.pred_col, "rater_b": r, "kappa": k} for r, k in llm_pairs]
    )
    out_path = f"agreement_{args.pred_col}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
