"""
Build a majority-vote ground truth label for each tweet from the 8
reviewer columns in Veteran_Tweet_Label_PTSD.csv.

Why: 8 different humans labeled each tweet and they frequently disagree
(that's expected -- crisis language is genuinely ambiguous, and the
dissertation itself flags low inter-rater agreement as a limitation).
Before you can score any LLM prompt, you need one single "correct"
answer per tweet to compare against. Majority vote is the simplest,
most standard way to collapse 8 opinions into one label.

Rows where there's a TIE for the most common label (no single label
has strictly the most votes) are flagged as agreement_status == "tie".
These are excluded from scoring by default because there's no reliable
ground truth for them -- but they're worth a manual look, since they're
probably the genuinely hardest/most ambiguous tweets.

Usage:
  python build_ground_truth.py --input "Veteran_Tweet_Label_PTSD (1).csv" \
      --output Veteran_Tweet_Label_PTSD_with_ground_truth.csv
"""

import argparse
from collections import Counter

import pandas as pd


def majority_label(row):
    counts = Counter(row.dropna())
    if not counts:
        return None, 0, "no_labels"
    top_label, top_count = counts.most_common(1)[0]
    tied = [label for label, c in counts.items() if c == top_count]
    status = "tie" if len(tied) > 1 else "majority"
    return top_label, top_count, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Veteran_Tweet_Label_PTSD (1).csv")
    parser.add_argument("--output", default="Veteran_Tweet_Label_PTSD_with_ground_truth.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    reviewer_cols = [c for c in df.columns if c.startswith("reviewer")]
    if not reviewer_cols:
        raise SystemExit("No columns starting with 'reviewer' found -- check the CSV format.")

    results = df[reviewer_cols].apply(majority_label, axis=1)
    df["ground_truth_label"] = results.apply(lambda x: x[0])
    df["agreement_count"] = results.apply(lambda x: x[1])
    df["agreement_status"] = results.apply(lambda x: x[2])

    n_tie = (df["agreement_status"] == "tie").sum()
    print(f"Built ground truth for {len(df)} tweets.")
    print(f"  {len(df) - n_tie} have a clear majority label.")
    print(f"  {n_tie} are ties (no clear majority) -- excluded from scoring by default.")
    print("\nGround truth label distribution (majority rows only):")
    print(df.loc[df["agreement_status"] == "majority", "ground_truth_label"].value_counts())

    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
