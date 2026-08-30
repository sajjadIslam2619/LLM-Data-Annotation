"""
Compute the Vendi Score (Friedman & Dieng, 2023) to evaluate dataset
diversity/quality, per Sajjad's request.

What the Vendi Score is: a diversity metric that takes a set of samples
and a pairwise similarity function, and returns the "effective number of
unique items" in the set. If you have 149 tweets but many are near-
duplicates of each other, the Vendi Score will be meaningfully lower than
149 -- that's the signal that the dataset has less real diversity than
its raw row count suggests.

We use n-gram (word 1-gram + 2-gram) Jaccard similarity as the kernel,
rather than the package's built-in nltk-based tokenizer, so this runs
with zero extra downloads (no nltk punkt data needed).

Reference: https://github.com/vertaix/Vendi-Score
Paper: https://arxiv.org/abs/2210.02410

Usage:
  pip install vendi_score
  python compute_vendi_score.py
"""

import re

import pandas as pd
from vendi_score import vendi

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"


def get_ngrams(text: str, n: int) -> set:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def ngram_set(text: str, ns=(1, 2)) -> set:
    s = set()
    for n in ns:
        s |= get_ngrams(text, n)
    return s


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def most_similar_pairs(tweets, ngram_sets, top_n=10):
    """Find the most similar tweet pairs -- useful for spot-checking what's
    dragging the Vendi Score down (near-duplicates, template tweets, etc.)."""
    pairs = []
    for i in range(len(tweets)):
        for j in range(i + 1, len(tweets)):
            sim = jaccard(ngram_sets[i], ngram_sets[j])
            if sim > 0:
                pairs.append((sim, tweets[i], tweets[j]))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:top_n]


def main():
    df = pd.read_csv(GROUND_TRUTH_CSV)
    tweets = df["tweet"].tolist()
    ngram_sets = [ngram_set(t) for t in tweets]

    overall_vs = vendi.score(ngram_sets, jaccard)
    print("=== Overall Dataset Vendi Score ===")
    print(f"Vendi Score: {overall_vs:.2f} (out of {len(tweets)} tweets)")
    print(f"Effective diversity ratio: {overall_vs / len(tweets):.1%}")
    print("(100% would mean every tweet is maximally distinct; lower means")
    print(" more redundancy/near-duplication than the raw row count suggests)")

    print("\n=== Vendi Score by Ground-Truth Label (majority rows only) ===")
    print("(Lower relative diversity within a class = that class's examples")
    print(" may be too repetitive to teach the model much variety)")
    majority = df[df["agreement_status"] == "majority"]
    for label, group in majority.groupby("ground_truth_label"):
        class_tweets = group["tweet"].tolist()
        if len(class_tweets) < 2:
            print(f"  {label}: only {len(class_tweets)} tweet(s), Vendi Score not meaningful")
            continue
        class_ngrams = [ngram_set(t) for t in class_tweets]
        class_vs = vendi.score(class_ngrams, jaccard)
        print(f"  {label}: Vendi Score {class_vs:.2f} / {len(class_tweets)} tweets "
              f"({class_vs/len(class_tweets):.1%} effective diversity)")

    print("\n=== Top 10 Most Similar Tweet Pairs (likely near-duplicates) ===")
    top_pairs = most_similar_pairs(tweets, ngram_sets, top_n=10)
    for sim, t1, t2 in top_pairs:
        print(f"\n  similarity={sim:.2f}")
        print(f"    A: {t1}")
        print(f"    B: {t2}")


if __name__ == "__main__":
    main()
