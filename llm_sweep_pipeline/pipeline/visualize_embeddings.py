"""
SBERT + UMAP visualization of the tweet dataset, per Sajjad's suggestion
(email, "Visualization approach").

What this does, plainly:
  1. Reads each tweet and places it on an imaginary "meaning map" using a
     pretrained sentence-embedding model (SBERT) -- tweets with similar
     meaning end up near each other, regardless of exact wording.
  2. That map has too many directions (384) to draw, so UMAP squishes it
     down to 2 directions we can actually plot on a page.
  3. Plots the 149 tweets as dots, colored by their human-assigned
     ground-truth label, so you can see by eye whether the label
     categories form distinct clusters or overlap/blur together.

Why it matters for this project specifically: the EDA (Phase 1) already
found weak human reviewer agreement (Cohen's Kappa 0.156) and specific
class-confusion patterns (label 2/3, label 4/5). This plot checks whether
that fuzziness reflects genuinely ambiguous tweet content (overlapping
clusters) or inconsistent labeling of otherwise-distinct tweets (separated
clusters, mixed colors).

Runs entirely locally -- no OpenAI key, no Ollama server. First run
downloads a small (~90MB) embedding model from Hugging Face.

Setup:
  pip install sentence-transformers umap-learn matplotlib plotly pandas

Usage:
  python visualize_embeddings.py
  python visualize_embeddings.py --exclude-ties   # drop tweets with no majority label
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sentence_transformers import SentenceTransformer
import umap

GROUND_TRUTH_CSV = "Veteran_Tweet_Label_PTSD_with_ground_truth.csv"

# Consistent, readable colors per label -- severity roughly maps to a
# green -> yellow -> orange -> red progression, with grey/blue for the
# two non-severity categories.
LABEL_COLORS = {
    "1-No Concern": "#2ca02c",
    "2- Distressed Self-Disclosure": "#bcbd22",
    "3-Self-Identified Crisis Symptoms and Seeking Support": "#ff7f0e",
    "4- Self-Identified Crisis Symptoms but Not Seeking Support": "#d62728",
    "5- Acute Crisis with Suicidal Ideation": "#8b0000",
    "6- Not Relevant to the Research": "#7f7f7f",
    "Other": "#1f77b4",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-ties", action="store_true",
                         help="Drop tweets where reviewers had no majority label (agreement_status != 'majority')")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                         help="SBERT model name (default: small, fast, good general-purpose)")
    args = parser.parse_args()

    df = pd.read_csv(GROUND_TRUTH_CSV)
    if args.exclude_ties:
        df = df[df["agreement_status"] == "majority"].reset_index(drop=True)

    print(f"Embedding {len(df)} tweets with {args.model}...")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(df["tweet"].tolist(), show_progress_bar=True)

    print("Projecting to 2D with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # Static PNG (matplotlib) -- for docs/slides
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color in LABEL_COLORS.items():
        subset = df[df["ground_truth_label"] == label]
        if len(subset) == 0:
            continue
        ax.scatter(subset["x"], subset["y"], c=color, label=label, s=40,
                   alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.set_title("Tweet embeddings (SBERT + UMAP), colored by ground-truth label")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig("tweet_embeddings_umap.png", dpi=150)
    print("Saved: tweet_embeddings_umap.png")

    # Interactive HTML (plotly) -- hover over a dot to read the actual tweet,
    # useful for spotting exactly which tweets land in overlap/blur zones.
    fig2 = px.scatter(
        df, x="x", y="y", color="ground_truth_label",
        color_discrete_map=LABEL_COLORS,
        hover_data={"tweet": True, "x": False, "y": False, "ground_truth_label": False},
        title="Tweet embeddings (SBERT + UMAP) -- hover to read each tweet",
    )
    fig2.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
    fig2.write_html("tweet_embeddings_umap.html")
    print("Saved: tweet_embeddings_umap.html (open in a browser, hover over dots)")

    df.to_csv("tweet_embeddings_umap_coords.csv", index=False)
    print("Saved: tweet_embeddings_umap_coords.csv (x/y coords + labels, for further analysis)")


if __name__ == "__main__":
    main()
