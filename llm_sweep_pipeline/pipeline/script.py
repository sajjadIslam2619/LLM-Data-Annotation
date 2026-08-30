import pandas as pd
df = pd.read_csv('Veteran_Tweet_Label_PTSD (1).csv')
print(df.head())
print(f"\nTotal number of tweets: {len(df)}")

df['tweet_length'] = df['tweet'].str.len()
largest = df.loc[df['tweet_length'].idxmax(), 'tweet']
smallest = df.loc[df['tweet_length'].idxmin(), 'tweet']
print(f"\nLargest tweet ({df['tweet_length'].max()} chars):\n{largest}")
print(f"\nSmallest tweet ({df['tweet_length'].min()} chars):\n{smallest}")

from collections import Counter
import re

all_words = re.findall(r'\b\w+\b', ' '.join(df['tweet'].dropna()).lower())
print(f"\nTotal number of words: {len(all_words)}")

stopwords = {
    'i', 'you', 'a', 'to', 'and', 'it', 'is', 'the', 'this', 'of', 'in',
    'my', 'me', 'we', 'he', 'she', 'they', 'was', 'are', 'be', 'been',
    'have', 'has', 'had', 'do', 'did', 'not', 'on', 'at', 'for', 'with',
    'that', 'but', 'or', 'so', 'if', 'an', 'as', 'up', 'out', 'no', 'can',
    'just', 'all', 'what', 'your', 'its', 'by', 'from', 'get', 'got', 'like',
    'will', 'when', 'who', 'how', 'am', 'about', 'their', 'more', 'our', 'than'
}
hashtags = {'ptsdchat'}
contraction_artifacts = {'t', 'm', 's', 'd', 'll', 're', 've'}  # from don't, I'm, he's, I'd, they'll, we're, I've

filtered_words = [
    w for w in all_words
    if w not in stopwords
    and w not in hashtags
    and w not in contraction_artifacts
    and not w.isdigit()
    and len(w) > 1
]

word_counts = Counter(filtered_words)
print(f"\nTop 10 most frequent words (after removing stopwords & hashtags):")
for word, count in word_counts.most_common(10):
    print(f"  {word}: {count}")

print(f"\nTop 10 least frequent words:")
for word, count in word_counts.most_common()[:-11:-1]:
    print(f"  {word}: {count}")

reviewer_cols = [c for c in df.columns if 'reviewer' in c.lower()]
all_labels = pd.Series(df[reviewer_cols].values.ravel()).dropna().unique()
all_labels_sorted = sorted(all_labels)
print(f"\nNumber of unique labels/classes: {len(all_labels_sorted)}")
print("Labels:")
for label in all_labels_sorted:
    print(f"  {label}")

# --- Label Distribution ---
print("\n=== Label Distribution (across all reviewers) ===")
label_counts = pd.Series(df[reviewer_cols].values.ravel()).dropna().value_counts()
for label, count in label_counts.items():
    pct = count / label_counts.sum() * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# --- Missing Values per Reviewer ---
print("\n=== Missing Values per Reviewer ===")
for col in reviewer_cols:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")

# --- Inter-Reviewer Agreement (Fleiss' Kappa) ---
print("\n=== Inter-Reviewer Agreement ===")
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

kappas = []
pairs = list(combinations(reviewer_cols, 2))
for r1, r2 in pairs:
    common = df[[r1, r2]].dropna()
    if len(common) > 1:
        try:
            k = cohen_kappa_score(common[r1], common[r2])
            kappas.append((r1, r2, k, len(common)))
        except:
            pass

kappas_sorted = sorted(kappas, key=lambda x: x[2], reverse=True)
print(f"  Pairwise Cohen's Kappa (top 5 most agreeing pairs):")
for r1, r2, k, n in kappas_sorted[:5]:
    print(f"    {r1} vs {r2}: κ={k:.3f} (n={n})")
print(f"  Pairwise Cohen's Kappa (top 5 least agreeing pairs):")
for r1, r2, k, n in kappas_sorted[-5:]:
    print(f"    {r1} vs {r2}: κ={k:.3f} (n={n})")
avg_kappa = sum(k for _, _, k, _ in kappas) / len(kappas)
print(f"  Average pairwise κ: {avg_kappa:.3f}")
print(f"  Interpretation: <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect")

# --- Tweet Length by Label (using mode label per tweet) ---
print("\n=== Avg Tweet Length by Most Common Label ===")
df['mode_label'] = df[reviewer_cols].mode(axis=1)[0]
length_by_label = df.groupby('mode_label')['tweet_length'].mean().sort_values(ascending=False)
for label, avg_len in length_by_label.items():
    print(f"  {label}: {avg_len:.1f} chars avg")

# --- Reviewer Activity ---
print("\n=== Tweets Labeled per Reviewer ===")
for col in reviewer_cols:
    count = df[col].notna().sum()
    print(f"  {col}: {count} tweets labeled")

# --- Label Agreement Rate per Tweet ---
print("\n=== Tweet-level Agreement ===")
def agreement_rate(row):
    vals = row.dropna()
    if len(vals) == 0:
        return None
    return (vals == vals.mode()[0]).sum() / len(vals)

df['agreement'] = df[reviewer_cols].apply(agreement_rate, axis=1)
print(f"  Avg agreement rate per tweet: {df['agreement'].mean():.2%}")
print(f"  Tweets with full consensus (100%): {(df['agreement'] == 1.0).sum()}")
print(f"  Tweets with majority agreement (>=75%): {(df['agreement'] >= 0.75).sum()}")
print(f"  Tweets with low agreement (<50%): {(df['agreement'] < 0.5).sum()}")

# --- Majority Vote Ground Truth ---
print("\n=== Majority Vote Label Distribution ===")
mv_counts = df['mode_label'].value_counts()
for label, count in mv_counts.items():
    pct = count / len(df) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# --- Reviewer Bias ---
print("\n=== Reviewer Bias (most assigned label per reviewer) ===")
for col in reviewer_cols:
    top = df[col].value_counts()
    print(f"  {col}: most → '{top.index[0]}' ({top.iloc[0]}x) | least → '{top.index[-1]}' ({top.iloc[-1]}x)")

# --- Most Common Words per Majority Label ---
print("\n=== Top 5 Words per Label (majority vote) ===")
for label in df['mode_label'].unique():
    subset = df[df['mode_label'] == label]['tweet'].dropna()
    words = re.findall(r'\b\w+\b', ' '.join(subset).lower())
    words = [w for w in words if w not in stopwords and w not in hashtags
             and w not in contraction_artifacts and not w.isdigit() and len(w) > 1]
    top5 = Counter(words).most_common(5)
    top5_str = ', '.join(f"{w}({c})" for w, c in top5)
    print(f"  {label[:40]}: {top5_str}")

# --- Most Controversial Tweets ---
print("\n=== 3 Most Controversial Tweets (lowest agreement) ===")
controversial = df.nsmallest(3, 'agreement')[['tweet', 'agreement', 'mode_label']]
for _, row in controversial.iterrows():
    print(f"  Agreement: {row['agreement']:.0%} | Mode label: {row['mode_label']}")
    print(f"  Tweet: {row['tweet'][:100]}...")
    print()

# --- Severity Score ---
print("\n=== Crisis Severity Overview ===")
severity_map = {
    '1-No Concern': 1,
    '2- Distressed Self-Disclosure': 2,
    '3-Self-Identified Crisis Symptoms and Seeking Support': 3,
    '4- Self-Identified Crisis Symptoms but Not Seeking Support': 4,
    '5- Acute Crisis with Suicidal Ideation': 5,
    '6- Not Relevant to the Research': 0,
    'Other': 0
}
df['severity'] = df['mode_label'].map(severity_map)
crisis_tweets = df[df['severity'] >= 3]
print(f"  Tweets with any crisis signal (labels 3-5): {len(crisis_tweets)} ({len(crisis_tweets)/len(df)*100:.1f}%)")
print(f"  Acute crisis tweets (label 5): {(df['severity'] == 5).sum()} ({(df['severity']==5).sum()/len(df)*100:.1f}%)")
print(f"  Avg severity score (0-5): {df[df['severity']>0]['severity'].mean():.2f}")
