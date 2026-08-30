"""
Post-run quality gate. Run this after any OpenAI sweep finishes.

Answers three questions, in order of how much they matter:
  1. Did the runs actually complete, or did some model fail silently?
  2. Does any OpenAI model on Variant F beat qwen3:8b's 0.715?
     (This is the fair comparison the project has never been able to make.)
  3. Is the Phase 4 A-E ablation restored and does it still support
     the "D over E, macro F1 over accuracy" decision?

No model calls, no API key needed. Reads only local files.

Usage:
  python check_new_runs.py
"""

import os
import sqlite3

import pandas as pd

BEST_LOCAL = 0.715          # qwen3:8b under Variant F -- the number to beat
BEST_LOCAL_NAME = "qwen3:8b"
HUMAN_KAPPA = 0.156


def rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def check_f_sweep():
    rule("1. VARIANT F ON OPENAI  --  the comparison that was blocked")
    f = "full_sweep_F_refined_definitions_summary.csv"
    if not os.path.exists(f):
        print("  MISSING: the F sweep did not write a summary. Did it error?")
        return
    d = pd.read_csv(f)
    oa = d[d.backend == "openai"] if "backend" in d.columns else d
    if not len(oa):
        print("  No OpenAI rows in the F summary -- the sweep ran but skipped OpenAI.")
        return

    print(f"  {'model':<20}{'acc':>8}{'macro F1':>10}{'unparse':>9}   verdict")
    print("  " + "-" * 62)
    for _, r in oa.sort_values("macro_f1", ascending=False).iterrows():
        unp = int(r.get("unparseable", 0) or 0)
        if unp > 14:
            v = f"UNRELIABLE ({unp}/149 unparseable)"
        elif r.macro_f1 > BEST_LOCAL:
            v = f"BEATS {BEST_LOCAL_NAME} (+{r.macro_f1 - BEST_LOCAL:.3f})"
        else:
            v = f"below {BEST_LOCAL_NAME} ({r.macro_f1 - BEST_LOCAL:+.3f})"
        print(f"  {str(r.model):<20}{r.accuracy:>8.3f}{r.macro_f1:>10.3f}{unp:>9}   {v}")

    clean = oa[oa.get("unparseable", 0).fillna(0) <= 14] if "unparseable" in oa else oa
    winners = clean[clean.macro_f1 > BEST_LOCAL]
    print()
    if len(winners):
        print(f"  => An OpenAI model beats the best local result on the SAME prompt.")
        print(f"     Slide 6's \"best in project\" claim needs updating before the meeting.")
    else:
        print(f"  => No OpenAI model beat {BEST_LOCAL_NAME} ({BEST_LOCAL}) on Variant F.")
        print(f"     This is now a FAIR comparison -- same prompt, same 140 tweets.")
        print(f"     The 'local matches frontier models' claim is defensible. Say it.")


def check_ablation():
    rule("2. PHASE 4 ABLATION  --  restored?")
    f = "openai_ablation_summary.csv"
    if not os.path.exists(f):
        print("  MISSING: openai_ablation_summary.csv not written.")
        return
    d = pd.read_csv(f)
    print(d.to_string(index=False))
    if "variant" in d.columns and "macro_f1" in d.columns:
        try:
            dd = d.set_index("variant")
            D = dd.loc[[i for i in dd.index if str(i).startswith("D")][0], "macro_f1"]
            E = dd.loc[[i for i in dd.index if str(i).startswith("E")][0], "macro_f1"]
            print()
            if D > E:
                print(f"  => D ({D:.3f}) still beats E ({E:.3f}) on macro F1.")
                print("     The metric-decision argument on slide 3 holds, now with live data.")
            else:
                print(f"  => WARNING: E ({E:.3f}) >= D ({D:.3f}) this time.")
                print("     Slide 3's argument does NOT reproduce. Check per-class F1 on")
                print("     Acute Crisis before presenting that slide.")
        except (IndexError, KeyError):
            pass


def check_agreement():
    rule("3. ANNOTATOR AGREEMENT  --  did the new runs change the headline?")
    f = "agreement_summary_all.csv"
    if not os.path.exists(f):
        print("  MISSING: run run_all_agreement_checks.py first.")
        return
    d = pd.read_csv(f)
    k = "llm_human_mean_kappa"
    n_over = (d[k] > HUMAN_KAPPA).sum()
    print(f"  {n_over} of {len(d)} runs exceed the {HUMAN_KAPPA} human-human baseline.")
    print(f"  (Slide 9 currently says 47 of 87 -- update if this differs.)")
    top = d.nlargest(5, k)
    print("\n  Top 5 by LLM-vs-human kappa:")
    for _, r in top.iterrows():
        print(f"    {r.file[:52]:<54}{r[k]:.3f}")


def check_mlflow():
    rule("4. MLFLOW  --  is everything logged and recoverable?")
    if not os.path.exists("mlflow.db"):
        print("  No mlflow.db found.")
        return
    c = sqlite3.connect("mlflow.db")
    n = c.execute("select count(*) from runs").fetchone()[0]
    print(f"  {n} runs logged. (Was 176 before this session.)")
    if n <= 176:
        print("  WARNING: no new runs logged. Did you forget --use-mlflow?")
        print("  Without it, an overwritten summary CSV is unrecoverable -- this is")
        print("  exactly how the Phase 4 and Phase 11 data were lost.")
    else:
        print(f"  {n - 176} new runs logged. The results survive a filename collision.")


if __name__ == "__main__":
    check_f_sweep()
    check_ablation()
    check_agreement()
    check_mlflow()
    rule("NEXT")
    print("  - If anything above says UNRELIABLE or WARNING, fix it before the deck.")
    print("  - Otherwise: update slide 6 / slide 10 with the new comparison, and")
    print("    add a Phase 19 entry to RESEARCH_LOG.md.")
    print()
