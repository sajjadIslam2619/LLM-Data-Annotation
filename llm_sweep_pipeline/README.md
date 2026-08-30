# llm_sweep_pipeline

Code for the LLM-vs-human crisis-severity labeling sweep (veteran tweets, PTSD/crisis label taxonomy), pushed as part of `sajjadIslam2619/LLM-Data-Annotation`.

## Structure

- **`pipeline/`** — the core, reusable pipeline: prompt templates (`prompts.py`), label normalization (`labels.py`), scoring (`score_utils.py`, `compute_micro_f1.py`, `compute_precision_recall.py`), and the sweep runners (`run_full_sweep.py`, `run_new_experiments.py`, `run_remaining_table5.py`, `run_*_ablation.py`, `run_ollama_compare.py`), plus agreement/reliability checks and misc utilities.
- **`analysis/`** — one-off aggregation/rebuild scripts (prefixed `_`) used to compute summary tables and metrics from already-saved per-tweet result files, without re-running any model calls.
- **`shell/`** — overnight/background sweep launch scripts.
- **`handoff_G_remaining/`** — the self-contained, dependency-minimal package sent to a collaborator to run the remaining `G_official_reasoning_format` Ollama models (no API keys needed).

## Not included (yet)

Per instructions, this push is **code only** — no data, images, or docs:

- All `*_results.csv` / `*_summary.csv` per-run output files, and the ground-truth labeled dataset.
- `mlflow.db`, `mlruns/` (MLflow tracking data).
- `.docx` / `.pptx` / `.pdf` deliverables, `.png` plots, `tweet_embeddings_umap.html`.
- Project markdown docs (`HANDOFF.md`, `RESEARCH_LOG.md`, `EXPERIMENT_TRACKER.md`, `AGREEMENT_SUMMARY.md`, `GUIDE.md`) and `.tex` manuscript source.
- The two EDA notebooks (`ptsd_tweet_eda.ipynb`, `ptsd_tweet_eda_executed.ipynb`) — left out for now since the executed one has plots/output baked in.

Pushed 2026-08-30.
