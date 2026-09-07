# LLM-Data-Annotation

Human and LLM annotation of veteran tweets for PTSD / crisis severity, with inter-rater reliability analysis and an LLM labeling sweep pipeline.

## What’s here

| Path | Description |
|------|-------------|
| `data_preparation.ipynb` | Data loading and preparation |
| `data_explore.ipynb` | EDA and baseline IRR / agreement analysis |
| `label_consolidation.ipynb` | Label remapping (L1–L7), consolidations, and IRR for reviewers and LLMs |
| `llm_sweep_pipeline/` | Code to run and score LLM annotation sweeps ([details](llm_sweep_pipeline/README.md)) |
| `Veteran_Tweet_Label_PTSD_*.csv` | Ground-truth + LLM predictions (Prompt G) and consolidated label variants |

## Labels

| Code | Meaning |
|------|---------|
| L1 | No Concern |
| L2 | Distressed Self-Disclosure |
| L3 | Self-Identified Crisis Symptoms and Seeking Support |
| L4 | Self-Identified Crisis Symptoms but Not Seeking Support |
| L5 | Acute Crisis with Suicidal Ideation |
| L6 | Not Relevant to the Research |
| L7 | Other |

Some analyses consolidate L6/L7 with L1 (or L7 into L6). See `label_consolidation.ipynb`.

## LLMs in the merged tables

- **Open-source:** Llama3.1-8B, Hermes3-8B, Qwen2.5-7B  
- **Hosted:** GPT-5.4-mini, GPT-5.6-Terra, Claude-Opus-5  

## License

Apache License 2.0 — see [LICENSE](LICENSE).
