# Chinese Patent Topic Discovery

Unsupervised topic discovery experiments for Chinese patent titles and abstracts.

This repository releases the cleaned patent text, NMF topic outputs, CETopic-style subtopic outputs, parameter sweep tables, an IPC-based external evaluation, sentence-embedding cache, and word-cloud artifacts.

## Contents

```text
data/processed/                         Patent text and released topic assignments
data/evaluation/                        IPC labels used for external topic checks
experiments/                            Runnable topic-modeling entrypoints
experiments/results/model_comparison/   Parameter sweeps and evaluation tables
artifacts/embeddings/                   Released sentence embedding cache
artifacts/checkpoints/                  Small TopicX checkpoint artifact
artifacts/wordclouds/                   Topic visualization grids
```

## Released Experiments

The release includes:

- NMF topic assignments and topic summaries
- CETopic-style subtopic assignments
- Parameter sweeps over topic counts and feature settings
- Topic quality metrics: TU, NPMI, and C_V
- Multi-label IPC level-1 evaluation for unsupervised topic-to-label mapping
- Sentence-transformer embedding cache for the patent corpus

The goal is to share the actual experiment assets, not to force every reader to rebuild the entire pipeline.

## Quick Start

Install lightweight dependencies:

```bash
pip install -r requirements.txt
```

Validate the release:

```bash
python scripts/validate_release.py
```

Run a small NMF demo:

```bash
python experiments/nmf_topics.py --topics 8
```

Run a small parameter sweep:

```bash
python experiments/parameter_sweep.py --max-samples 1000
```

The sentence-transformer pipeline is optional:

```bash
pip install -r requirements-embeddings.txt
```

## Important Files

```text
data/processed/nmf_8_topic_assignments.csv
data/processed/cetopic_subtopic_assignments.csv
experiments/results/model_comparison/parameter_sweep_summary.csv
experiments/results/model_comparison/multi_label_ipc_eval_summary.csv
artifacts/embeddings/patent_embeddings_paraphrase_multilingual_minilm_l12_v2.npy
```

## Data Notice

The dataset contains public patent metadata and derived topic outputs. Topic names and groups are exploratory labels, not ground-truth classes. See `DATA_NOTICE.md` before redistribution or commercial use.

