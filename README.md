# Chinese Patent Topic Discovery

Unsupervised topic discovery experiments for Chinese patent titles and abstracts.

This repository releases the cleaned patent text, NMF topic outputs, CETopic-style subtopic outputs, parameter sweep tables, an IPC-based external evaluation, sentence-embedding cache, and word-cloud artifacts.

## What This Project Shows

The project covers an end-to-end unsupervised patent topic-discovery workflow:

- cleaning public patent titles and abstracts for topic modeling
- comparing TF-IDF + KMeans, TF-IDF + NMF, and Sentence-BERT + CETopic-style clustering
- running parameter sweeps over topic counts and dimensionality settings
- evaluating discovered topics with topic quality metrics and IPC-based external checks
- releasing topic assignments, embedding artifacts, and word-cloud summaries

The full methodology and result discussion are in [`docs/technical_report.md`](docs/technical_report.md).
Large downloadable artifacts are described in [`docs/release_assets.md`](docs/release_assets.md).

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

The release includes the experiment assets needed to inspect the results without rebuilding the entire pipeline.

## Method Summary

The released text table contains 9,867 cleaned public patent records. Topic models use titles and abstracts only. IPC labels are not used to train the topic models; they are used later as an external check on whether discovered topics align with coarse technical areas.

NMF is the strongest lightweight interpretable baseline for broad topic naming. Sentence-BERT + CETopic-style clustering is better for fine-grained subtopic discovery, especially when similar patents use different surface words.

The final semantic output keeps 30 discovered subtopics and provides a readable macro-category summary. On the IPC-based external test split, the topic-to-IPC mapping reaches `0.5578` micro-F1, compared with `0.3241` for the majority baseline.

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
