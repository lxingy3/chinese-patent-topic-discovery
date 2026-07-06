# Model Card

## Task

Discover themes from Chinese patent title and abstract text without supervised labels.

## Baseline

The included baseline uses character n-gram TF-IDF features and non-negative matrix factorization.

```bash
python experiments/nmf_topics.py --topics 8
```

## Exported topic outputs

The repository includes previous NMF and CETopic-style assignments. These files are experiment outputs, not ground-truth labels.

## Limits

Short patent abstracts can produce broad or mixed topics. Topic names should be inspected before downstream use.

