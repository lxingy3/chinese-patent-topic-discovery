# Chinese Patent Topic Discovery

Topic discovery experiments for Chinese patent titles and abstracts.

This project is about unsupervised patent understanding: finding themes, inspecting topic quality, and comparing simple NMF topics with CETopic-style subtopics.

## What this repo covers

- Public patent title and abstract text
- NMF topic assignments and topic summaries
- CETopic-style subtopic assignments
- Manually grouped topic summaries for high-level inspection
- A small runnable NMF baseline

## Dataset

Main text file:

```text
data/processed/patent_texts.csv
```

Exported topic files:

```text
data/processed/nmf_8_topic_assignments.csv
data/processed/nmf_8_topic_summary.csv
data/processed/cetopic_subtopic_assignments.csv
data/processed/cetopic_manual_category_summary.csv
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/validate_release.py
python experiments/nmf_topics.py --topics 8
```

The default run uses a deterministic sample for speed. Use `--max-samples 0 --max-features 50000` for a larger run.

## Notes

The topic labels are exploratory outputs. They are useful for browsing patent themes and comparing clustering behavior, but they should not be treated as ground-truth class labels.

## License

Code is MIT licensed. The patent text is compiled from public patent metadata; see `DATA_NOTICE.md` before redistribution or commercial use.
