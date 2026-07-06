from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, default=ROOT / "data" / "processed" / "nmf_8_topic_assignments.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "evaluation" / "ipc_level_1_labels.csv")
    args = parser.parse_args()

    topics = pd.read_csv(args.assignments).fillna("")
    labels = pd.read_csv(args.labels).fillna("")
    merged = topics.merge(labels, on=["patent_title", "patent_abstract"], how="inner")
    if merged.empty:
        raise SystemExit("No matching rows between topic assignments and IPC labels")

    topic_to_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in merged.itertuples(index=False):
        topic_to_labels[str(row.topic_id)][str(row.ipc_level_1)] += 1

    rows = []
    for topic_id, counts in sorted(topic_to_labels.items(), key=lambda item: int(item[0])):
        total = sum(counts.values())
        label, count = counts.most_common(1)[0]
        rows.append(
            {
                "topic_id": topic_id,
                "samples": total,
                "top_ipc_level_1": label,
                "top_ipc_share": count / total,
                "label_counts": dict(counts),
            }
        )

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

