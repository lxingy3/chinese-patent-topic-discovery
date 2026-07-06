from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        "data/processed/patent_texts.csv",
        "data/processed/nmf_8_topic_assignments.csv",
        "data/processed/nmf_8_topic_summary.csv",
        "data/processed/cetopic_subtopic_assignments.csv",
        "data/processed/cetopic_manual_category_summary.csv",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")

    texts = pd.read_csv(ROOT / "data" / "processed" / "patent_texts.csv")
    if not {"patent_title", "patent_abstract"}.issubset(texts.columns):
        raise SystemExit("Text file is missing required columns")
    if texts.duplicated(["patent_title", "patent_abstract"]).any():
        raise SystemExit("Duplicate title and abstract pairs found")
    if len(texts) < 1000:
        raise SystemExit("Dataset is unexpectedly small")

    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".csv", ".cff"}:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            for marker in ["C:\\Users", "Users\\\\33672", "\u4efb\u52a14", "\u674e\u7fd4\u5b87"]:
                if marker in text:
                    raise SystemExit(f"Forbidden marker {marker!r} found in {path}")

    print(f"Validated {len(texts)} patent text rows.")


if __name__ == "__main__":
    main()
