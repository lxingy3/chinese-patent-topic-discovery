from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from nmf_topics import patent_ngrams


ROOT = Path(__file__).resolve().parents[1]


def metrics(matrix, labels: list[int]) -> dict[str, float]:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    return {
        "silhouette": float(silhouette_score(dense, labels, sample_size=min(1000, len(labels)), random_state=42)),
        "calinski_harabasz": float(calinski_harabasz_score(dense, labels)),
        "davies_bouldin": float(davies_bouldin_score(dense, labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_texts.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results" / "parameter_sweep_smoke.json")
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--topics", default="8,12,16")
    parser.add_argument("--max-features", type=int, default=12000)
    args = parser.parse_args()

    df = pd.read_csv(args.data).fillna("")
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42)
    text = df["patent_title"].astype(str) + "\n" + df["patent_abstract"].astype(str)

    vectorizer = TfidfVectorizer(analyzer=patent_ngrams, min_df=2, max_features=args.max_features)
    x = vectorizer.fit_transform(text)
    rows = []
    for topic_count in [int(value) for value in args.topics.split(",") if value.strip()]:
        nmf = NMF(n_components=topic_count, init="nndsvda", random_state=42, max_iter=300)
        nmf_labels = nmf.fit_transform(x).argmax(axis=1)
        rows.append({"model": "NMF", "topics": topic_count, **metrics(x, nmf_labels)})

        svd = TruncatedSVD(n_components=min(50, x.shape[1] - 1), random_state=42).fit_transform(x)
        kmeans_labels = KMeans(n_clusters=topic_count, random_state=42, n_init="auto").fit_predict(svd)
        rows.append({"model": "SVD+KMeans", "topics": topic_count, **metrics(svd, kmeans_labels)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

