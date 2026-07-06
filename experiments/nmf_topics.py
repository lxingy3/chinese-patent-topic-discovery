from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]


BOILERPLATE = [
    "本发明",
    "本实用新型",
    "所述",
    "一种",
    "包括",
    "公开",
    "涉及",
    "提供",
    "设置",
    "固定连接",
    "连接有",
    "安装有",
    "技术领域",
]

STOP_TERMS = {
    "第一",
    "第二",
    "第三",
    "装置",
    "方法",
    "系统",
    "进行",
    "通过",
    "用于",
    "具有",
    "实现",
    "连接",
    "固定",
    "内部",
    "设有",
    "机构",
    "的内",
    "动组",
    "动机构",
    "动组件",
    "测装",
    "测装置",
    "检测装",
    "检测装置",
    "制模",
    "制模块",
    "控制模",
}


def normalize_text(value: str) -> str:
    for phrase in BOILERPLATE:
        value = value.replace(phrase, " ")
    return value


def patent_ngrams(value: str) -> list[str]:
    grams: list[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", normalize_text(value)):
        for n in (2, 3, 4):
            grams.extend(chunk[i : i + n] for i in range(0, max(len(chunk) - n + 1, 0)))
    return [term for term in grams if term not in STOP_TERMS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "patent_texts.csv")
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=12_000)
    parser.add_argument("--max-iter", type=int, default=400)
    args = parser.parse_args()

    df = pd.read_csv(args.data).fillna("")
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42)
    text = df["patent_title"].astype(str) + "\n" + df["patent_abstract"].astype(str)

    vectorizer = TfidfVectorizer(analyzer=patent_ngrams, min_df=2, max_features=args.max_features)
    matrix = vectorizer.fit_transform(text)

    model = NMF(n_components=args.topics, random_state=42, init="nndsvda", max_iter=args.max_iter)
    weights = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()

    print(f"samples={len(df)} topics={args.topics}")
    for topic_id, row in enumerate(model.components_):
        top = row.argsort()[-args.top_terms :][::-1]
        print(f"topic_{topic_id}: " + ", ".join(terms[i] for i in top))

    assigned = weights.argmax(axis=1)
    counts = pd.Series(assigned).value_counts().sort_index()
    print("topic_counts=" + ", ".join(f"{idx}:{count}" for idx, count in counts.items()))


if __name__ == "__main__":
    main()
