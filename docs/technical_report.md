# Technical Report

This project studies unsupervised topic discovery for Chinese patent titles and abstracts. It releases cleaned patent text, NMF topic outputs, CETopic-style semantic topic outputs, parameter sweeps, IPC-based external evaluation tables, sentence embeddings, and visualization artifacts.

## Data Source

The source data is public patent metadata used for patent-domain analysis. Each record contains a patent title and abstract. The topic modeling experiments use the text fields, while IPC labels are kept only for external evaluation of the discovered topics.

The released text table contains 9,867 cleaned patent records. The IPC-based evaluation tables use the available public IPC labels to test whether unsupervised topics align with coarse technical areas. The IPC labels are not used as training targets for topic discovery.

## Preprocessing

All models start from a common text preparation stage.

The title and abstract are combined into one patent text field. For sparse topic models, the title is weighted more heavily during feature extraction because it often contains the clearest technical object. For sentence-embedding models, the full title and abstract are encoded without repeating the title, so the semantic vector does not become title-only.

The preprocessing removes several types of weak topic words:

- general Chinese stopwords
- patent boilerplate such as invention, disclosure, method, device, system, and module
- broad structural words such as component, mechanism, unit, and assembly
- very rare terms that appear too few times to define a stable topic
- overly common terms that appear across many unrelated patents

The goal is not to remove every common patent word. It is to keep the topic-word lists from being dominated by generic patent language.

## Model Principles

### TF-IDF + KMeans

KMeans is the simplest clustering baseline. It converts each patent into a TF-IDF vector and groups nearby vectors into clusters. It is fast and easy to inspect, but it tends to produce broad mixed clusters when the vocabulary contains many generic engineering words.

### TF-IDF + NMF

NMF factorizes the document-term matrix into document-topic and topic-term matrices. Compared with KMeans, it usually gives cleaner topic words because each topic is represented by high-weight terms. It is a strong interpretable baseline for short patent text, especially when the goal is to name broad technical themes.

### Sentence-BERT + CETopic-Style Clustering

The semantic topic route uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to encode each patent into a dense vector. CETopic-style clustering then groups patents in semantic space and extracts topic words from the clustered documents.

This helps with cases where different patents describe similar ideas with different surface words. For example, visual inspection, image recognition, and target detection may be semantically close even when their exact tokens differ.

The released CETopic-style output keeps 30 subtopics and then provides a manual macro-category summary. The manual categories are a readable organization layer over the discovered subtopics; they are not supervised training labels.

## Code Design

The repository keeps the runnable scripts small and focused.

- `experiments/nmf_topics.py` runs a lightweight NMF topic baseline.
- `experiments/parameter_sweep.py` compares KMeans, NMF, and CETopic-style settings across topic counts and feature choices.
- `experiments/ipc_topic_eval.py` maps discovered topics to IPC level-1 labels for external evaluation.
- `scripts/validate_release.py` checks the expected data, artifacts, and result files.

The heavier semantic artifacts are included so readers can inspect the actual experiment outputs:

- sentence embedding cache under `artifacts/embeddings/`
- TopicX checkpoint artifact under `artifacts/checkpoints/`
- topic assignment tables under `data/processed/`
- word-cloud grids under `artifacts/wordclouds/`

## Result Analysis

The sparse baselines and semantic topic models behave differently.

KMeans improves as the cluster count grows, but it still tends to form mixed engineering clusters. In the released parameter comparison, KMeans with 30 clusters reduces the largest-topic ratio to 0.1760, but the top-word lists repeat many terms across topics.

NMF is more interpretable. In one parameter sweep, NMF with 10 topics has a high silhouette score of 0.2922, a low Davies-Bouldin score of 0.9370, and only 2 repeated words among the top-5 topic terms. It is the best lightweight choice when the goal is broad, readable topic naming.

CETopic-style semantic clustering is stronger for fine-grained discovery. With 30 topics and 2-dimensional reduction, the released comparison reaches a silhouette score of 0.4463, a Calinski-Harabasz score of 22064.02, and a Davies-Bouldin score of 0.6915. The largest topic ratio drops to 0.0501, which means the model avoids collapsing many patents into one overly broad cluster.

The final 30-subtopic output is summarized into macro groups such as production line, information systems and data, electrical energy, bio-ecology and environment, agriculture and food, material preparation, and optics. The largest macro group is production line, with 3,936 patents, while information systems and data contains 2,614 patents. This matches the shape of the corpus: it contains many engineering, manufacturing, energy, and information-system patents.

The IPC-based external evaluation gives an additional check. On the test split, the topic-to-IPC mapping reaches 0.5578 micro-F1, compared with 0.3241 for the majority baseline. This does not turn the unsupervised model into a classifier, but it shows that the discovered topics are not arbitrary clusters; they preserve useful technical-area structure.

## Limitations

Topic names are interpretive labels, not ground truth. The model discovers clusters, and humans still need to decide whether a cluster name is useful.

Patent abstracts contain repeated formulaic language. Even after stopword filtering, generic engineering terms can still appear in multiple topics.

The best number of topics depends on the use case. Fewer topics are easier to read, while 30 subtopics preserve more technical detail. The project keeps both broad NMF outputs and fine-grained CETopic-style outputs because they answer different analysis questions.

Sentence-BERT embeddings are multilingual and general-purpose. They work well enough for this corpus, but a patent-domain Chinese encoder may improve semantic clustering.

The IPC evaluation is only an external proxy. IPC labels are assigned for legal and technical classification, while topic clusters are intended for exploratory analysis. Agreement with IPC is useful, but it is not the only measure of topic quality.

## Future Work

Useful extensions include:

- patent-domain sentence embeddings
- BERTopic and contextualized topic model comparisons
- automatic topic-count selection instead of fixed sweeps
- human evaluation of topic names and representative patents
- hierarchical topic grouping from subtopics to macro themes
- better handling of low-support subtopics
- publishing a Hugging Face dataset card for the cleaned public patent metadata

The current release is designed as an inspectable topic-discovery benchmark rather than a finished industrial taxonomy.
