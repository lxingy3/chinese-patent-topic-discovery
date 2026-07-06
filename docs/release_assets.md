# Release Assets

Large experiment assets are published through GitHub Releases so the repository stays easy to clone while the full topic-discovery bundle remains available.

## Asset Layout

The `v0.2.0` release contains three downloadable bundles:

- `patent-topic-data-v0.2.0.zip`: cleaned patent texts, released topic assignments, IPC labels used for external checks, and field schema.
- `patent-topic-results-v0.2.0.zip`: parameter sweeps, topic quality metrics, IPC-based evaluation outputs, and supporting result tables.
- `patent-topic-artifacts-v0.2.0.zip`: sentence embedding cache, TopicX checkpoint artifact, and word-cloud grids.

## When to Use Each Bundle

Use the data bundle if you want to inspect the patent corpus or released topic assignments.

Use the results bundle if you want to audit parameter choices, topic metrics, or the IPC-based external evaluation.

Use the artifact bundle if you want to reuse the released embedding cache or visual summaries.

The scripts in `experiments/` are kept in the repository because they are small and useful for review. The release assets keep larger derived files available without turning every clone into a full artifact download.
