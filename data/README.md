# Data and artifact policy

Large datasets and generated layouts are not stored in Git. The repository tracks
compact CSV/JSON summaries, input manifests, hashes, plots, and the paper.

## Sources

- Yandex DEEP billion-scale vectors: the downloader in
  `experiments/cache_yandex_deep1b.py` records remote identity and supports verified,
  resumable prefixes. Dataset information and attribution are at
  <https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search>.
- Wikipedia: `experiments/cache_wikipedia_embeddings.py` prepares a revision-pinned
  prefix of `CohereLabs/wikipedia-2023-11-embed-multilingual-v3` and records shard
  hashes.
- arXiv: the large-corpus reports use 723,457 mean-pooled BGE-small paper embeddings.
  Their prepared-array and source-index hashes are recorded under
  `docs/results/large-arxiv-*`.

Use `artifacts/` for downloads, prepared matrices, layouts, and checkpoints. That
directory and common array/model formats are ignored by Git.

Historical result files contain absolute paths from the machine that produced each
run. These fields are retained unchanged because they are part of the signed-off
provenance record. Reproduction should use the dataset identifiers, hashes, manifests,
and current command-line output paths rather than expecting the old paths to exist.
