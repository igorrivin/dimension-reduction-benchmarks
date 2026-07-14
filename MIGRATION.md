# Repository extraction

This project was separated from an unpublished experimental branch of `dire-evoc` so
that cross-method research would not live inside one of the products under comparison.

The first commit preserves the already-validated structure-aware AE, prepass, and Yandex
result bundle with its original authorship and timestamp. The next extraction commit
adds the earlier estimator, fixed/large-corpus experiments, paper, and compact evidence.
Large source data and layouts were deliberately excluded.

Historical result files and figure manifests were not rewritten: their hashes and local
paths describe the original executions. Live code and documentation use this standalone
repository's package name, while `dire-rapids`, cuML UMAP, and `dire-evoc` local-dimension
estimators remain external comparison dependencies.
