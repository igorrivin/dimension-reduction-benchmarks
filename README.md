# Dimension Reduction Benchmarks

Experimental code, evidence, and a technical note comparing autoencoders, DiRe,
RAPIDS UMAP, and PCA for intrinsic-dimension estimation and fixed-width dimension
reduction.

This repository is deliberately separate from the implementations being compared. It
is a neutral research workspace, not a production RAPIDS component, and is not endorsed
by NVIDIA. If parts of the autoencoder work mature, they can be proposed independently
for integration into `dire-rapids` or another library without moving the benchmark and
its historical evidence.

## What is here

- `src/dimension_reduction_benchmarks/autoencoder.py`: a reconstruction-error elbow
  estimator for global latent dimension.
- `experiments/`: fixed-width, large-corpus, structure-aware, prepass, precision, and
  Yandex DEEP benchmark harnesses.
- `tests/`: focused CPU tests for the estimator and experiment protocols.
- `docs/`: detailed reports and compact machine-readable result bundles.
- `papers/ae_dire_umap_note.pdf`: the current technical note and its LaTeX source.
- `data/README.md`: dataset sources and the boundary between tracked evidence and large
  local artifacts.

The current evidence is exploratory. No tested method is a universal winner, and the
structure-aware AE and AE-prepass ideas are not ready for production use. The strongest
scaling result is narrower: a small streaming autoencoder showed approximately linear,
bounded-memory behavior through 100 million Yandex vectors, while its strict neighbor
retrieval quality remained weak.

## Installation

The estimator and CPU protocol tests need NumPy, SciPy, and scikit-learn. PyTorch is an
optional dependency:

```bash
python -m pip install -e '.[ae,data,test]'
python -m pytest -q
```

The large GPU experiments were run in RAPIDS 26.06 with CUDA 13. The observed environment
is recorded in `requirements/rapids-26.06-cu13.yml`:

```bash
conda env create -f requirements/rapids-26.06-cu13.yml
conda activate dimension-reduction-benchmarks
python -m pip install -e '.[ae,data,dire,monitoring,topology,viz,test]'
```

`dire-rapids` is consumed through its exported Python API. A sibling source checkout is
supported by historical reproduction commands, but it is not a library dependency of
this repository.

## Autoencoder dimension estimate

```python
from dimension_reduction_benchmarks import autoencoder_intrinsic_dimension

result = autoencoder_intrinsic_dimension(X, max_dimension=16, repeats=3)
print(result.dimension)
print(result.reconstruction_errors)
```

The result is an architecture- and optimization-dependent global reconstruction
diagnostic. It is not interchangeable with local estimators such as TwoNN or
Levina--Bickel MLE.

The optional dimension-estimator comparison example uses TwoNN and MLE from
`dire-evoc`; install it with `python -m pip install -e '.[ae,comparison]'`.

## Reports and evidence

- [Fixed-dimension comparison](docs/fixed-dimension-comparison.rst)
- [Large arXiv and Wikipedia comparison](docs/large-embedding-comparison.rst)
- [Structure-aware AE and prepass pilot](docs/results/structure-aware-ae-prepass-pilot/summary.md)
- [Yandex DEEP scaling report](docs/results/yandex-deep1b-scaling/summary.md)
- [Technical note](papers/ae_dire_umap_note.pdf)

Tracked JSON and CSV files are compact evidence summaries. Multi-gigabyte source data,
prepared matrices, layouts, model checkpoints, and caches are intentionally excluded.
Historical provenance retains the absolute paths used during the original runs; those
paths are records, not portable download links.

## License

Apache License 2.0. This choice is compatible with a possible future contribution of
selected implementation code to `dire-rapids`; the benchmark repository itself remains
independent.
