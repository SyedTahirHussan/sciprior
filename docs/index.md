# sciprior

Generative priors, calibration diagnostics, and inverse-problem tools for scientific
machine learning.

## The idea

Five scientific problems are the same inverse problem in different costumes:

| Domain | Forward operator | Latent signal |
|---|---|---|
| Radio interferometry | Sparse Fourier sampling | Sky brightness |
| Weather downscaling | Spatial coarsening | Fine-scale state |
| HEP unfolding | Detector response | Particle-level truth |
| Image restoration | Degradation | Original image |
| Anomaly detection | — | Density under a prior |

Each recovers a signal from incomplete, noisy measurements — and each needs *honest
uncertainty*, not just a point estimate. `sciprior` implements that shared machinery
once so results across domains are directly comparable.

## Install

```bash
uv add git+https://github.com/SyedTahirHussan/sciprior
```

## Design notes

`sciprior.calibration` has no torch dependency. It can be installed standalone for
meta-analysis of results produced elsewhere.

Every `MeasurementOperator` must pass `dot_product_test`, which verifies the adjoint
identity `<Ax, y> == <x, A*y>`. A wrong adjoint is the most common bug in
inverse-problem code and produces reconstructions that look plausible while being
incorrect.
