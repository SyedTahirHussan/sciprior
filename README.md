# sciprior

[![ci](https://github.com/SyedTahirHussan/sciprior/actions/workflows/ci.yml/badge.svg)](https://github.com/SyedTahirHussan/sciprior/actions/workflows/ci.yml)

Generative priors, calibration diagnostics, and inverse-problem tools for scientific
machine learning. The shared core of a six-project AI-for-science research programme.

## Why

Five scientific problems — radio interferometry, weather downscaling, HEP unfolding,
image restoration, and anomaly detection — are the same inverse problem: recover a
signal from incomplete, noisy measurements, **with honest uncertainty**. `sciprior`
implements that shared machinery once, so results across domains are directly comparable.

## Install

```bash
uv add git+https://github.com/SyedTahirHussan/sciprior
```

## Modules

| Module | Contents | Requires torch |
|---|---|---|
| `sciprior.calibration` | Coverage, simulation-based calibration, ECE, conformal prediction | No |
| `sciprior.score` | VP-SDE, denoising score matching, SDE samplers | Yes |
| `sciprior.inverse` | Measurement operators, diffusion posterior sampling | Yes |
| `sciprior.viz` | Shared visual identity for all figures | No |

`sciprior.calibration` has no *import-time* torch dependency: it imports only numpy and
scipy, so it can be imported and used without loading torch, even though installing the
`sciprior` package pulls torch in as a dependency of the other modules.

## Licence

MIT (code). Figures and documentation CC-BY-4.0.
