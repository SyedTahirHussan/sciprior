from __future__ import annotations

import subprocess
import sys


def test_importing_calibration_does_not_load_torch() -> None:
    """`sciprior.calibration` must be importable without pulling torch into memory.

    This is the property documented in `README.md`, `docs/index.md`, and both
    package docstrings: `calibration` has no *import-time* torch dependency, so it
    can be used for meta-analysis without loading a deep-learning stack, even
    though installing `sciprior` pulls torch in for `score`/`inverse`/`viz`.

    The check must run in a fresh interpreter. An in-process assertion would be
    worthless here: by the time this test suite is running, `test_dps.py` and
    friends have already imported torch into `sys.modules`, so any in-process
    check of `"torch" not in sys.modules` would trivially fail regardless of
    whether `sciprior.calibration` itself imports it. A subprocess is the only way
    to observe what `import sciprior.calibration` alone pulls in.

    This test is the *only* thing enforcing the boundary: nothing else in CI would
    notice a contributor adding a torch-using helper under `calibration/`, or a
    convenience re-export in `sciprior/__init__.py`, that quietly breaks it.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sciprior.calibration, sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing sciprior.calibration pulled in torch (stdout={result.stdout!r}, "
        f"stderr={result.stderr!r})"
    )
