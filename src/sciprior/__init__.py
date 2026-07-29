"""Generative priors, calibration diagnostics, and inverse-problem tools.

`sciprior` is the shared core of an AI-for-science research programme. The
`calibration` subpackage has no *import-time* torch dependency, so it can be
imported and used without loading torch, even though installing the `sciprior`
package pulls torch in for the other modules.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
