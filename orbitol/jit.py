"""Optional Numba compilation for small numerical kernels.

``njit`` compiles with ``numba.njit(cache=True, nogil=True)`` when Numba is importable (a pinned dependency) and
``ORBITOL_NUMBA`` is not ``"0"``. Otherwise it returns the function unchanged, so the same kernels run as plain
Python: slower, with the same results. Tests check both modes where Numba exists.
Compiled kernels release the GIL so CPU decoding can overlap GPU sampling without changing numerics.
Set ``NUMBA_CACHE_DIR`` to a shared directory so jobs reuse compiled kernels.
"""
import os
import warnings

try:
    import numba
except ImportError:
    numba = None

ENABLED = numba is not None and os.environ.get("ORBITOL_NUMBA", "1") != "0"
if numba is None and os.environ.get("ORBITOL_NUMBA", "1") != "0":
    warnings.warn("Numba is not installed: orbitol kernels (fast decode) run as slow plain Python. Use the Orbitol "
                  "environment ($ORBITOL_PYTHON), or set ORBITOL_NUMBA=0 to silence this.", RuntimeWarning, stacklevel=2)


def njit(function):
    return numba.njit(cache=True, nogil=True)(function) if ENABLED else function
