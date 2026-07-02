r"""Computational backend helper for the entire library.

This module acts as a dynamic proxy to either :mod:`numpy` (CPU) or :mod:`cupy` (GPU).
By importing this module as a namespace, arrays handling is hardware-agnostic.

**Convention:**
Always import this module itself as the array namespace. Do *not* import individual
functions from it directly.

.. code-block:: python

    # CORRECT:
    import cards.backend as xp

    # WRONG:
    from cards.backend import zeros

Note
----
Type checking is deliberately relaxed via :data:`typing.Any` to allow both CPU arrays
(:class:`numpy.ndarray`) and GPU arrays (:class:`cupy.ndarray`) to pass static linter
analysis without throwing class mismatches.

Examples
--------
>>> import cards.backend as xp
>>> def foo(x: xp.ndarray) -> xp.ndarray:
...     return xp.minimum(x.cumsum(), 4)
...
>>> xp.set_backend("cupy")
>>> x = xp.array([2, 3, 4])     # cupy.ndarray
>>> foo(x)                      # cupy.ndarray as well
array([2, 4, 4])
>>>
>>> xp.set_backend("numpy")
>>> x = xp.array([2, 3, 4])     # numpy.ndarray
>>> foo(x)                      # numpy.ndarray as well
array([2, 4, 4])
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    ndarray = Any
    r"""Type alias accepting any array-like object during static checking."""

    def set_backend(new_backend: str) -> None: ...
    def __getattr__(name: str) -> Any: ...


_xp = importlib.import_module("numpy")


def get_backend() -> str:
    r"""Return the name of the currently active backend ('numpy' or 'cupy')."""
    return _xp.__name__


def set_backend(new_backend: str) -> None:
    r"""Globally switch the backend engine for the entire library.

    Parameters
    ----------
    new_backend : {'numpy', 'cupy'}
        The target compute engine. Use ``'numpy'`` for CPU processing or ``'cupy'`` for
        CUDA GPU execution.

    Raises
    ------
    ValueError
        If the requested backend is not supported.
    ModuleNotFoundError
        If ``'cupy'`` is requested on a system without CuPy installed.

    Examples
    --------
    >>> import cards.backend as xp
    >>> x = xp.zeros((10, 10)) # numpy.ndarray by default
    >>> type(x)
    <class 'numpy.ndarray'>
    >>> xp.set_backend("cupy")
    >>> x = xp.zeros((10, 10))
    >>> type(x)
    <class 'cupy.ndarray'>
    """
    global _xp
    if new_backend in ("numpy", "cupy"):
        _xp = importlib.import_module(new_backend)
    else:
        raise ValueError(
            f"Unsupported backend: '{new_backend}'. "
            "Choose 'numpy', or 'cupy' if available."
        )


def __getattr__(name: str) -> Any:
    r"""Dynamically forward missing attribute lookups to the active backend.

    This implements PEP 562 module-level attribute redirection. Whenever code calls
    ``xp.func()`` where ``func`` is not explicitly defined in this file, CPython
    intercepts the lookup and retrieves it from the active backend module.
    """

    # prevent access to dunder attributes, breaks the documentation generator otherwise
    if name.startswith("__"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return getattr(_xp, name)
