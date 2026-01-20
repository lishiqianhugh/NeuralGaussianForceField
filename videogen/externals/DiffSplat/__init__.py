import sys as _sys

from . import src as _src
from . import extensions as _extensions

_sys.modules.setdefault("src", _src)
_sys.modules.setdefault("extensions", _extensions)

from .gen_diffsplat import DiffSplat

__all__ = [
    # Expose class for direct import
    "DiffSplat",
]

