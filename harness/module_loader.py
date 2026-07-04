"""Compatibility helper for loading optional worker modules."""

from __future__ import annotations

import importlib
from typing import Any, Callable


def load_callable(module_name: str, function_name: str = "run") -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return fn
