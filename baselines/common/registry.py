"""Model registry — look up baseline classes by name."""
from __future__ import annotations

from typing import Any, Callable, Type

REGISTRY: dict[str, Type] = {}


def register_baseline(name: str) -> Callable:
    """Decorator that registers a baseline class under *name*."""
    def decorator(cls: Type) -> Type:
        if name in REGISTRY:
            raise ValueError(f"baseline {name!r} already registered")
        REGISTRY[name] = cls
        return cls
    return decorator


def get_baseline(name: str) -> Type:
    """Retrieve a registered baseline class."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown baseline {name!r}; registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
