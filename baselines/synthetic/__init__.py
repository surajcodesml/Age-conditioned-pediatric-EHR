"""Synthetic benchmark runner (S0–S3 core + S5 heterogeneous persistence)."""

__all__ = [
    "BENCHMARK_SCENARIOS",
    "CORE_SCENARIOS",
    "S5_PERSISTENCE_GROUPS",
]


def __getattr__(name: str):
    if name in __all__:
        from baselines.synthetic.data_adapter import (
            BENCHMARK_SCENARIOS,
            CORE_SCENARIOS,
            S5_PERSISTENCE_GROUPS,
        )
        return {
            "BENCHMARK_SCENARIOS": BENCHMARK_SCENARIOS,
            "CORE_SCENARIOS": CORE_SCENARIOS,
            "S5_PERSISTENCE_GROUPS": S5_PERSISTENCE_GROUPS,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
