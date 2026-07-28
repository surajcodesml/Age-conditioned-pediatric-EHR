"""Signal battery: does the data / objective contain anything for the mechanism to find?

Nothing here trains. Forward passes and SQL only (optional one backward in D4/D10).
All console / JSON I/O goes through ``model_new.diagnostics``.
"""

from __future__ import annotations

__all__ = [
    "SIGNAL_SEED",
    "N_VAL_BATCHES",
    "SMOKE_N_BATCHES",
    "SMOKE_PATIENT_FRAC",
    "SMOKE_N_BOOT",
    "SMOKE_N_PERM",
    "SMOKE_TOP_CODES",
    "N_BOOT",
    "N_PERM",
    "TOP_CODES",
    "DUCKDB_THREADS",
    "DUCKDB_MEMORY",
    "DATALOADER_WORKERS",
    "D7_AGE_BANDS",
    "D7_LAG_BINS",
    "D1_JITTER_DAYS",
    "D1_ARMS",
    "D1_VANILLA_CONDITIONS",
    "D1_KERNEL_CONDITIONS",
    "KS",
    "T4_SIGMA_CONTENT",
    "CACHE_VERSION",
    "D9_TOP_CODES_LIFT",
    "MIN_GPU_BATCH_SIZE",
]

SIGNAL_SEED = 0
N_VAL_BATCHES = 200
SMOKE_N_BATCHES = 2
SMOKE_PATIENT_FRAC = 0.01
SMOKE_N_BOOT = 50
SMOKE_N_PERM = 5
SMOKE_TOP_CODES = 20

N_BOOT = 1000
N_PERM = 100
TOP_CODES = 200

DUCKDB_THREADS = 10
DUCKDB_MEMORY = "16GB"
DATALOADER_WORKERS = 4

D7_AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("18-35", 18.0, 35.0),
    ("35-50", 35.0, 50.0),
    ("50-65", 50.0, 65.0),
    ("65-80", 65.0, 80.0),
    ("80+", 80.0, float("inf")),
)

D7_LAG_BINS: tuple[tuple[str, float, float], ...] = (
    ("0-7d", 0.0, 7.0),
    ("7-30d", 7.0, 30.0),
    ("30-90d", 30.0, 90.0),
    ("90-365d", 90.0, 365.0),
    ("365d-3y", 365.0, 3 * 365.25),
    (">3y", 3 * 365.25, float("inf")),
)

# D1 v2: jitter only on vanilla; kernel only for inertness.
D1_JITTER_DAYS: tuple[int, ...] = (7, 30, 365)
D1_ARMS: tuple[str, ...] = ("vanilla", "kernel")
D1_VANILLA_CONDITIONS: tuple[str, ...] = (
    "true", "true_repeat", "constant", "shuffle_within",
    "jitter_7", "jitter_30", "jitter_365",
)
D1_KERNEL_CONDITIONS: tuple[str, ...] = ("true", "constant", "shuffle_within")

KS: tuple[int, ...] = (5, 10, 20)

# From completed T4 (kernel/encoder_layer0); do not recompute.
T4_SIGMA_CONTENT = 3.2929

CACHE_VERSION = 2
D9_TOP_CODES_LIFT = 2000
MIN_GPU_BATCH_SIZE = 128
