# `model_new/` invariants

Every ID below is **HARD**: it is an unambiguous property of the code, checkable without
reference to any experimental result. Each maps to exactly one test in `model_new/tests/`.

Scientific quantities (kernel magnitude, `Δα` decomposition, headroom, recall) are
**MEASURE**, are never asserted, and live in `diagnostics.py` / `preflight.py`.

| ID | Statement | Test |
|---|---|---|
| `INV-BASIS` | No constant term (`T₀`) exists anywhere in the parameterization. `alpha_base` has exactly `s=5` entries, the generator emits `s`, and the basis returned by `ChebyshevKernel` is `T₁…T_s`. | `tests/test_inv_basis.py` |
| `INV-DOMAIN` | `τ̃ = 2τ/τ_max − 1 ∈ [−1, 1]` for every valid pair, at pretrain and at fine-tune. Values outside are clamped and counted, never silently wrapped. | `tests/test_inv_domain.py` |
| `INV-TMAX` | `τ_max` used at fine-tune equals the checkpoint value bit-for-bit. An explicit override that disagrees raises; the fine-tune corpus never re-derives it. | `tests/test_inv_tmax.py` |
| `INV-DEMO-SPLIT` | `age_years` is its own batch key. No module reads age out of `demographics`. The demographic vector still *contains* age (channel 0, raw years in the batch; standardized inside `forward` using frozen constants), and every arm receives the identical demographic tensor. | `tests/test_inv_demo_split.py` |
| `INV-QUERY` | Perturbing `age_years[:, j]` changes encoder output row `j` and no other row. Conditioning is on the **query** age. Scope: the full encoder at `n_layers=1`, block 0's output at `n_layers ≥ 2` — see note below. | `tests/test_inv_query.py` |
| `INV-ZERO-A` | At init, `vanilla`, `kernel` and `random_constant` produce **bit-identical** logits (same head width); `additive` matches to float32 tolerance (~1e-7 matmul reduction-order noise over its wider zero-padded head input). The head is drawn at the widest arm's width and sliced, so shared columns and xavier's `fan_in` are arm-independent. | `tests/test_inv_zero_a.py` |
| `INV-ZERO-B` | For `additive`, zeroing the generator's concat columns in the head leaves logits unchanged at initialization. (`additive` has a wider head, so a cross-arm bit-identity test is not well-posed.) | `tests/test_inv_zero_b.py` |
| `INV-ARM` | `kernel` and `random_constant` have identical trainable parameter counts. `vanilla` has zero age parameters. `additive` has a generator and no kernel-side age parameters. No constraint is placed on `additive`'s total. | `tests/test_inv_arm.py` |
| `INV-GROUPS` | The three optimizer groups (`backbone`, `age`, `head`) are pairwise disjoint and their union is exactly the trainable parameter set. Membership comes from module-declared sets, never from name matching. | `tests/test_inv_groups.py` |
| `INV-LOG` | No `print(` outside `diagnostics.py` and the `__main__` blocks of `train.py` / `train_finetune.py`. No file imports `model/`, `model_ablation/`, or `finetune/`. | `tests/test_inv_log.py` |
| `INV-FROZEN` | `embedding_table` and every Fourier frequency buffer have `requires_grad == False`, are persistent, and are restored from the checkpoint rather than rebuilt from defaults. | `tests/test_inv_frozen.py` |
| `INV-NAN` | A ragged batch (including a length-1 row and a heavily padded row) yields finite gradients for every trainable parameter, in every arm. | `tests/test_inv_nan.py` |
| `INV-STATS-SINGLE` | Corpus statistics come from exactly one function, `data.corpus_stats`. `preflight` calls it once. `τ_max` and every per-event statistic are exact over the full split; only the O(L²) pairwise quantities are sampled, and the sample size is recorded on the returned object. | `tests/test_inv_stats_single.py` |
| `INV-AGESTD` | The demographic-age standardization constants `(mean, sd)` are frozen from the pretraining corpus, serialize with the checkpoint, and are reused verbatim at fine-tune; an override that disagrees raises. `age_years` fed to `ψ` stays raw — only demographic channel 0 is standardized. | `tests/test_inv_agestd.py` |

Supporting tests (not invariants, still HARD):

| Test | Statement |
|---|---|
| `tests/test_chebyshev_numpy.py` | `ChebyshevKernel` matches `numpy.polynomial.chebyshev.chebval` with a zeroed constant term to 1e-6. |
| `tests/test_checkpoint_roundtrip.py` | Save → load reproduces bit-identical logits for every arm, and restores `τ_max` and the frequency buffers rather than rebuilding them. |
| `tests/test_no_stray_logging.py` | The grep half of `INV-LOG`, kept separately so the failure message names the offending file and line. |

## Notes

### `--center_delta_alpha` collapses `random_constant` onto `vanilla`

`--center_delta_alpha` subtracts the mean of `Δα` over a fixed reference age grid. The
`random_constant` generator receives a *constant* input, so its `Δα` is constant over that
grid and centering makes it **exactly zero**. Enabling the flag therefore turns
`random_constant` into `vanilla` plus dead parameters, and destroys the capacity control.
The flag defaults to **off** and `arms.assert_arm_invariants` raises if it is combined with
the `random_constant` arm.

### Why `INV-ZERO-A` needs a dedicated re-initialization pass

Constructing the age modules consumes draws from the global RNG. A naive `manual_seed(s)`
therefore gives *different* shared backbone parameters per arm, and `INV-ZERO-A` would fail
for a reason that has nothing to do with the age pathway. After construction, every non-age
trainable parameter is re-initialized from a **per-parameter** `torch.Generator` seeded by
`(seed, parameter name)`, so a shape change in one arm (e.g. `additive`'s wider head input)
cannot shift the draws for any other parameter. The head's first layer is additionally drawn
at the widest arm's input width and sliced, so `fan_in` — and therefore the xavier scale of
every weight in that layer — is arm-independent. Without that, `additive` would start from a
different point in function space, an uncontrolled difference `INV-ZERO-B` cannot see.

### Why `additive` is not bit-identical, and why that is correct

`additive`'s head input is `s` columns wider. Those columns are multiplied by an exactly-zero
generator output at init, so they contribute nothing — but the matmul reduces over a longer
axis and accumulates in a different order, leaving float32 rounding noise of order `1e-7`.
`INV-ZERO-A` asserts bit-identity for the three same-width arms and a `< 1e-5` bound for
`additive`. The `additive` extra columns are given a **normal** draw, never zeros: zeroing
them as well as the generator's final layer would make both gradients vanish permanently
(`∂L/∂W_c = δ·gᵀ = 0` because `g = 0`, and `∂L/∂g = W_cᵀδ = 0` because `W_c = 0`), so the
pathway could never start. This mirrors the kernel arm, where a nonzero `α_base` plays the
role of `W_c`.

### Why the diagonal of the padding mask is forced `True`

`pad_mask = mask[:,None,:] & mask[:,:,None]` leaves padded rows entirely `False`. A row of
all `-inf` makes `softmax` return `NaN`. The downstream `masked_fill` happens to repair both
the value and the gradient — `masked_fill`'s backward *fills* rather than multiplies — but
relying on that is fragile. Setting the diagonal `True` makes `softmax` well-defined
everywhere. `INV-NAN` is kept as a test regardless.
