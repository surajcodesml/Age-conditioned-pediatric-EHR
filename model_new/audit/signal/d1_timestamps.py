"""D1 — Timestamp dependence (GPU).

v2 condition grid: jitter only on vanilla; kernel only for inertness assertion.
11 forwards instead of 18.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT
from model_new.audit.signal import (
    D1_JITTER_DAYS,
    D1_KERNEL_CONDITIONS,
    D1_VANILLA_CONDITIONS,
    KS,
)
from model_new.audit.signal.common import (
    SIGNAL_SEED,
    add_common_args,
    assert_batch_hash,
    assert_constant_tau_zero,
    assert_shuffle_preserves_multiset,
    autotune_batch_size,
    base_result_meta,
    ensure_batches,
    eval_model_on_store,
    iter_packed_batches,
    load_arm_model,
    mutate_constant_timestamps,
    mutate_jitter,
    mutate_shuffle_within,
    paired_delta_ci,
    probe_precision,
    require_cuda,
    resolve_device,
    write_json_atomic,
)
from model_new.train import set_seed


def _summarize_vs_true(cond: dict, true: dict, patient_ids: np.ndarray,
                       n_boot: int, seed: int) -> dict:
    out = {
        "bce_mean": cond["bce_mean"],
        "recall": cond["recall"],
        "batch_list_hash": cond["batch_list_hash"],
        "delta_vs_true": {},
    }
    if "per_code_hit_miss" in cond:
        out["per_code_hit_miss"] = cond["per_code_hit_miss"]
    for name, key in [("bce", "bce_per_example")] + [
        (f"recall@{k}", None) for k in KS
    ]:
        if name.startswith("recall@"):
            a = cond["recall_per_example"][name]
            b = true["recall_per_example"][name]
        else:
            a = cond[key]
            b = true[key]
        d = paired_delta_ci(a, b, patient_ids, n_boot=n_boot, seed=seed)
        out["delta_vs_true"][name] = {
            "point": d["point"],
            "ci": {
                "lo": d["ci"]["lo"], "hi": d["ci"]["hi"],
                "excludes_zero": d["ci"].get("excludes_zero"),
                "degenerate": d["ci"].get("degenerate", False),
            },
            "covers_zero": d["covers_zero"],
        }
    return out


def _conditions_for_arm(arm: str, seed: int) -> dict:
    base = {
        "true": None,
        "true_repeat": None,
        "constant": mutate_constant_timestamps,
        "shuffle_within": lambda b: mutate_shuffle_within(
            b, np.random.default_rng(seed + 11)),
        **{
            f"jitter_{k}": (
                lambda b, kk=k: mutate_jitter(
                    b, float(kk), np.random.default_rng(seed + 100 + int(kk)))
            )
            for k in D1_JITTER_DAYS
        },
    }
    keep = D1_VANILLA_CONDITIONS if arm == "vanilla" else D1_KERNEL_CONDITIONS
    return {k: base[k] for k in keep}


def run_d1(ctx: dict, store: dict, device: torch.device) -> dict:
    flags = ctx["flags"]
    n_boot = int(flags["n_boot"])
    seed = int(ctx.get("seed", SIGNAL_SEED))
    set_seed(seed)
    require_cuda(device, batch_size=int(store["batch_size"]), label="D1")
    assert_batch_hash(store, ctx["batch_meta"]["batch_list_hash"])
    patient_ids = np.asarray(store["patient_ids"])

    probe = next(iter_packed_batches(store))
    probe_clean = {k: v for k, v in probe.items()
                   if k not in ("target_gap_days", "age_last") and not str(k).startswith("_")}
    const_b = mutate_constant_timestamps(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in probe_clean.items()}
    )
    assert_constant_tau_zero(const_b)
    shuf_b = mutate_shuffle_within(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in probe_clean.items()},
        np.random.default_rng(seed + 11),
    )
    assert_shuffle_preserves_multiset(probe_clean, shuf_b)

    arm_results = {}
    precision_info = {}
    autotune_info = {}
    n_forwards = 0

    for arm in ("vanilla", "kernel"):
        model, ckpt_meta = load_arm_model(ctx, arm, device)
        prec = probe_precision(model, store, device)
        precision_info[arm] = prec
        dtype = torch.bfloat16 if prec["dtype"] == "bf16" else torch.float32
        autotune_info[arm] = {
            "fitted_batch_size": autotune_batch_size(model, store, device),
            "materialized_batch_size": int(store["batch_size"]),
        }

        conditions = _conditions_for_arm(arm, seed)
        cond_out = {}
        true_res = None
        for cname, mut in conditions.items():
            res = eval_model_on_store(
                model, store, device, mutate_batch=mut, dtype=dtype,
                collect_per_code=True,
            )
            n_forwards += 1
            if res["batch_list_hash"] != store["batch_list_hash"]:
                if cname in ("true", "true_repeat", "constant", "shuffle_within"):
                    raise AssertionError(
                        f"[HARD] {arm}/{cname} batch hash {res['batch_list_hash']} "
                        f"!= {store['batch_list_hash']}")
            if cname == "true":
                true_res = res
                cond_out[cname] = {
                    "bce_mean": res["bce_mean"],
                    "recall": res["recall"],
                    "batch_list_hash": res["batch_list_hash"],
                    "per_code_hit_miss": res["per_code_hit_miss"],
                    "_full": res,
                }
            elif cname == "true_repeat":
                assert true_res is not None
                for k in KS:
                    a = res["recall_per_example"][f"recall@{k}"]
                    b = true_res["recall_per_example"][f"recall@{k}"]
                    if not np.array_equal(a, b):
                        raise AssertionError(
                            f"[HARD] determinism fail {arm} recall@{k}: true != true_repeat")
                if not np.array_equal(res["bce_per_example"], true_res["bce_per_example"]):
                    raise AssertionError(
                        f"[HARD] determinism fail {arm} bce: true != true_repeat")
                d = _summarize_vs_true(res, true_res, patient_ids, n_boot, seed)
                for metric, blob in d["delta_vs_true"].items():
                    blob["point"] = 0.0
                    blob["ci"] = {"lo": 0.0, "hi": 0.0, "degenerate": True,
                                  "excludes_zero": False}
                    blob["covers_zero"] = True
                cond_out[cname] = d
            else:
                cond_out[cname] = _summarize_vs_true(
                    res, true_res, patient_ids, n_boot, seed
                )

        cond_out["true"] = {
            "bce_mean": true_res["bce_mean"],
            "recall": true_res["recall"],
            "batch_list_hash": true_res["batch_list_hash"],
            "per_code_hit_miss": true_res["per_code_hit_miss"],
        }
        arm_results[arm] = {
            "checkpoint": ckpt_meta,
            "precision": prec,
            "conditions": cond_out,
            "condition_names": list(conditions.keys()),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    v_const = arm_results["vanilla"]["conditions"]["constant"]["delta_vs_true"]["recall@10"]
    k_const = arm_results["kernel"]["conditions"]["constant"]["delta_vs_true"]["recall@10"]
    kernel_ge_vanilla_degrade = bool(k_const["point"] <= v_const["point"] + 1e-12)

    jitter_curve = {
        f"jitter_{k}": arm_results["vanilla"]["conditions"][f"jitter_{k}"]
        ["delta_vs_true"]["recall@10"]
        for k in D1_JITTER_DAYS
    }

    def _onset(curve: dict) -> int | None:
        for k in D1_JITTER_DAYS:
            d = curve[f"jitter_{k}"]
            if d["point"] < 0 and d["ci"].get("excludes_zero"):
                return int(k)
        return None

    onset_vanilla = _onset(jitter_curve)
    abs_const = abs(v_const["point"])
    if abs_const < 0.01:
        timing_route = "objective timing-blind → build TTE"
    elif abs_const > 0.05:
        timing_route = "timing already matters → multi-head only, skip TTE"
    else:
        timing_route = "timing weakly used → TTE optional"

    out = {
        **base_result_meta(ctx, store),
        "device": str(device),
        "n_forwards": n_forwards,
        "precision": precision_info,
        "autotune": autotune_info,
        "arms": arm_results,
        "assertions": {
            "determinism_true_vs_true_repeat": True,
            "constant_tau_max_zero": True,
            "shuffle_preserves_multiset": True,
            "kernel_constant_degrades_at_least_vanilla": kernel_ge_vanilla_degrade,
            "cuda_batch_size_ok": True,
        },
        "headlines": {
            "vanilla_constant_delta_recall@10": v_const,
            "kernel_constant_delta_recall@10": k_const,
            "jitter_onset_days": {"vanilla": onset_vanilla},
            "jitter_curve_delta_recall@10": {"vanilla": jitter_curve},
        },
        "verdict": {
            "kernel_temporal_pathway_inert": not kernel_ge_vanilla_degrade,
            "timing_route": timing_route,
            "jitter_onset_route": (
                "kernel over-parameterised → drop s to 2–3"
                if (onset_vanilla is not None and onset_vanilla >= 365)
                or onset_vanilla is None
                else "timing resolution within clinical range"
            ),
        },
    }
    if not kernel_ge_vanilla_degrade:
        out["verdict"]["finding"] = (
            "kernel constant-τ degrade < vanilla → kernel temporal pathway inert"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    D.print_block("D1 timestamp dependence", [
        f"out={out_dir}  smoke={args.smoke}  device={device}",
        f"vanilla conditions={list(D1_VANILLA_CONDITIONS)}",
        f"kernel conditions={list(D1_KERNEL_CONDITIONS)}",
    ])
    ctx, store = ensure_batches(
        out_dir, smoke=args.smoke, batch_size=args.batch_size,
        force=args.force, run_root=args.run_root,
    )
    ctx["seed"] = args.seed
    result = run_d1(ctx, store, device)
    write_json_atomic(out_dir / "d1_timestamps.json", result)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        xs = list(D1_JITTER_DAYS)
        curve = result["headlines"]["jitter_curve_delta_recall@10"]["vanilla"]
        ys = [curve[f"jitter_{k}"]["point"] for k in xs]
        lo = [curve[f"jitter_{k}"]["ci"]["lo"] for k in xs]
        hi = [curve[f"jitter_{k}"]["ci"]["hi"] for k in xs]
        ax.errorbar(xs, ys,
                    yerr=np.array([[y - l, h - y] for y, l, h in zip(ys, lo, hi)]).T,
                    fmt="o-", capsize=3, color="#1f4e79", label="vanilla")
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Jitter ±days")
        ax.set_ylabel("Δ recall@10 vs true")
        ax.set_title("D1: jitter curve (vanilla)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "d1_jitter_curve.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        D.print_block("D1 figure", [f"skipped: {e}"])

    h = result["headlines"]
    D.print_block("D1 results", [
        f"batch_list_hash={result['batch_list_hash']}",
        f"n_forwards={result['n_forwards']}",
        f"determinism OK; constant τ=0 OK; shuffle multiset OK; cuda B OK",
        f"vanilla constant Δr@10={h['vanilla_constant_delta_recall@10']['point']:.4e}",
        f"kernel  constant Δr@10={h['kernel_constant_delta_recall@10']['point']:.4e}",
        f"kernel≥vanilla degrade={result['assertions']['kernel_constant_degrades_at_least_vanilla']}",
        f"jitter onset vanilla={h['jitter_onset_days']['vanilla']}",
        f"timing_route: {result['verdict']['timing_route']}",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
