"""Write AGE_AUDIT_REPORT.md and figures from age_audit.json / train.json logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from model_new import diagnostics as D
from model_new.audit.common import read_json


def _safe(d: dict, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_figures(audit: dict, out_dir: Path, run_dirs: dict[str, Path]) -> list[str]:
    """Figures from existing logs where possible; no extra forward passes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1) Δα(a) from train.json delta_alpha_grid at selected epoch.
    try:
        kernel_train = read_json(run_dirs["kernel"] / "train.json")
        ep = audit["t0"]["selected_epoch"]["kernel"]
        rec = next(x for x in kernel_train if int(x["epoch"]) == ep)
        grid = rec["delta_alpha_grid"]
        ages = np.asarray(grid["ages"], dtype=np.float64)
        support_min = float(audit.get("t3", {}).get("support_age_min", 16.6))
        # Corpus age density from t3 if present.
        dens_ages = None

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, site in zip(axes, grid["sites"]):
            da = np.asarray(grid["sites"][site], dtype=np.float64)  # [A, s]
            norms = np.linalg.norm(da, axis=-1)
            mean = da.mean(axis=0)
            resid = da - mean
            var_norms = np.linalg.norm(resid, axis=-1)
            ax.plot(ages, norms, label="‖Δα‖", color="#1b4f72")
            ax.plot(ages, var_norms, label="‖Δα̃‖", color="#b03a2e")
            ax.axvline(support_min, color="#566573", ls="--", lw=1, label="support min")
            ax.set_title(site)
            ax.set_xlabel("age (y)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("norm")
        fig.suptitle(f"Δα(a) at selected epoch {ep} (from train.json)")
        fig.tight_layout()
        path = fig_dir / "delta_alpha_curves.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    except Exception as exc:
        written.append(f"delta_alpha_curves FAILED: {exc}")

    # 2) w(τ|a) from train.json w_curves.
    try:
        kernel_train = read_json(run_dirs["kernel"] / "train.json")
        ep = audit["t0"]["selected_epoch"]["kernel"]
        rec = next(x for x in kernel_train if int(x["epoch"]) == ep)
        wc = rec["w_curves"]
        tau = np.asarray(wc["tau_grid"], dtype=np.float64)
        ages = wc["ages"]
        # Pick 5 ages closest to 18,25,40,65,80.
        targets = [18, 25, 40, 65, 80]
        pick = [min(range(len(ages)), key=lambda i: abs(ages[i] - t)) for t in targets]
        site = "pooling" if "pooling" in wc["sites"] else next(iter(wc["sites"]))
        log_w = np.asarray(wc["sites"][site]["log_w"], dtype=np.float64)
        dens = np.asarray(audit.get("t4", {}).get("tau_density") or [], dtype=np.float64)
        tau_g = np.asarray(audit.get("t4", {}).get("tau_grid") or [], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, ti in zip(pick, targets):
            ax.plot(tau, log_w[i], label=f"a≈{ages[i]:.0f} (target {ti})")
        if dens.size and tau_g.size:
            ax2 = ax.twinx()
            ax2.fill_between(tau_g, dens, alpha=0.2, color="#7f8c8d", label="τ density")
            ax2.set_ylabel("empirical τ density")
        ax.set_xlabel("τ")
        ax.set_ylabel("log w")
        ax.set_title(f"w(τ|a) at {site}, epoch {ep}")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        path = fig_dir / "w_tau_curves.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    except Exception as exc:
        written.append(f"w_tau_curves FAILED: {exc}")

    # 3) Permutation null histogram.
    try:
        null = audit["t6"]["per_arm"]["kernel"]["null_recall@10"]
        true = audit["t6"]["per_arm"]["kernel"]["true"]["recall@10"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null, bins=30, color="#5d6d7e", alpha=0.85, label="permutation null")
        ax.axvline(true, color="#c0392b", lw=2, label=f"true ages ({true:.4f})")
        ax.set_xlabel("recall@10")
        ax.set_title("T6 permutation null (kernel)")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / "permutation_null.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    except Exception as exc:
        written.append(f"permutation_null FAILED: {exc}")

    # 4) Per-frequency band usability.
    try:
        comps = audit["t7"]["a_fourier_band"]["components"]
        colors = {"usable": "#1e8449", "near_constant": "#f4d03f", "hash_like": "#c0392b"}
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = [c["m"] for c in comps]
        ys = [c["cycles_over_u_span"] for c in comps]
        cs = [colors[c["band"]] for c in comps]
        ax.bar(xs, ys, color=cs)
        ax.axhline(0.25, color="#566573", ls="--", lw=1)
        ax.axhline(3.0, color="#566573", ls="--", lw=1)
        ax.set_xlabel("frequency index m")
        ax.set_ylabel("u_span / p_m (cycles)")
        ax.set_title("Fourier band usability over corpus u-range")
        fig.tight_layout()
        path = fig_dir / "fourier_band_usability.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    except Exception as exc:
        written.append(f"fourier_band_usability FAILED: {exc}")

    return written


def write_report(audit: dict, out_dir: Path, run_dirs: dict[str, Path]) -> Path:
    t1 = audit.get("t1", {})
    t2 = audit.get("t2", {})
    t3 = audit.get("t3", {})
    t4 = audit.get("t4", {})
    t5 = audit.get("t5", {})
    t6 = audit.get("t6", {})
    t7 = audit.get("t7")

    # Decision logic per interpretation rules.
    t5_nz = bool(t5.get("kernel_effect_nonzero"))
    t6_sig = bool(t6.get("kernel_age_dependent"))
    t3_inert = all(v.get("inert") for v in (t3.get("kernel_inert_on_support") or {}).values())
    t4_R = t4.get("kernel_max_R")
    t4_dead = (t4_R is not None and np.isfinite(t4_R) and t4_R < 0.01)
    w2_stuck = False
    esc = (t2.get("saddle_escape") or {}).get("kernel") or {}
    if esc:
        w2_stuck = all(v.get("stuck_near_zero") for v in esc.values())
    w2_final = max((v.get("final_W2_fro") or 0 for v in esc.values()), default=0.0)

    pool_tv = _safe(t5, "headline_25_vs_75", "direct_25_vs_75_pool_tv")
    if t5_nz and t6_sig:
        verdict = "YES"
        deciding = (
            f"T5 kernel effect nonzero and T6 p<0.05 "
            f"(p_bce={_safe(t6,'per_arm','kernel','p_bce')}, "
            f"p_r@10={_safe(t6,'per_arm','kernel','p_recall@10')}; "
            f"headline 25-vs-75 pool TV={pool_tv})."
        )
    elif t5_nz and not t6_sig:
        verdict = "NO"
        deciding = (
            f"T5 shows a real forward-pass age effect "
            f"(25-vs-75 pool TV={pool_tv}), but T6 permutation p≥0.05 "
            f"(p_bce={_safe(t6,'per_arm','kernel','p_bce')}, "
            f"p_r@10={_safe(t6,'per_arm','kernel','p_recall@10')}) — "
            f"the kernel moves attention/representations without helping the pretraining "
            f"objective under true ages. Deciding number: T6 p_recall@10."
        )
    elif (not t3_inert) and (t4_dead or not t5_nz):
        verdict = "NO"
        deciding = (
            f"T3 shows age-varying Δα but T4 R={t4_R} < 0.01 or T5≈0 — "
            f"the generator learned an age function the softmax discards "
            f"(mechanism-level failure)."
        )
    elif t3_inert and w2_stuck:
        verdict = "NO"
        deciding = (
            f"T3≈0 with ‖W2‖≈{w2_final:.2e} — the zero-init saddle never broke "
            f"(init/optimizer failure)."
        )
    elif t3_inert and w2_final > 1e-3:
        verdict = "NO"
        deciding = (
            f"T3≈0 with ‖W2‖={w2_final:.3e} large — generator learned a constant; "
            f"check T7(a) band resolution."
        )
    else:
        verdict = "NO"
        deciding = (
            f"T5 effect={'nonzero' if t5_nz else '≈0'}, "
            f"T6 active={t6_sig}, T3 inert={t3_inert}, T4 R={t4_R}. "
            f"No evidence of a functionally active age-dependent kernel."
        )

    # Table rows.
    def ci_str(ci):
        if not ci:
            return "—"
        return f"[{ci.get('lo'):.4g}, {ci.get('hi'):.4g}]"

    rows = []
    # T1
    for name, d in (t1.get("kernel_minus_random_constant") or {}).items():
        rows.append((
            "T1", f"kernel−rc {name}",
            f"{d.get('kernel_minus_random_constant'):.4g}",
            "0 (ref)", "—",
            ci_str(d.get("ci")),
            "fail" if d.get("covers_zero") else "pass",
        ))
    # T2
    for site, v in ((t2.get("saddle_escape") or {}).get("kernel") or {}).items():
        rows.append((
            "T2", f"‖W2‖ escape {site}",
            f"final={v.get('final_W2_fro'):.3e} ep={v.get('first_escape_epoch')}",
            f"{_safe(t2,'saddle_escape','random_constant',site,'final_W2_fro')}",
            "n/a", "—",
            "pass" if v.get("escaped") else "fail",
        ))
    # T3
    for site, v in (t3.get("kernel_inert_on_support") or {}).items():
        rows.append((
            "T3", f"varying_frac {site}",
            f"{v.get('varying_frac'):.4f}",
            "0 (asserted)", "0",
            "—",
            "fail" if v.get("inert") else "pass",
        ))
    # T4
    rows.append((
        "T4", "max R = max D / σ_content",
        f"{t4_R}",
        f"{_safe(t4,'per_arm','random_constant', next(iter((_safe(t4,'per_arm','random_constant') or {})), None), 'R')}",
        "0",
        "—",
        "fail" if t4_dead else "pass",
    ))
    # T5
    rows.append((
        "T5", "25-vs-75 pool TV",
        f"{_safe(t5,'headline_25_vs_75','direct_25_vs_75_pool_tv')}",
        "0", "0",
        ci_str(_safe(t5, "headline_25_vs_75", "direct_25_vs_75_pool_tv_ci")),
        "pass" if t5_nz else "fail",
    ))
    rows.append((
        "T5", "25-vs-75 ‖Δh‖/‖h‖",
        f"{_safe(t5,'headline_25_vs_75','direct_25_vs_75_rel_dh')}",
        "0", "0",
        ci_str(_safe(t5, "headline_25_vs_75", "direct_25_vs_75_rel_dh_ci")),
        "pass" if t5_nz else "fail",
    ))
    # T6
    rows.append((
        "T6", "permutation p (BCE / r@10)",
        f"{_safe(t6,'per_arm','kernel','p_bce')} / {_safe(t6,'per_arm','kernel','p_recall@10')}",
        f"{_safe(t6,'per_arm','random_constant','p_bce')} / {_safe(t6,'per_arm','random_constant','p_recall@10')}",
        f"{_safe(t6,'per_arm','vanilla','p_bce')} / {_safe(t6,'per_arm','vanilla','p_recall@10')}",
        "—",
        "pass" if t6_sig else "fail",
    ))

    figs = write_figures(audit, out_dir, run_dirs)

    cannot_rule_out = [
        "Effects that appear only after fine-tuning on PIC (this audit is pretrained checkpoints only).",
        "Effects confined to rare age bands with n below the bootstrap's resolution.",
        "Higher-order interactions between R1 and R2 that cancel under the R2-only intervention.",
        "Non-attention pathways (none exist for kernel by design) or bugs outside the tested arms.",
        "Different random seeds (only seed 0 is present under model_new/run/).",
    ]
    if audit.get("meta", {}).get("max_val_batches"):
        cannot_rule_out.append(
            f"Full-split effects beyond the evaluated "
            f"{audit['meta']['max_val_batches']} validation batches "
            f"({audit['meta'].get('n_examples')} sequences)."
        )

    lines = []
    lines.append("# Age-route audit report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"**The pretrained kernel arm did {'not ' if verdict == 'NO' else ''}"
        f"learn a functionally active, age-dependent temporal kernel.** {deciding}"
    )
    lines.append("")
    if not t1.get("age_route_measurable_on_pretrain_objective", True):
        lines.append(
            f"> **T1 PROMINENT:** {t1.get('headline')}"
        )
        lines.append("")
    lines.append("## Results table")
    lines.append("")
    lines.append("| test | metric | kernel | random_constant | vanilla | CI | pass/fail |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")

    if verdict == "NO" and t7 is not None:
        lines.append("## Ranked null causes (T7)")
        lines.append("")
        for i, c in enumerate(t7.get("ranked_causes") or [], 1):
            lines.append(f"{i}. **{c['id']}** (strength={c['strength']:.3f}): {c['number']}")
            lines.append(f"   - explains null if: {c['explains_null_if']}")
        lines.append("")
        e = t7.get("e_r1_redundancy") or {}
        if e.get("r1_saturates_age") and not t5_nz and not t6_sig:
            lines.append(
                "R1 already saturates the age signal in `h` "
                f"(R²={e.get('linear_probe_R2_h_to_age'):.3f}). "
                "The premise that the kernel route has residual age information to capture "
                "is the problem, not (only) the architecture."
            )
            lines.append("")

    lines.append("## What this audit cannot rule out")
    lines.append("")
    for item in cannot_rule_out:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for f in figs:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    t0 = audit.get("t0", {})
    lines.append(f"- seed={t0.get('seed')}, tau_max={t0.get('tau_max')}")
    lines.append(f"- selected epochs: {t0.get('selected_epoch')}")
    lines.append(f"- checkpoints: {t0.get('checkpoint_paths')}")
    lines.append(f"- batch_order_hash: {t0.get('batch_order_hash')}")
    lines.append(f"- audit seed: {audit.get('meta', {}).get('seed')}")
    lines.append("")

    path = out_dir / "AGE_AUDIT_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    D.print_block("AGE_AUDIT_REPORT.md", [f"wrote {path}", f"verdict={verdict}", deciding])
    return path
