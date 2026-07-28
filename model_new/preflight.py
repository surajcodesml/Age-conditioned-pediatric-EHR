#!/usr/bin/env python3
"""Phase 10 review checkpoint. Run against real batches before a multi-day pretrain.

This is a **review checkpoint, not an automated gate**: it computes everything, prints a
summary, and exits cleanly. Nothing here raises except the two structural checks it borrows
(INV-DEMO-SPLIT, INV-ZERO-A/B), which are HARD everywhere.

The measurement that matters most is the **headroom**: load the initialised model, force two
maximally different coefficient vectors (sharp decay vs flat), and measure ``max|Dlogit|`` on
real batches, in absolute terms and relative to the batch's logit standard deviation. If
maximally different kernels barely move the output, no arm can differ from any other and
pretraining cannot answer the question. For reference, the same measurement on PIC gave
``max|Dlogit| = 0.0059``.

The context for that number is the **within-row tau spread**. Softmax is invariant to a
per-row constant, so it is the spread of ``tau`` across the keys of a single attention row --
not the marginal ``tau`` histogram -- that determines whether kernel shape can matter at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_new import diagnostics as D
from model_new.age_encoding import LinearAgeFourier, LogAgeFourier, characterize_band
from model_new.arms import ARMS
from model_new.data import (
    TensorizedPretrainDataset, corpus_stats, demo_layout,
    dataloader_worker_init, make_collate,
)
from model_new.model import DKMModel
from model_new.train import resolve_block_flags, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]

# Two EQUAL-NORM, opposite-shape kernels on tau_tilde in [-1, 1]. The earlier probe used
# alpha_flat = 0 -- literally no kernel -- so it measured kernel-versus-nothing rather than
# shape discrimination. These have the same L2 norm and opposite sign on T_1 and T_3, so the
# comparison is "can the model tell a decaying kernel from a growing one", which is the claim.
SHAPE_SCALE = 2.0


def forced_alpha_pair(s: int, scale: float = SHAPE_SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """(decaying, growing) -- equal norm, opposite shape. ``[-2,0,-1,0,0]`` vs ``[+2,0,+1,0,0]``."""
    a = torch.zeros(s)
    a[0] = scale
    if s >= 3:
        a[2] = scale / 2.0
    return -a, a


def _logits_of(out: dict) -> torch.Tensor:
    """The head's output, whichever task the model was built for.

    ``task='pretrain'`` emits ``code_logits [B, |V|]`` and ``task='classification'``
    emits ``logits [B]``. The probe is the same measurement either way -- how far two
    maximally different kernel shapes move the head -- so it reads whichever key is
    present rather than existing twice (section 10).
    """
    for key in ("code_logits", "logits"):
        if key in out:
            return out[key]
    raise KeyError(f"model output has neither 'code_logits' nor 'logits': {sorted(out)}")


@torch.no_grad()
def headroom(model: DKMModel, batches: list[dict], s: int) -> dict:
    """max|Dlogit| between two forced kernel shapes, absolute and relative to logit sd."""
    sharp, flat = forced_alpha_pair(s)
    sites = model.kernel_sites()
    saved = [site.alpha_base.detach().clone() for _, site in sites]

    def run(alpha: torch.Tensor) -> list[torch.Tensor]:
        for _, site in sites:
            site.alpha_base.copy_(alpha.to(site.alpha_base.device))
        return [_logits_of(model(b)).float().cpu() for b in batches]

    a_logits, b_logits = run(sharp), run(flat)
    for (_, site), old in zip(sites, saved):
        site.alpha_base.copy_(old)

    diffs = torch.cat([(a - b).abs().reshape(-1) for a, b in zip(a_logits, b_logits)])
    sds = torch.tensor([b.std() for b in b_logits])
    max_abs = float(diffs.max())
    sd = float(sds.mean())
    return {
        "alpha_decaying": sharp.tolist(),
        "alpha_growing": flat.tolist(),
        "probe": "equal-norm opposite-shape kernels ([-2,0,-1,0,0] vs [+2,0,+1,0,0])",
        "max_abs_delta_logit": max_abs,
        "mean_abs_delta_logit": float(diffs.mean()),
        "p99_abs_delta_logit": _quantiles(diffs, [0.99])["0.99"],
        "logit_sd": sd,
        "max_delta_over_logit_sd": max_abs / sd if sd > 0 else float("inf"),
        "n_batches": len(batches),
        "reference_pic_max_abs_delta_logit": 0.0059,
    }


def _quantiles(x: torch.Tensor, qs: list[float]) -> dict[str, float]:
    """``torch.quantile`` caps input size, and a full batch of pairs is far past it."""
    a = x.detach().cpu().numpy()
    return {str(q): float(np.percentile(a, 100.0 * q)) for q in qs}


@torch.no_grad()
def batch_domain(batches: list[dict], tau_max: float) -> dict:
    """The frozen ``tau_max`` domain as the LOADED batches actually exercise it: ``tau_tilde``
    range and clamp rate. Distinct from :func:`data.corpus_stats`, which reports the tau
    distribution and within-row spread over the corpus; this confirms the specific batches a
    forward pass will see stay in ``[-1, 1]``."""
    from model_new.data import pairwise_tau

    tvs = []
    for b in batches:
        mask = b["attention_mask"].cpu()
        tau = pairwise_tau(b["timestamps_days"].cpu(), mask)   # tau is not in the batch
        pair = mask.unsqueeze(2) & mask.unsqueeze(1)
        tvs.append(tau[pair])
    tv = torch.cat(tvs)
    tt = 2.0 * tv / tau_max - 1.0
    return {
        "tau_max_used": float(tau_max),
        "tau_tilde_min": float(tt.min()),
        "tau_tilde_max": float(tt.max()),
        "clamp_rate": float(((tt < -1) | (tt > 1)).float().mean()),
    }


@torch.no_grad()
def pic_conditioning(pic_dir: Path, vocab_path: Path, tau_max: float, s: int,
                     n_windows: int = 400, seed: int = 0) -> dict | None:
    """M1 -- Chebyshev conditioning on the PIC tau_tilde distribution under the frozen
    pretrain ``tau_max``.

    PIC stays are ICU-length, so under the frozen ``tau_max`` the PIC tau_tilde distribution
    occupies a narrow sub-interval near -1. Chebyshev polynomials are near-orthogonal on the
    whole ``[-1, 1]``, not on a sub-interval, so the basis can be well-conditioned at
    pretraining and ill-conditioned at fine-tune. Both ``tau_max`` and the basis are frozen
    in the checkpoint, so this must be known before pretraining -- if it is bad, the response
    is to change them now.

    Returns None if PIC tensors are not found; never builds fine-tune machinery.
    """
    from model_new.data_finetune import TensorizedFinetuneDataset

    split = None
    for cand in ("train", "."):
        d = pic_dir / cand if cand != "." else pic_dir
        if list(d.glob("shard_*.npz")):
            split = d
            break
    if split is None:
        return None

    ds = TensorizedFinetuneDataset(split, max_seq_len=1024)
    rng = np.random.default_rng(seed)
    taus: list[np.ndarray] = []
    for j in _sample_indices_pic(len(ds), n_windows, seed):
        ts = ds[int(j)]["timestamps_days"].astype(np.float64)
        if ts.size < 2:
            continue
        iu = np.triu_indices(ts.size, k=1)
        d = np.abs(ts[:, None] - ts[None, :])[iu]
        if d.size > 3000:
            d = rng.choice(d, 3000, replace=False)
        taus.append(np.log1p(d / 7.0))
    if not taus:
        return None
    tau = np.concatenate(taus)
    tt = np.clip(2.0 * tau / tau_max - 1.0, -1.0, 1.0)
    cond = D.gram_condition_numbers(tau, s, tau_max)
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    return {
        "pic_dir": str(split),
        "n_pairs": int(tau.size),
        "tau_tilde_min": float(tt.min()),
        "tau_tilde_max": float(tt.max()),
        "tau_tilde_quantiles": {str(q): float(np.percentile(tt, 100 * q)) for q in qs},
        "occupancy_fraction_of_domain": float((tt.max() - tt.min()) / 2.0),
        "chebyshev_cond_no_constant": cond["chebyshev_no_constant"],
        "chebyshev_cond_with_constant": cond["chebyshev_with_constant"],
        "monomial_cond_no_constant": cond["monomial_no_constant"],
        "mimic_reference_chebyshev_cond_no_constant": 15.1,
    }


def _sample_indices_pic(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.arange(n) if k >= n else rng.choice(n, size=k, replace=False)


@torch.no_grad()
def structural_checks(batches: list[dict], num_codes: int, embedding_path: Path,
                      demo_dim: int, demo_channels, race_encoding: str,
                      tau_max: float, model_kwargs: dict) -> dict:
    """INV-DEMO-SPLIT and INV-ZERO-A/B, evaluated on real data. HARD."""
    out: dict = {}
    b0 = batches[0]
    out["INV-DEMO-SPLIT"] = {
        "age_years_is_own_key": "age_years" in b0,
        "demo_dim": int(b0["demographics"].shape[-1]),
        "demo_channels": list(demo_channels),
        "age_present_in_demo_channel_0": bool(
            torch.allclose(b0["demographics"][..., 0].cpu(),
                           (b0["age_years"] * b0["attention_mask"]).cpu(), atol=1e-6)),
    }

    # standardization was applied to demo channel 0; record it (Fix C item 5).
    out["INV-DEMO-SPLIT"]["age_channel_standardized_in_forward"] = True

    def build(arm: str) -> DKMModel:
        return DKMModel(num_codes=num_codes, embedding_path=embedding_path, arm=arm,
                        demo_dim=demo_dim, demo_channels=demo_channels,
                        race_encoding=race_encoding, tau_max=tau_max, **model_kwargs)

    logits = {}
    for arm in ARMS:
        m = build(arm).to(b0["code_indices"].device).eval()
        logits[arm] = m(b0)["code_logits"].float().cpu()
        if arm == "additive":
            # INV-ZERO-B: zeroing the head's concat columns must not change logits at init.
            before = logits[arm].clone()
            m.head.net[0].weight[:, -m.s:].zero_()
            out["INV-ZERO-B"] = {
                "max_abs_delta": float((m(b0)["code_logits"].float().cpu() - before).abs().max())}
    # INV-ZERO-A now spans all four arms: same-width arms are bit-identical, additive matches
    # to float tolerance (matmul reduction order over its wider zero-padded input).
    out["INV-ZERO-A"] = {
        f"{arm}_vs_vanilla_max_abs": float((logits[arm] - logits["vanilla"]).abs().max())
        for arm in ("kernel", "random_constant", "additive")}
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tensorized_dir", type=Path,
                   default=REPO_ROOT / "data/processed/tensorized_flat")
    p.add_argument("--embedding_path", type=Path,
                   default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--race_encoding", choices=("one_hot", "scalar"), default="one_hot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--s", type=int, default=5)
    p.add_argument("--age_M", type=int, default=16)
    p.add_argument("--age_p_min", type=float, default=0.15)
    p.add_argument("--age_p_max", type=float, default=6.0)
    p.add_argument("--tau_max", type=float, default=None)
    p.add_argument("--stats_sample_windows", type=int, default=4000,
                   help="windows sampled for the O(L^2) corpus statistics; tau_max and all "
                        "per-event stats are exact over the full split regardless.")
    p.add_argument("--pic_dir", type=Path,
                   default=REPO_ROOT / "data/finetune/heart_malformations",
                   help="PIC tensorized fine-tune data for the M1 conditioning check.")
    p.add_argument("--legacy_block", action="store_true")
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--no_layernorm", action="store_true")
    p.add_argument("--no_ffn", action="store_true")
    p.add_argument("--out", type=Path, default=Path("model_new/run/preflight.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")

    ds = TensorizedPretrainDataset(args.tensorized_dir / args.split, args.vocab_path,
                                   max_seq_len=args.max_seq_len)
    demo_dim, demo_channels = demo_layout(args.race_encoding)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=make_collate(args.race_encoding),
                        worker_init_fn=dataloader_worker_init,
                        generator=torch.Generator().manual_seed(args.seed))
    batches = []
    for i, b in enumerate(loader):
        if i >= args.n_batches:
            break
        batches.append({k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in b.items()})

    # Single source of truth for every corpus statistic (Fix A), including the exact tau_max
    # and the age standardization constants.
    stats = corpus_stats(ds, split=args.split, sample_windows=args.stats_sample_windows,
                         seed=args.seed)
    tau_max = float(args.tau_max) if args.tau_max is not None else stats.tau_max
    tau_source = (f"--tau_max override ({args.tau_max})" if args.tau_max is not None
                  else stats.tau_max_source)
    age_mean, age_sd = stats.event_age_mean, stats.event_age_sd

    use_residual, use_layernorm, use_ffn = resolve_block_flags(args)
    common = dict(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                  s=args.s, age_M=args.age_M, age_p_min=args.age_p_min,
                  age_p_max=args.age_p_max, seed=args.seed,
                  age_mean=age_mean, age_sd=age_sd)

    # The headroom depends on the encoder block, so report every configuration rather than
    # guessing which one to commit to. The one-component-at-a-time rows say WHICH part of
    # the block decides whether the kernel can move the output at all.
    head_rooms = {}
    for label, flags in (("standard_block", (True, True, True)),
                         ("legacy_block", (False, False, False)),
                         ("layernorm_only", (False, True, False)),
                         ("residual_only", (True, False, False)),
                         ("ffn_only", (False, False, True))):
        m = DKMModel(num_codes=ds.num_codes, embedding_path=args.embedding_path, arm="vanilla",
                     use_residual=flags[0], use_layernorm=flags[1], use_ffn=flags[2],
                     demo_dim=demo_dim, demo_channels=demo_channels,
                     race_encoding=args.race_encoding, tau_max=tau_max, **common).to(device).eval()
        head_rooms[label] = headroom(m, batches, args.s)

    domain = batch_domain(batches, tau_max)
    band = characterize_band(LogAgeFourier(M=args.age_M, p_min=args.age_p_min,
                                           p_max=args.age_p_max), M=args.age_M)
    legacy_band = characterize_band(LinearAgeFourier(M=args.age_M), M=args.age_M)
    struct = structural_checks(
        batches, ds.num_codes, args.embedding_path, demo_dim, demo_channels,
        args.race_encoding, tau_max,
        dict(use_residual=use_residual, use_layernorm=use_layernorm, use_ffn=use_ffn, **common))
    pic = pic_conditioning(args.pic_dir, args.vocab_path, tau_max, args.s, seed=args.seed)

    report = {
        "split": args.split, "n_batches": len(batches), "batch_size": args.batch_size,
        "device": str(device),
        "tau_max": tau_max, "tau_max_source": tau_source,
        "age_standardization": {"mean": age_mean, "sd": age_sd,
                                "applies_to": "demographic channel 0 only; age_years raw"},
        "batch_domain": domain,
        "headroom": head_rooms,
        "corpus": stats.to_json(),          # THE single corpus block
        "band_characterization": {"log_age": band, "legacy_linear": legacy_band},
        "structural": struct,
        "pic_conditioning": pic,
    }
    D.write_json(args.out, report)

    # ---- print everything and stop ---------------------------------------- #
    D.print_block("Phase 10 preflight  [MEASURE unless marked HARD]", [
        f"split={args.split}  batches={len(batches)}x{args.batch_size}  device={device}",
        f"tau_max={tau_max:.6f}   source: {tau_source}",
        f"age standardization: mean={age_mean:.4f} sd={age_sd:.4f} (demo channel 0 only)",
        f"report written to {args.out}",
    ])

    hl = [f"{'encoder block':<18}{'max|Dlogit|':>13}{'mean|Dlogit|':>14}"
          f"{'logit sd':>11}{'max/sd':>10}", "  " + "-" * 66]
    for label, hr in head_rooms.items():
        hl.append(f"{label:<18}{hr['max_abs_delta_logit']:>13.6f}"
                  f"{hr['mean_abs_delta_logit']:>14.6f}{hr['logit_sd']:>11.4f}"
                  f"{hr['max_delta_over_logit_sd']:>10.4f}")
    ref = head_rooms["standard_block"]["reference_pic_max_abs_delta_logit"]
    hl += [
        "  " + "-" * 66,
        f"reference: the same measurement on PIC gave max|Dlogit| = {ref}",
        "",
        "If maximally different kernels barely move the output, no arm can differ from any",
        "other and pretraining cannot answer the question. The ratio to the logit sd is the",
        "honest scale: a shift far below the spread the head already produces is invisible.",
        "",
        "The per-component rows are the actionable part. The kernel adds a bias of order",
        "||alpha||_1 to the pre-softmax scores; whether that bias survives depends on how",
        "large the QK term is beside it. Un-normalised 1024-d BGE embeddings give large QK",
        "logits, so a LayerNorm on the attention input changes the kernel's authority by more",
        "than the residual or the FFN do.",
    ]
    D.print_block("headroom: equal-norm decaying-vs-growing kernel on real batches  [MEASURE]", hl)

    sp_pad, sp_cau = stats.spread_padding_only, stats.spread_causal
    D.print_block("within-row tau spread  [MEASURE, from corpus_stats]", [
        "Softmax is invariant to a per-row constant, so within-row spread -- not the marginal",
        "tau histogram -- is what decides whether kernel SHAPE can matter.",
        "",
        "  quantile        : " + "  ".join(f"{q:>7}" for q in sp_pad),
        "  padding-only    : " + "  ".join(f"{v:7.3f}" for v in sp_pad.values()),
        "  causal (legacy) : " + "  ".join(f"{v:7.3f}" for v in sp_cau.values()),
        "",
        f"fraction of rows spread < 0.1 : {stats.frac_rows_spread_below_0p1:.4f}   "
        f"(causal: {stats.causal_frac_rows_spread_below_0p1:.4f})",
        "",
        "D4 removed causal masking. Under tril an early row sees only a handful of keys, so",
        "its lag spread collapses and the kernel has almost nothing to discriminate.",
        f"dt == 0 fraction (valid pairs): {stats.dt_zero_fraction:.4f}",
        f"timestamp resolution by magnitude (days): {stats.timestamp_resolution}",
        "",
        "As the LOADED batches exercise the frozen tau_max:",
        f"  tau_tilde range : [{domain['tau_tilde_min']:.4f}, {domain['tau_tilde_max']:.4f}]",
        f"  clamp rate      : {domain['clamp_rate']:.3e}",
    ])

    D.print_block("age support  [MEASURE, exact over full split]", [
        f"min={stats.event_age_min:.3f}  median={stats.event_age_median:.3f}  "
        f"max={stats.event_age_max:.3f}  mean={stats.event_age_mean:.3f}  sd={stats.event_age_sd:.3f}",
        f"integer-age fraction (events)          : {stats.event_integer_age_fraction:.5f}",
        f"fraction >= 89 (MIMIC censoring value) : {stats.event_age_ge_89_fraction:.4f}",
        f"events under 18                        : {stats.event_age_under_18_count} "
        f"({stats.event_age_under_18_fraction:.5f})",
        "band counts (events): " + "  ".join(
            f"{k}={v}" for k, v in stats.event_age_band_counts.items()),
        "",
        f"youngest observed event age is {stats.event_age_min:.1f} y; the pediatric range of",
        "Delta-alpha(a) is near-pure extrapolation after pretraining. train.json dumps it every epoch.",
    ])

    D.print_band_characterization(band, legacy_band)

    if pic is not None:
        D.print_block("M1  Chebyshev conditioning on the PIC tau_tilde distribution  [MEASURE]", [
            f"PIC dir            : {pic['pic_dir']}   n_pairs={pic['n_pairs']:,}",
            f"tau_tilde range    : [{pic['tau_tilde_min']:.4f}, {pic['tau_tilde_max']:.4f}]  "
            f"(occupies {pic['occupancy_fraction_of_domain']*100:.1f}% of [-1,1])",
            f"Chebyshev cond     : {pic['chebyshev_cond_no_constant']:.1f} (no T0)   "
            f"{pic['chebyshev_cond_with_constant']:.1f} (with T0)",
            f"MIMIC reference    : {pic['mimic_reference_chebyshev_cond_no_constant']} (no T0)",
            "",
            "PIC stays are ICU-length, so under the frozen pretrain tau_max the PIC lags sit in",
            "a narrow sub-interval near -1 where the Chebyshev basis is no longer orthogonal.",
            "If this cond number is 1e3 or worse, tau_max and the basis -- both frozen in the",
            "checkpoint -- should be reconsidered NOW, before pretraining.",
        ])
    else:
        D.print_block("M1  PIC conditioning  [MEASURE]",
                      [f"no PIC shards under {args.pic_dir}; skipped. Re-run with --pic_dir."])

    s = struct
    D.print_block("structural checks on real data  [HARD]", [
        f"INV-DEMO-SPLIT age_years is its own key      : {s['INV-DEMO-SPLIT']['age_years_is_own_key']}",
        f"INV-DEMO-SPLIT demo_dim                      : {s['INV-DEMO-SPLIT']['demo_dim']} "
        f"{s['INV-DEMO-SPLIT']['demo_channels']}",
        f"INV-DEMO-SPLIT age also present in demo[..0] : "
        f"{s['INV-DEMO-SPLIT']['age_present_in_demo_channel_0']}  (intended: every arm gets it)",
        f"INV-DEMO-SPLIT age channel standardized      : "
        f"{s['INV-DEMO-SPLIT']['age_channel_standardized_in_forward']}  (demo[..0] only; psi raw)",
        f"INV-ZERO-A kernel vs vanilla max|Dlogit|      : "
        f"{s['INV-ZERO-A']['kernel_vs_vanilla_max_abs']:.3e}  (bit-identical)",
        f"INV-ZERO-A random_constant vs vanilla        : "
        f"{s['INV-ZERO-A']['random_constant_vs_vanilla_max_abs']:.3e}  (bit-identical)",
        f"INV-ZERO-A additive vs vanilla               : "
        f"{s['INV-ZERO-A']['additive_vs_vanilla_max_abs']:.3e}  (float matmul noise, was ~3.2)",
        f"INV-ZERO-B additive concat columns zeroed    : "
        f"{s['INV-ZERO-B']['max_abs_delta']:.3e}",
    ])

    D.print_block("preflight complete", [
        "Nothing was trained. Read the headroom number against the within-row spread before",
        "committing to a multi-day pretrain.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
