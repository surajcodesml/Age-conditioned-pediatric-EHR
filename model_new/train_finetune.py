#!/usr/bin/env python3
"""Fine-tuning: binary disease classification from the shared pretrained backbone.

Two legacy defects are structurally impossible here.

D8 / INV-TMAX -- ``tau_max`` is read from the checkpoint and reused **bit-for-bit**. It is
never re-derived from the fine-tune corpus, and an explicit ``--tau_max`` that disagrees
with the checkpoint raises. Re-deriving it would silently change the meaning of every
learned coefficient.

D9 -- there is no ``return_repr_only``. The classification head sits on the pooled ``h``,
exactly as pretraining does, so the pooling-site age parameters carry gradient.

Every arm runs this file with identical flags apart from ``--arm`` and ``--run_name``, and
three further properties are enforced rather than assumed:

INV-FT-ARM -- the arm is **read** from the checkpoint, not asserted by the caller. A
``--arm`` that disagrees with the checkpoint raises. A mismatched arm silently invalidates
the whole ablation, so it cannot be a warning.

INV-FT-FROZEN -- ``tau_max``, the age standardization constants, the Fourier frequency
buffers, the race one-hot ordering and the Chebyshev degree ``s`` all come from the
checkpoint and are checked bit-identical after loading. Any recomputation on the fine-tune
corpus is a hard error.

INV-FT-ORDER -- the data order is hashed and the hash is compared against every sibling
arm's ``pic_config.json`` under the same run root, so a downstream difference between arms
cannot come from batch ordering. The train loader is driven by an explicitly owned
``torch.Generator``; the global RNG would give the arms different shuffles, because
constructing the age modules consumes a different number of draws per arm.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as _dt
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_new import diagnostics as D
from model_new.arms import ARMS
from model_new.data import RACE_LABELS, dataloader_worker_init, demo_layout
from model_new.data_finetune import (
    TensorizedFinetuneDataset, check_tau_max, make_finetune_collate,
)
from model_new.model import DKMModel
from model_new.optim import build_param_groups
from model_new.train import _git, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, default=None,
                   help="optional. The arm is read from the checkpoint; this flag exists "
                        "only so a disagreement is loud (INV-FT-ARM).")
    p.add_argument("--allow_arm_mismatch", action="store_true",
                   help="DECISION D2 shared-vanilla only: fine-tune --arm from a backbone "
                        "pretrained under a different arm. Recorded in pic_config.json.")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_root", type=Path, default=Path("model_new/run_finetune"))
    p.add_argument("--pretrained_ckpt", type=Path, required=True)
    p.add_argument("--tensorized_dir", type=Path, required=True,
                   help="directory containing train/ val/ test/ shard_*.npz")
    p.add_argument("--embedding_path", type=Path,
                   default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--tau_max", type=float, default=None,
                   help="must equal the checkpoint value; provided only so a mismatch is loud")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_age", type=float, default=1e-3)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--band_table", choices=sorted(D.BAND_TABLES), default="adult",
                   help="age-band table for every stratified metric in this run. Use "
                        "'pediatric' for PIC (D-1).")
    p.add_argument("--task_name", type=str, default=None,
                   help="the fine-tune task, recorded in pic_config.json")
    p.add_argument("--primary_task", type=str, default=None,
                   help="the task the headline result comes from, DECLARED BEFORE the run")
    p.add_argument("--primary_endpoint", type=str, default="val_auprc",
                   help="the endpoint the headline result comes from, DECLARED BEFORE "
                        "the run")
    p.add_argument("--vocab_choice", type=str, default=None,
                   help="free text recording DECISION D3 for this run, e.g. "
                        "'pic_bge_table' or 'reindexed_into_mimic_vocab'")
    p.add_argument("--deviation", action="append", default=[],
                   help="repeatable; each entry lands in pic_config.deviations_from_pretrain")
    return p


def resolve_tau_max(ckpt: dict, override: float | None) -> float:
    """INV-TMAX. The checkpoint is the single source of truth."""
    if "tau_max" not in ckpt:
        raise AssertionError(
            "[INV-TMAX] checkpoint has no tau_max. It must be stored at pretrain time and "
            "reused verbatim; recomputing it from the fine-tune corpus would change the "
            "meaning of every learned coefficient."
        )
    ckpt_tau = float(ckpt["tau_max"])
    if override is not None and float(override) != ckpt_tau:
        raise AssertionError(
            f"[INV-TMAX] --tau_max={float(override)!r} disagrees with the checkpoint's "
            f"{ckpt_tau!r}. Refusing to run: tau_max must match bit-for-bit."
        )
    return ckpt_tau


def resolve_age_standardization(ckpt: dict, override: tuple[float, float] | None
                                ) -> tuple[float, float]:
    """INV-AGESTD. The demographic-age standardization constants come from the checkpoint and
    are reused verbatim, exactly like tau_max.

    Re-deriving (mean, sd) from the fine-tune corpus would put a PIC child near 0 -- PIC's own
    mean -- instead of near -3, its true position relative to the adult pretraining corpus,
    silently redefining what the demographic age feature means to demo_proj.
    """
    cfg = ckpt.get("config", {}).get("model", {})
    std = cfg.get("age_standardization")
    if std is None:
        raise AssertionError(
            "[INV-AGESTD] checkpoint has no age_standardization constants. They must be "
            "stored at pretrain time and reused verbatim."
        )
    mean, sd = float(std["mean"]), float(std["sd"])
    if override is not None and (float(override[0]) != mean or float(override[1]) != sd):
        raise AssertionError(
            f"[INV-AGESTD] override (mean={override[0]}, sd={override[1]}) disagrees with the "
            f"checkpoint's (mean={mean}, sd={sd}). Refusing to run: constants must match."
        )
    return mean, sd


def checkpoint_arm(ckpt: dict) -> str:
    """The arm the pretrained weights were produced by. Read, never assumed."""
    top = ckpt.get("arm")
    inner = ckpt.get("config", {}).get("arm")
    found = {a for a in (top, inner) if a is not None}
    if not found:
        raise AssertionError(
            "[INV-FT-ARM] the checkpoint records no arm (neither ckpt['arm'] nor "
            "ckpt['config']['arm']). The arm cannot be supplied by the caller: it is a "
            "property of the pretrained weights.")
    if len(found) > 1:
        raise AssertionError(
            f"[INV-FT-ARM] the checkpoint disagrees with itself about the arm: "
            f"ckpt['arm']={top!r} vs ckpt['config']['arm']={inner!r}")
    arm = found.pop()
    if arm not in ARMS:
        raise AssertionError(f"[INV-FT-ARM] checkpoint arm {arm!r} is not one of {ARMS}")
    return arm


def resolve_arm_from_checkpoint(ckpt: dict, override: str | None, *,
                                allow_mismatch: bool = False) -> str:
    """INV-FT-ARM. The arm is a property of the checkpoint, not of the command line.

    ``train.py`` writes the arm in two places -- the top-level ``arm`` field and
    ``config["arm"]`` -- and both are consulted. A ``--arm`` that disagrees raises by
    default: running the ``kernel`` fine-tune script against the ``vanilla`` backbone would
    produce a complete, plausible run whose numbers belong to neither arm, and nothing
    downstream could detect it.

    ``allow_mismatch`` is the shared-vanilla design of DECISION D2, where every arm is
    fine-tuned *from* the vanilla backbone on purpose. It is not a softening of the
    invariant: the mismatch must be named on the command line, the effective arm becomes
    the override, and both arms are recorded in ``pic_config.json`` as a deviation --
    exactly the treatment ``eval_pretrain`` gives ``--allow_config_diff``.
    """
    arm = checkpoint_arm(ckpt)
    if override is None or override == arm:
        return arm
    if not allow_mismatch:
        raise AssertionError(
            f"[INV-FT-ARM] --arm {override!r} disagrees with the checkpoint's {arm!r}. "
            f"Refusing to run: a mismatched arm silently invalidates the ablation. If this "
            f"is the shared-vanilla design (DECISION D2), pass --allow_arm_mismatch, which "
            f"records both arms in pic_config.json.")
    return override


def _fourier_buffer_keys(model: DKMModel) -> dict[str, tuple[str, str]]:
    """-> ``{site: (frequencies key, periods key)}`` in ``state_dict`` coordinates.

    Membership comes from the module graph -- ``diagnostics.age_conditioner_sites`` and an
    identity comparison against ``named_modules`` -- never from matching on the shape of a
    parameter name (D6). The names are then used only to index the checkpoint's dict.
    """
    prefix_of = {id(mod): name for name, mod in model.named_modules()}
    out: dict[str, tuple[str, str]] = {}
    for site, cond in D.age_conditioner_sites(model):
        p = prefix_of.get(id(cond.fourier))
        if p is None:
            raise AssertionError(f"[INV-FT-FROZEN] site {site} has a Fourier module that is "
                                 f"not registered on the model")
        out[site] = (f"{p}.frequencies", f"{p}.periods")
    return out


def assert_frozen_constants(model: DKMModel, ckpt: dict) -> dict:
    """INV-FT-FROZEN. Everything frozen at pretraining is bit-identical after loading.

    Five quantities, each of which silently redefines a learned weight if it drifts:
    ``tau_max`` (the domain every Chebyshev coefficient is defined against), the age
    standardization constants (what ``demo_proj`` channel 0 means), the Fourier frequency
    buffers (what ``psi(a)`` is), the race one-hot ordering (which column is which group),
    and the degree ``s`` (how many coefficients there are). Recomputing any of them on PIC
    is a hard error, not a warning: the failure mode is a run that finishes and reports
    numbers, not one that crashes.
    """
    cfg = ckpt.get("config", {}).get("model", {})
    state = ckpt["model_state_dict"]
    report: dict[str, Any] = {}

    ckpt_tau = float(ckpt["tau_max"])
    site_taus = {name: float(site.kernel.tau_max) for name, site in model.kernel_sites()}
    bad = {k: v for k, v in site_taus.items() if v != ckpt_tau}
    if bad:
        raise AssertionError(
            f"[INV-FT-FROZEN] tau_max is not bit-identical to the checkpoint's "
            f"{ckpt_tau!r} at {bad}. Every learned coefficient is defined against this "
            f"domain; a different value redefines alpha rather than failing.")
    if "tau_max" in cfg and float(cfg["tau_max"]) != ckpt_tau:
        raise AssertionError(
            f"[INV-FT-FROZEN] the checkpoint's config says tau_max={cfg['tau_max']!r} but "
            f"its tau_max field says {ckpt_tau!r}")
    report["tau_max"] = {"value": ckpt_tau, "sites": site_taus, "source": "checkpoint"}

    std = cfg.get("age_standardization") or ckpt.get("age_standardization")
    if std is None:
        raise AssertionError("[INV-FT-FROZEN] the checkpoint carries no age_standardization")
    if float(model.age_mean) != float(std["mean"]) or float(model.age_sd) != float(std["sd"]):
        raise AssertionError(
            f"[INV-FT-FROZEN] age standardization ({float(model.age_mean)}, "
            f"{float(model.age_sd)}) is not bit-identical to the checkpoint's "
            f"({std['mean']}, {std['sd']}). Re-deriving these on PIC would put a child at "
            f"~0 -- PIC's own mean -- instead of ~-3.5 relative to the adult corpus.")
    report["age_standardization"] = {"mean": float(std["mean"]), "sd": float(std["sd"]),
                                     "source": "checkpoint"}

    fourier: dict[str, dict] = {}
    buffers = dict(model.named_buffers())
    for site, (fk, pk) in _fourier_buffer_keys(model).items():
        for key in (fk, pk):
            if key not in state:
                raise AssertionError(
                    f"[INV-FT-FROZEN] the checkpoint has no buffer {key!r}, so the Fourier "
                    f"band was rebuilt from defaults rather than restored (INV-FROZEN)")
            got, want = buffers[key].detach().cpu(), state[key].detach().cpu()
            if got.shape != want.shape or not bool(torch.equal(got, want)):
                raise AssertionError(
                    f"[INV-FT-FROZEN] buffer {key!r} differs from the checkpoint. psi(a) "
                    f"would mean something different at fine-tune than at pretraining.")
            if buffers[key].requires_grad:
                raise AssertionError(f"[INV-FT-FROZEN] buffer {key!r} requires grad")
        fourier[site] = {"frequencies_key": fk, "periods_key": pk,
                         "n": int(buffers[fk].numel())}
    report["fourier_buffers"] = fourier

    ckpt_channels = tuple(cfg.get("demo_channels") or ())
    if not ckpt_channels:
        raise AssertionError("[INV-FT-FROZEN] the checkpoint records no demo_channels, so "
                             "the race one-hot ordering cannot be read from it")
    if tuple(model.demo_channels) != ckpt_channels:
        raise AssertionError(
            f"[INV-FT-FROZEN] demographic channel ordering {tuple(model.demo_channels)} "
            f"differs from the checkpoint's {ckpt_channels}; a permuted one-hot silently "
            f"relabels every race group for demo_proj.")
    expected = ("age_years", "sex") + tuple(f"race_{r}" for r in RACE_LABELS)
    if cfg.get("race_encoding", "one_hot") == "one_hot" and ckpt_channels != expected:
        raise AssertionError(
            f"[INV-FT-FROZEN] the checkpoint's one-hot ordering {ckpt_channels} is not the "
            f"ordering data.RACE_LABELS produces {expected}; the encoder that wrote the "
            f"shards and the model that reads them disagree.")
    report["race_encoding"] = {"encoding": cfg.get("race_encoding", "one_hot"),
                               "channels": list(ckpt_channels), "source": "checkpoint"}

    ckpt_s = int(cfg["s"])
    site_s = {name: int(site.kernel.s) for name, site in model.kernel_sites()}
    if int(model.s) != ckpt_s or any(v != ckpt_s for v in site_s.values()):
        raise AssertionError(
            f"[INV-FT-FROZEN] Chebyshev degree s={int(model.s)} / sites {site_s} disagrees "
            f"with the checkpoint's {ckpt_s}")
    for name, site in model.kernel_sites():
        if int(site.alpha_base.numel()) != ckpt_s:
            raise AssertionError(
                f"[INV-FT-FROZEN] site {name} has {int(site.alpha_base.numel())} "
                f"coefficients, not s={ckpt_s} (INV-BASIS: no constant term)")
    report["s"] = {"value": ckpt_s, "sites": site_s, "source": "checkpoint"}
    return report


# --------------------------------------------------------------------------- #
# INV-FT-ORDER -- data order, hashed and compared across arms                 #
# --------------------------------------------------------------------------- #
def train_order_hash(n_examples: int, seed: int, epochs: int) -> dict:
    """A digest of the shuffled training order the four arms must share.

    ``RandomSampler`` draws ``torch.randperm(n, generator=g)`` once per epoch. The same
    generator object is handed to the ``DataLoader`` below, so seeding it here and replaying
    the draws gives the order the run will actually consume -- and, more to the point, gives
    an order that cannot differ between arms. Without an owned generator the sampler falls
    back to the global RNG, whose state depends on how many draws constructing the age
    modules consumed, which is arm-dependent.
    """
    g = torch.Generator().manual_seed(int(seed))
    h = hashlib.blake2b(digest_size=16)
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(int(n_examples), generator=g).numpy().astype(np.int64)
        h.update(np.ascontiguousarray(perm).tobytes())
    return {"hash": h.hexdigest(), "n_examples": int(n_examples), "seed": int(seed),
            "epochs": int(epochs),
            "definition": "blake2b-128 over torch.randperm(n, generator=Generator(seed)) "
                          "replayed once per epoch"}


class FinetuneBatchOrderHash:
    """A digest of the batch sequence an evaluation pass actually consumed.

    The fine-tune twin of ``eval_pretrain.BatchOrderHash``: same role, same guarantee,
    different batch contract (``labels [B]`` instead of ``target_codes [B, |V|]``).
    ``shuffle=False`` makes cross-arm equality true by construction; the hash is what turns
    "by construction" into "checked", and ``eval_finetune`` asserts it per arm.
    """

    DEFINITION = ("blake2b-128 over (lengths, code_indices, labels) of every batch in "
                  "order, shuffle=False")

    def __init__(self) -> None:
        self._h = hashlib.blake2b(digest_size=16)
        self.n_batches = 0
        self.n_rows = 0

    def update(self, batch: dict) -> None:
        self.n_batches += 1
        self.n_rows += int(batch["lengths"].shape[0])
        self._h.update(np.ascontiguousarray(batch["lengths"].cpu().numpy()).tobytes())
        self._h.update(np.ascontiguousarray(batch["code_indices"].cpu().numpy()).tobytes())
        self._h.update(np.ascontiguousarray(batch["labels"].cpu().numpy()).tobytes())

    @property
    def hexdigest(self) -> str:
        return self._h.hexdigest()

    def to_json(self) -> dict:
        return {"hash": self.hexdigest, "n_batches": self.n_batches,
                "n_rows": self.n_rows, "definition": self.DEFINITION}


def eval_order_hash(loader) -> dict:
    """A digest of a deterministic (``shuffle=False``) evaluation pass."""
    h = FinetuneBatchOrderHash()
    for batch in loader:
        h.update(batch)
    return h.to_json()


def assert_order_matches_siblings(run_root: Path, run_name: str, arm: str,
                                  hashes: dict) -> dict:
    """INV-FT-ORDER. Compare against every sibling arm's ``pic_config.json``.

    A sibling is any other run directory under the same root that carries a
    ``pic_config.json`` for a different arm on the same task. If its data-order hashes
    differ, the arms were not fed the same data in the same order and no cross-arm
    difference is attributable to the arm.
    """
    siblings: dict[str, dict] = {}
    root = Path(run_root)
    for cfg_path in sorted(root.glob("*/pic_config.json")):
        if cfg_path.parent.name == run_name:
            continue
        try:
            blob = json.loads(cfg_path.read_text())
        except Exception:
            continue
        other = blob.get("data_order", {})
        if not other or blob.get("task") != hashes["task"]:
            continue
        siblings[blob.get("arm", cfg_path.parent.name)] = {
            "run": cfg_path.parent.name, "data_order": other}
        mismatched = [k for k in ("train", "val", "test")
                      if k in hashes["hashes"] and k in other.get("hashes", {})
                      and hashes["hashes"][k]["hash"] != other["hashes"][k]["hash"]]
        if mismatched:
            raise AssertionError(
                f"[INV-FT-ORDER] arm={arm} consumes a different data order than "
                f"{cfg_path.parent.name} on {mismatched}. The arms are not being compared "
                f"on the same data, so a downstream difference cannot be attributed to the "
                f"arm. Check --seed and --batch_size.")
    return {"n_siblings_checked": len(siblings),
            "siblings": {k: v["run"] for k, v in siblings.items()}}


def load_backbone(model: DKMModel, state: dict, arm: str) -> dict:
    """Load the shared pretrain backbone. Missing/unexpected keys must be age modules or the
    head, never backbone weights.

    The pretrain head predicts |V| codes and the fine-tune head predicts one logit, so its
    keys are dropped before loading: ``strict=False`` tolerates *absent* keys but still
    raises on a shape mismatch for a key that is present.
    """
    dropped = sorted(k for k in state if k.startswith("head."))
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    incompatible = model.load_state_dict(state, strict=False)
    age_ids = {id(p) for p in model.age_parameters()}
    age_keys = {name for name, p in model.named_parameters() if id(p) in age_ids}
    age_prefixes = tuple(sorted({k.rsplit(".", 2)[0] for k in age_keys})) or ("__none__",)

    def ok(key: str) -> bool:
        return key.startswith("head.") or key.startswith(age_prefixes) or ".age." in key

    bad_missing = [k for k in incompatible.missing_keys if not ok(k)]
    bad_unexpected = [k for k in incompatible.unexpected_keys if not ok(k)]
    if bad_missing or bad_unexpected:
        raise AssertionError(
            f"[HARD] backbone weights did not transfer for arm={arm}. "
            f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}"
        )
    return {"missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "dropped_pretrain_head_keys": dropped}


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, bands=None) -> dict:
    was_training = model.training
    model.eval()
    logits, labels, ages, losses = [], [], [], []
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        with ctx:
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(out["logits"].float(), batch["labels"])
        losses.append(float(loss))
        logits.append(out["logits"].float().cpu())
        labels.append(batch["labels"].cpu())
        ages.append(out["age_last"].cpu())
    if was_training:
        model.train()
    y = torch.cat(labels).numpy()
    p = torch.sigmoid(torch.cat(logits)).numpy()
    a = torch.cat(ages)
    res = {"loss": float(np.mean(losses)) if losses else float("nan"), "n": int(y.size)}
    res.update(_binary_metrics(y, p))
    idx = D.band_index(a, bands)
    res["by_band"] = {}
    for i, name in enumerate(D.band_names(bands)):
        sel = idx == i
        entry = {"n": int(sel.sum()), "pos": int(y[sel].sum()) if sel.any() else 0}
        if sel.sum() > 0:
            entry.update(_binary_metrics(y[sel], p[sel]))
        res["by_band"][name] = entry
    return res


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out: dict[str, float] = {"prevalence": float(y.mean()) if y.size else float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if 0 < y.sum() < y.size:
            out["auroc"] = float(roc_auc_score(y, p))
            out["auprc"] = float(average_precision_score(y, p))
    except Exception:
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    bands = D.resolve_bands(args.band_table)

    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
    # D-2 / INV-FT-ARM.
    ckpt_arm = checkpoint_arm(ckpt)
    arm = resolve_arm_from_checkpoint(ckpt, args.arm,
                                      allow_mismatch=args.allow_arm_mismatch)
    tau_max = resolve_tau_max(ckpt, args.tau_max)
    age_mean, age_sd = resolve_age_standardization(ckpt, None)
    pre_cfg = ckpt.get("config", {}).get("model", {})
    race_encoding = pre_cfg.get("race_encoding", "one_hot")
    demo_dim, demo_channels = demo_layout(race_encoding)
    if pre_cfg.get("demo_dim") not in (None, demo_dim):
        raise AssertionError(
            f"[HARD] checkpoint demo_dim={pre_cfg.get('demo_dim')} but this race_encoding "
            f"gives {demo_dim}; the demographic layout must be identical across pretrain and "
            f"fine-tune and across arms.")

    collate = make_finetune_collate(race_encoding)
    loader_kw = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                     worker_init_fn=dataloader_worker_init,
                     persistent_workers=args.num_workers > 0)
    splits = {name: TensorizedFinetuneDataset(args.tensorized_dir / name,
                                              max_seq_len=args.max_seq_len)
              for name in ("train", "val", "test")
              if (args.tensorized_dir / name).exists()}
    if "train" not in splits:
        raise FileNotFoundError(f"no train/ split under {args.tensorized_dir}")
    # D-4 / INV-FT-ORDER. An OWNED generator, not the global RNG: constructing the age
    # modules consumes a different number of global draws per arm, so with the default
    # sampler the four arms would shuffle differently and every cross-arm difference would
    # be confounded with batch order.
    train_gen = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True,
                              drop_last=False, generator=train_gen, **loader_kw)
    eval_loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, **loader_kw)
                    for k, v in splits.items() if k != "train"}

    # DECISION D3 option (b): build from --embedding_path (PIC table), not the pretrain
    # vocab size. The checkpoint's embedding_table is [30637, 1024]; the PIC table is
    # [2200, 1024], so num_codes must come from the table we are about to use.
    emb_table = DKMModel._load_embedding_table(args.embedding_path, None)
    num_codes = int(emb_table.shape[0]) - 2
    model = DKMModel(
        num_codes=num_codes, embedding_table=emb_table, arm=arm, seed=args.seed,
        d_model=pre_cfg.get("d_model", 256), n_layers=pre_cfg.get("n_layers", 1),
        n_heads=pre_cfg.get("n_heads", 1), use_residual=pre_cfg.get("use_residual", True),
        use_layernorm=pre_cfg.get("use_layernorm", True), use_ffn=pre_cfg.get("use_ffn", True),
        s=pre_cfg.get("s", 5), tau_max=tau_max,
        age_M=pre_cfg.get("fourier", {}).get("M", 16),
        age_p_min=pre_cfg.get("fourier", {}).get("p_min", 0.15),
        age_p_max=pre_cfg.get("fourier", {}).get("p_max", 6.0),
        age_hidden=pre_cfg.get("age_hidden", 64),
        gen_final_bias=pre_cfg.get("gen_final_bias", False),
        center_delta_alpha=pre_cfg.get("center_delta_alpha", False),
        demo_dim=demo_dim, demo_channels=demo_channels, race_encoding=race_encoding,
        demo_hidden=pre_cfg.get("demo_hidden", 64), age_mean=age_mean, age_sd=age_sd,
        task="classification",
    )
    # Same substitution eval_finetune / pic_audit use: do not relax load_backbone; replace
    # the mismatched buffer explicitly and record it (INV-FROZEN does not hold for the table).
    state = dict(ckpt["model_state_dict"])
    emb_substituted = False
    if ("embedding_table" in state
            and tuple(state["embedding_table"].shape) != tuple(model.embedding_table.shape)):
        state["embedding_table"] = model.embedding_table.detach().clone()
        emb_substituted = True
    load_info = load_backbone(model, state, arm)
    load_info["embedding_table_substituted_not_restored"] = emb_substituted
    if emb_substituted:
        load_info["inv_frozen_note"] = (
            "INV-FROZEN's 'restored from the checkpoint rather than rebuilt' clause does "
            "not hold for embedding_table under DECISION D3 option (b); the table is a "
            "different matrix over a different vocabulary.")
    model.to(device)

    # D-3 / INV-FT-FROZEN. Everything frozen at pretraining, checked bit-identical AFTER
    # the load. The two older single-quantity checks stay: they name their own invariant
    # in the failure message, which is what a reader of the log needs.
    if abs(model.tau_max - tau_max) > 0:
        raise AssertionError(f"[INV-TMAX] model tau_max={model.tau_max} != checkpoint {tau_max}")
    if float(model.age_mean) != age_mean or float(model.age_sd) != age_sd:
        raise AssertionError(
            f"[INV-AGESTD] model age std ({float(model.age_mean)}, {float(model.age_sd)}) != "
            f"checkpoint ({age_mean}, {age_sd})")
    frozen = assert_frozen_constants(model, ckpt)

    groups, group_report = build_param_groups(model, args.lr_backbone, args.lr_age, args.lr_head)
    opt = torch.optim.Adam(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    theta0 = D.snapshot_parameters(groups)

    domain = check_tau_max(splits["train"], tau_max, n_samples=min(2000, len(splits["train"])),
                           seed=args.seed)
    config = {
        "run_id": args.run_name, "arm": arm, "seed": args.seed,
        "pretrain_arm": ckpt_arm,
        "arm_source": ("--arm (checkpoint says %r; --allow_arm_mismatch was given, the "
                       "shared-vanilla design of DECISION D2)" % ckpt_arm
                       if arm != ckpt_arm else
                       "checkpoint (INV-FT-ARM); --arm was "
                       + (f"{args.arm!r} and agrees" if args.arm else "not supplied")),
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "model": model.config_dict(),
        "params": model.parameter_report(),
        "optim": {**group_report, "batch_size": args.batch_size, "epochs": args.epochs,
                  "optimizer": "Adam"},
        "data": {"tensorized_dir": str(args.tensorized_dir),
                 "split_sizes": {k: len(v) for k, v in splits.items()},
                 "race_encoding": race_encoding},
        "tau_max": tau_max,
        "tau_max_source": ckpt.get("tau_max_source"),
        "tau_domain_check": domain,
        "state_dict_load": load_info,
        "frozen_constants": frozen,
        "band_table": args.band_table,
    }
    D.write_json(run_dir / "config.json", config)

    # D-5 / D-4. pic_config.json is written ONCE, before the first step, and records the
    # things a reader needs to know were decided in advance rather than after the numbers.
    task_name = args.task_name or Path(args.tensorized_dir).name
    order = {"task": task_name,
             "hashes": {"train": train_order_hash(len(splits["train"]), args.seed,
                                                  args.epochs)}}
    for name, ld in eval_loaders.items():
        order["hashes"][name] = eval_order_hash(ld)
    sibling_report = assert_order_matches_siblings(Path(args.run_root), args.run_name, arm,
                                                   order)
    pic_config = {
        "written_before_first_step": True,
        "timestamp": config["timestamp"],
        "task": task_name,
        "arm": arm,
        "pretrain_arm": ckpt_arm,
        "arm_source": config["arm_source"],
        "backbone_design": "arm_matched" if arm == ckpt_arm else "shared_backbone",
        "source_checkpoint": {
            "path": str(Path(args.pretrained_ckpt).resolve()),
            "config_hash": D.config_hash(ckpt.get("config", {})),
            "pretrain_run_id": ckpt.get("config", {}).get("run_id"),
            "epoch": ckpt.get("epoch"),
        },
        "tau_max": {"value": tau_max, "source": ckpt.get("tau_max_source"),
                    "frozen_in_checkpoint": True},
        "vocabulary_and_embeddings": {
            "embedding_path": str(args.embedding_path),
            "num_codes": int(model.num_codes),
            "embedding_dim": int(model.embedding_dim),
            "choice": args.vocab_choice,
            "embedding_table_substituted_not_restored": emb_substituted,
            "note": "DECISION D3. Recorded, not inferred: if --vocab_choice is null the "
                    "run did not declare which transfer route it used.",
        },
        "race_encoding": frozen["race_encoding"],
        "age_standardization": frozen["age_standardization"],
        "band_table": {"name": args.band_table,
                       "bands": {n: [lo, hi] for n, lo, hi in bands}},
        "optimizer_groups": {
            "lrs": {k: group_report[k] for k in
                    ("lr_backbone", "lr_age", "lr_head")},
            "n_tensors": group_report["n_tensors"],
            "n_params": group_report["n_params"],
        },
        "primary_task": args.primary_task,
        "primary_endpoint": args.primary_endpoint,
        "primary_declared_before_run": True,
        "data_order": order,
        "data_order_siblings_checked": sibling_report,
        "deviations_from_pretrain": list(args.deviation) + ([
            f"arm {arm} fine-tuned from a backbone pretrained as {ckpt_arm} "
            f"(--allow_arm_mismatch; shared-backbone design)"] if arm != ckpt_arm else []) + ([
            "DECISION D3 option (b): embedding_table substituted from --embedding_path; "
            "not restored from the pretrain checkpoint (shape mismatch)"]
            if emb_substituted else []) + [
            "task head: |V|-way code BCE -> a single logit (binary classification)",
            f"lr_backbone {args.lr_backbone} (pretraining used "
            f"{ckpt.get('config', {}).get('optim', {}).get('lr_backbone')})",
            f"batch_size {args.batch_size} (pretraining used "
            f"{ckpt.get('config', {}).get('optim', {}).get('batch_size')})",
            f"corpus: {task_name} at {args.tensorized_dir} (pretraining used "
            f"{ckpt.get('config', {}).get('data', {}).get('paths', {}).get('tensorized_dir')})",
        ],
    }
    D.write_json(run_dir / "pic_config.json", pic_config)

    D.print_config_summary(config)
    D.print_kv("INV-TMAX / fine-tune tau domain  [MEASURE]", domain)
    D.print_block("fine-tune declarations  [written before the first step]", [
        f"arm            : {arm}   (read from the checkpoint, INV-FT-ARM)",
        f"task           : {task_name}",
        f"primary_task   : {args.primary_task}   primary_endpoint: {args.primary_endpoint}",
        f"band table     : {args.band_table}  ({', '.join(D.band_names(bands))})",
        f"vocab choice   : {args.vocab_choice}",
        f"frozen         : tau_max, age (mean, sd), Fourier buffers, race ordering, s "
        f"-- all bit-identical to the checkpoint (INV-FT-FROZEN)",
        "data order hash: " + "  ".join(f"{k}={v['hash']}"
                                        for k, v in order["hashes"].items()),
        f"siblings checked: {sibling_report['n_siblings_checked']} "
        f"{sibling_report['siblings'] or ''}   (INV-FT-ORDER)",
        f"written        : {run_dir / 'pic_config.json'}",
    ])

    records: list[dict] = []
    best, bad_epochs = -np.inf, 0
    t_start = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        model.reset_clamp_stats()
        running, n_batches = 0.0, 0
        diag_batch = None
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            if diag_batch is None:
                diag_batch = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in batch.items()}
            ctx = (torch.amp.autocast(device_type="cuda", enabled=True) if use_amp
                   else nullcontext())
            with ctx:
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(out["logits"].float(), batch["labels"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], args.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss)
            n_batches += 1

        ages_flat = diag_batch["age_years"][diag_batch["attention_mask"]].detach()
        record = {
            "epoch": epoch, "wall_clock_s": time.time() - t_start,
            "train_loss": running / max(1, n_batches),
            "eval": {k: evaluate(model, ld, device, use_amp, bands=bands)
                     for k, ld in eval_loaders.items()},
            "alpha": D.alpha_diagnostics(model, ages_flat, bands=bands),
            "delta_alpha_grid": D.delta_alpha_grid(model),
            "w_curves": D.w_curves(model),
            "param_drift": D.parameter_drift(groups, theta0),
            "clamp_rate": D.clamp_rates(model),
        }
        records.append(record)
        D.append_train_json(run_dir / "train.json", records)
        D.print_finetune_epoch(record, bands=bands)

        ckpt_payload = {
            "epoch": epoch, "arm": arm, "seed": args.seed,
            "model_state_dict": model.state_dict(), "tau_max": tau_max,
            "age_standardization": {"mean": age_mean, "sd": age_sd},
            "config": config,
        }
        torch.save(ckpt_payload, run_dir / f"epoch_{epoch:03d}.pt")

        score = record["eval"].get("val", {}).get("auprc", -np.inf)
        if score > best:
            best, bad_epochs = score, 0
            torch.save(ckpt_payload, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                D.print_block("early stop", [f"no val AUPRC improvement in {args.patience} "
                                             f"epochs; best={best:.6f}"])
                break

    D.print_block("done", [f"run_dir: {run_dir}", f"best val AUPRC: {best:.6f}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
