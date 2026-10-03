"""Verify C04/C05/C06 architecture, grads, and artifact provenance. No retrain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAT))
sys.path.insert(0, str(ROOT))

import atomic  # noqa: E402
from atomic.config import experiment_configs  # noqa: E402
from atomic.models import build_atomic  # noqa: E402
from baselines.common.training import set_seed  # noqa: E402
from ladder.artifacts import file_sha256  # noqa: E402

ARTIFACT = ROOT / "artifacts" / "dtr_atomic_followup"
EXPS = (
    "C04_no_content_persistence",
    "C05_shared_beta_mixture",
    "C06_component_beta_mixture",
)
SCENARIO, ARM, SEED = "s2", "age_temporal", 0
FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def cfg_for(exp_id: str) -> dict:
    for cfg in experiment_configs():
        if cfg["experiment_id"] == exp_id:
            return cfg
    raise KeyError(exp_id)


def run_dir(exp: str) -> Path:
    return ARTIFACT / exp / SCENARIO / ARM / f"seed_{SEED}"


def make_batch(n_codes: int = 64, n_targets: int = 32):
    torch.manual_seed(0)
    bsz, encounters, codes = 4, 5, 6
    enc_codes = torch.randint(1, n_codes, (bsz, encounters, codes))
    mask = torch.ones(bsz, encounters, codes, dtype=torch.bool)
    tau = torch.rand(bsz, encounters) * 3 + 0.1
    pad = torch.zeros(bsz, encounters, dtype=torch.bool)
    pad[:, -1] = True
    tau[:, -1] = 0
    return {
        "enc_code_ids": enc_codes,
        "enc_code_mask": mask,
        "enc_tau": tau,
        "enc_padding_mask": pad,
        "age": torch.tensor([2.0, 7.0, 12.0, 17.0]),
        "labels": torch.rand(bsz, n_targets),
    }


def forward(model, batch):
    return model(
        batch["enc_code_ids"],
        batch["enc_code_mask"],
        batch["enc_tau"],
        batch["enc_padding_mask"],
        batch["age"],
        return_parts=True,
    )


def main() -> int:
    print("=" * 72)
    print("1) Model class + resolved config")
    print("=" * 72)
    models = {}
    cfgs = {}
    for exp in EXPS:
        cfg = cfg_for(exp)
        cfgs[exp] = cfg
        set_seed(0)
        model = build_atomic(cfg, 64, 32, age_temporal=True)
        models[exp] = model
        print(f"\n{exp}")
        print(f"  class: {type(model).__module__}.{type(model).__name__}")
        print(f"  variant/architecture: {model.variant}")
        print(
            "  resolved:"
            f" experiment_id={cfg['experiment_id']}"
            f" variant={cfg['variant']}"
            f" staged={cfg['staged']}"
            f" n_components={cfg.get('n_components')}"
            f" lambda_init={cfg.get('lambda_init')}"
            f" config_hash={cfg['config_hash'][:16]}"
        )

    print("\n" + "=" * 72)
    print("2) Trainable parameter names and counts")
    print("=" * 72)
    trainable = {}
    for exp, model in models.items():
        model.configure_arm_()
        rows = [(n, int(p.numel())) for n, p in model.named_parameters() if p.requires_grad]
        trainable[exp] = rows
        print(f"\n{exp}: n_tensors={len(rows)} n_params={sum(c for _, c in rows)}")
        for name, count in rows:
            mark = " <<<" if name in {"theta_k", "beta", "beta_k"} or name.startswith("f_mix") else ""
            print(f"  {name:50s} {count:8d}{mark}")

    c05 = {n for n, _ in trainable["C05_shared_beta_mixture"]}
    c06 = {n for n, _ in trainable["C06_component_beta_mixture"]}
    if "theta_k" not in c05:
        fail("C05 trainable set missing theta_k")
    else:
        ok("C05 has theta_k")
    if not any(n.startswith("f_mix") for n in c05):
        fail("C05 trainable set missing f_mix")
    else:
        ok("C05 has mixture-network parameters")
    if "beta" not in c05:
        fail("C05 trainable set missing shared beta")
    else:
        ok("C05 has shared beta")
    if "beta_k" not in c06:
        fail("C06 trainable set missing beta_k")
    else:
        ok("C06 has beta_k")

    print("\n" + "=" * 72)
    print("3) state_dict key-set differences")
    print("=" * 72)
    keys = {exp: set(models[exp].state_dict().keys()) for exp in EXPS}
    only_c05 = sorted(keys["C05_shared_beta_mixture"] - keys["C04_no_content_persistence"])
    only_c06 = sorted(keys["C06_component_beta_mixture"] - keys["C05_shared_beta_mixture"])
    only_shared = sorted(keys["C05_shared_beta_mixture"] - keys["C06_component_beta_mixture"])
    print(f"C05 - C04: {only_c05}")
    print(f"C06 - C05: {only_c06}")
    print(f"C05 - C06: {only_shared}")
    for required in ("theta_k", "f_mix.weight", "f_mix.bias", "beta"):
        if required not in only_c05:
            fail(f"expected {required} in C05-C04 key diff")
    if "beta_k" not in only_c06:
        fail("expected beta_k in C06-C05 key diff")
    if "beta" not in only_shared:
        fail("expected shared beta in C05-C06 key diff")
    if not FAILS:
        ok("state_dict key sets differ as expected")

    print("\n" + "=" * 72)
    print("4) Perturb theta_k / beta_k on one identical batch")
    print("=" * 72)
    batch = make_batch()
    set_seed(123)
    c05 = build_atomic(cfgs["C05_shared_beta_mixture"], 64, 32, age_temporal=True)
    set_seed(123)
    c06 = build_atomic(cfgs["C06_component_beta_mixture"], 64, 32, age_temporal=True)
    with torch.no_grad():
        c05.f_mix.weight.normal_(0, 0.05)
        c05.f_mix.bias.normal_(0, 0.05)
        c06.f_mix.weight.copy_(c05.f_mix.weight)
        c06.f_mix.bias.copy_(c05.f_mix.bias)
        c05.beta.fill_(0.8)
        c06.beta_k.copy_(torch.tensor([0.5, -0.3, 0.9]))
        c05.theta_k.copy_(torch.tensor([-0.2, 0.1, 0.4]))
        c06.theta_k.copy_(c05.theta_k)

    base05 = forward(c05, batch)["logits"].detach().clone()
    with torch.no_grad():
        c05.theta_k.add_(torch.tensor([1.5, -0.7, 0.3]))
    delta05 = float((forward(c05, batch)["logits"] - base05).abs().max())
    print(f"C05 max abs logit delta after theta_k perturb: {delta05:.6e}")
    if delta05 <= 1e-5:
        fail("C05 logits unchanged after theta_k perturb")
    else:
        ok("C05 theta_k perturb changes logits")

    base06 = forward(c06, batch)["logits"].detach().clone()
    saved = c06.beta_k.detach().clone()
    deltas = []
    outs = []
    for k in range(3):
        with torch.no_grad():
            c06.beta_k.copy_(saved)
            c06.beta_k[k] += 1.7
        out = forward(c06, batch)["logits"].detach()
        outs.append(out)
        deltas.append(float((out - base06).abs().max()))
    print(
        "C06 max abs logit delta after beta_k[k] perturb:"
        f" {[f'{x:.6e}' for x in deltas]}"
    )
    print(
        "C06 independence gaps:"
        f" |0-1|={float((outs[0]-outs[1]).abs().max()):.6e}"
        f" |0-2|={float((outs[0]-outs[2]).abs().max()):.6e}"
        f" |1-2|={float((outs[1]-outs[2]).abs().max()):.6e}"
    )
    if any(x <= 1e-5 for x in deltas):
        fail("C06 logits unchanged for some beta_k component")
    elif float((outs[0] - outs[1]).abs().max()) <= 1e-5:
        fail("C06 beta_k[0] and beta_k[1] produced identical logits")
    else:
        ok("C06 beta_k components change logits independently")

    print("\n" + "=" * 72)
    print("5) Backward grads")
    print("=" * 72)
    set_seed(7)
    c05 = build_atomic(cfgs["C05_shared_beta_mixture"], 64, 32, age_temporal=True)
    set_seed(7)
    c06 = build_atomic(cfgs["C06_component_beta_mixture"], 64, 32, age_temporal=True)
    with torch.no_grad():
        c05.f_mix.weight.normal_(0, 0.1)
        c05.f_mix.bias.normal_(0, 0.1)
        c06.f_mix.weight.copy_(c05.f_mix.weight)
        c06.f_mix.bias.copy_(c05.f_mix.bias)
        c05.beta.fill_(0.5)
        c06.beta_k.copy_(torch.tensor([0.4, -0.2, 0.6]))
        c05.theta_k.copy_(torch.tensor([0.2, -0.1, 0.3]))
        c06.theta_k.copy_(c05.theta_k)
    batch = make_batch()

    c05.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(forward(c05, batch)["logits"], batch["labels"])
    loss.backward()
    print("C05 grads:")
    for name in ("theta_k", "beta", "f_mix.weight", "f_mix.bias"):
        param = dict(c05.named_parameters())[name]
        if param.grad is None:
            fail(f"C05 {name} grad is None")
            continue
        score = float(param.grad.abs().sum())
        print(f"  {name}: abs_sum={score:.6e}")
        if score <= 0:
            fail(f"C05 {name} grad is zero")
    if not any(msg.startswith("C05 ") for msg in FAILS):
        ok("C05 theta_k, f_mix, and shared beta grads are nonzero")

    c06.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(forward(c06, batch)["logits"], batch["labels"])
    loss.backward()
    print("C06 grads:")
    if c06.beta_k.grad is None:
        fail("C06 beta_k grad is None")
    else:
        for k in range(3):
            score = float(c06.beta_k.grad[k].abs().sum())
            print(f"  beta_k[{k}]: abs_sum={score:.6e}")
            if score <= 0:
                fail(f"C06 beta_k[{k}] grad is zero")
        if not any(msg.startswith("C06 beta_k") for msg in FAILS):
            ok("C06 beta_k[k] grads are nonzero for each k")

    print("\n" + "=" * 72)
    print("6) Mixture weight checks")
    print("=" * 72)
    parts = forward(c05, batch)
    pi = parts["pi"]
    print(f"pi.shape={tuple(pi.shape)}")
    if pi.shape != (batch["age"].shape[0], batch["enc_tau"].shape[1], 3):
        fail(f"unexpected pi shape {tuple(pi.shape)}")
    hist = ~batch["enc_padding_mask"]
    sums = pi.sum(dim=-1)[hist]
    print(
        "sum_k pi on valid encounters:"
        f" mean={float(sums.mean()):.6f}"
        f" min={float(sums.min()):.6f}"
        f" max={float(sums.max()):.6f}"
    )
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        fail("pi rows do not sum to 1")
    valid = pi[hist]
    mean_pi = valid.mean(dim=0)
    std_pi = valid.std(dim=0)
    assign = valid.argmax(dim=-1)
    frac = torch.bincount(assign, minlength=3).float() / max(int(assign.numel()), 1)
    print(f"mean_pi={mean_pi.tolist()}")
    print(f"std_pi={std_pi.tolist()}")
    print(f"assignment_fraction={frac.tolist()}")
    if not FAILS or not any("pi" in msg for msg in FAILS):
        ok("pi shape and simplex constraints hold")

    print("\n" + "=" * 72)
    print("7) Artifact hashes")
    print("=" * 72)
    hashes = {}
    for exp in EXPS:
        directory = run_dir(exp)
        cfg_path = directory / "config.yaml"
        ckpt_path = directory / "checkpoint_best.pt"
        pred_path = directory / "predictions.parquet"
        for path in (cfg_path, ckpt_path, pred_path, directory / "manifest.json", directory / "metrics.json"):
            if not path.exists():
                fail(f"missing {path}")
        hashes[exp] = {
            "config": file_sha256(cfg_path),
            "checkpoint": file_sha256(ckpt_path),
            "predictions": file_sha256(pred_path),
        }
        print(f"\n{exp}")
        print(f"  dir: {directory}")
        print(f"  config.yaml  {hashes[exp]['config']}")
        print(f"  checkpoint   {hashes[exp]['checkpoint']}")
        print(f"  predictions  {hashes[exp]['predictions']}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        print(
            "  checkpoint contents:"
            f" variant={ckpt.get('config', {}).get('variant')}"
            f" theta_k={'theta_k' in sd}"
            f" beta_k={'beta_k' in sd}"
            f" f_mix={'f_mix.weight' in sd}"
            f" Atomic.beta={'beta' in sd}"
            f" base.beta={'base.beta' in sd}"
        )
        if "theta_k" in sd:
            print(f"    theta_k={sd['theta_k'].reshape(-1).tolist()}")
        if "beta_k" in sd:
            print(f"    beta_k={sd['beta_k'].reshape(-1).tolist()}")
        if "beta" in sd:
            print(f"    Atomic.beta={sd['beta'].reshape(-1).tolist()}")
        if "base.beta" in sd:
            print(f"    base.beta={sd['base.beta'].reshape(-1).tolist()}")
        if "f_mix.weight" in sd:
            print(
                "    f_mix.weight"
                f" abs_mean={float(sd['f_mix.weight'].abs().mean()):.6e}"
                f" abs_max={float(sd['f_mix.weight'].abs().max()):.6e}"
            )

    pred_hashes = [hashes[exp]["predictions"] for exp in EXPS]
    ckpt_hashes = [hashes[exp]["checkpoint"] for exp in EXPS]
    print("\nprediction hash equality:")
    print(f"  C04==C05: {pred_hashes[0] == pred_hashes[1]}")
    print(f"  C04==C06: {pred_hashes[0] == pred_hashes[2]}")
    print(f"  C05==C06: {pred_hashes[1] == pred_hashes[2]}")
    print("checkpoint hash equality:")
    print(f"  C04==C05: {ckpt_hashes[0] == ckpt_hashes[1]}")
    print(f"  C04==C06: {ckpt_hashes[0] == ckpt_hashes[2]}")
    print(f"  C05==C06: {ckpt_hashes[1] == ckpt_hashes[2]}")
    if len(set(pred_hashes)) < 3:
        fail("prediction hashes are not all distinct")
    else:
        ok("prediction hashes are not identical")

    print("\n" + "=" * 72)
    print("8) Provenance chain")
    print("=" * 72)
    for exp in EXPS:
        directory = run_dir(exp)
        metrics = json.loads((directory / "metrics.json").read_text())
        mechanism = json.loads((directory / "mechanism_metrics.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        ckpt = torch.load(directory / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        print(f"\n{exp}")
        print(f"  metrics.bce={metrics.get('bce')}")
        print(f"  predictions exist={(directory / 'predictions.parquet').exists()}")
        print(f"  manifest.checkpoint={manifest.get('checkpoint')}")
        print(f"  manifest.experiment_id={manifest.get('experiment_id')}")
        print(f"  checkpoint.config.experiment_id={ckpt['config'].get('experiment_id')}")
        print(f"  checkpoint.config.variant={ckpt['config'].get('variant')}")
        print(f"  mechanism.architecture={mechanism.get('architecture')}")
        if Path(manifest["checkpoint"]).resolve().parent != directory.resolve():
            fail(f"{exp}: checkpoint path escapes experiment directory")
        if manifest.get("experiment_id") != exp:
            fail(f"{exp}: manifest experiment_id mismatch")
        if ckpt["config"].get("experiment_id") != exp:
            fail(f"{exp}: checkpoint config experiment_id mismatch")
        if exp not in str(directory):
            fail(f"{exp}: run directory does not contain experiment id")
    if not any("Provenance" in msg or "manifest" in msg or "checkpoint config" in msg for msg in FAILS):
        ok("each artifact chain stays in its own experiment directory")

    print("\n" + "=" * 72)
    if FAILS:
        print(f"{len(FAILS)} verification failure(s):")
        for msg in FAILS:
            print(f"  - {msg}")
        return 1
    print("ALL VERIFICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
