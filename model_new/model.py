#!/usr/bin/env python3
"""``DKMModel`` -- one class, arm-gated.

    forward:
      1. assert demographics width and the presence of a separate ``age_years`` key
      2. Encoder -> E [B, L, d_model]
      3. AttentionPooling -> h [B, d_model]
      4. u = concat([h, demo_proj(demo_last)] + ([age_delta] if arm == "additive" else []))
      5. head: Linear -> GELU -> Linear -> |V|, final bias -7.0

No ``time_params_predictor`` (D10: pretraining is code BCE only). No ``return_repr_only``
reachable from training (D9: the fine-tune head operates on the pooled ``h``, so
pooling-site age parameters are not gradient-dead). Representation extraction for the
frozen linear probe lives in :meth:`extract_representations`, which is evaluation-only and
is never called from ``train.py`` / ``train_finetune.py``.

Age reaches the model by exactly two routes and both are explicit:

  * ``demographics[..., 0]`` -- the demographic feature, present in **every** arm. This is
    the route age already has and the one DKM has to improve on. Removing it would make
    ``vanilla`` age-blind and the baseline a strawman.
  * ``batch["age_years"]`` -- its own tensor, the only thing any kernel site reads (D3).

No module reads age out of ``demographics``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from model_new.age_encoding import AgeConditioner
from model_new.arms import ArmConfig, assert_arm_invariants, resolve_arm
from model_new.basis import DEFAULT_S
from model_new.data import tau_from_timestamps
from model_new.encoder import Encoder
from model_new.pooling import AttentionPooling

__all__ = ["DKMModel", "PredictionHead"]


class PredictionHead(nn.Module):
    """``Linear -> GELU -> Linear -> out_dim``. Declares itself as the ``head`` optimizer group.

    ``hidden_dim`` is passed in rather than derived from ``in_dim``, and every arm is given
    the **widest** arm's value. Otherwise ``additive`` -- whose input is ``s`` columns wider --
    would also get a wider hidden layer, so the second ``Linear`` would differ in shape too
    and no amount of slicing could make the arms share an initialization.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 final_bias: float = -7.0) -> None:
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.final_bias = float(final_bias)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.apply_final_bias_()

    @torch.no_grad()
    def apply_final_bias_(self) -> None:
        self.net[-1].bias.fill_(self.final_bias)

    def head_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class DKMModel(nn.Module):
    def __init__(
        self,
        *,
        num_codes: int,
        embedding_path: str | Path | None = None,
        embedding_table: torch.Tensor | None = None,
        arm: str = "vanilla",
        seed: int = 0,
        d_model: int = 256,
        n_layers: int = 1,
        n_heads: int = 1,
        use_residual: bool = True,
        use_layernorm: bool = True,
        use_ffn: bool = True,
        ffn_mult: int = 4,
        s: int = DEFAULT_S,
        tau_max: float = 6.5,
        age_M: int = 16,
        age_p_min: float = 0.15,
        age_p_max: float = 6.0,
        age_hidden: int = 64,
        gen_final_bias: bool = False,
        center_delta_alpha: bool = False,
        demo_dim: int = 9,
        demo_channels: tuple[str, ...] = (),
        race_encoding: str = "one_hot",
        demo_hidden: int = 64,
        age_mean: float = 0.0,
        age_sd: float = 1.0,
        task: str = "pretrain",
    ) -> None:
        super().__init__()
        self.cfg: ArmConfig = resolve_arm(arm)
        assert_arm_invariants(self.cfg, center_delta_alpha=center_delta_alpha)

        self.num_codes = int(num_codes)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.use_residual, self.use_layernorm, self.use_ffn = (
            bool(use_residual), bool(use_layernorm), bool(use_ffn))
        self.ffn_mult = int(ffn_mult)
        self.s = int(s)
        self.demo_dim = int(demo_dim)
        self.demo_channels = tuple(demo_channels)
        self.race_encoding = str(race_encoding)
        self.demo_hidden = int(demo_hidden)
        self.age_hparams = dict(M=int(age_M), p_min=float(age_p_min), p_max=float(age_p_max),
                                hidden=int(age_hidden), log_age=True)
        self.gen_final_bias = bool(gen_final_bias)
        self.center_delta_alpha = bool(center_delta_alpha)
        self.seed = int(seed)

        # Fix C -- the age channel of the DEMOGRAPHIC vector is standardized inside forward,
        # using constants frozen from the pretraining corpus. Raw age (median ~56) beside
        # eight 0/1 race/sex channels would dominate demo_proj's input scale ~50x, so R1 --
        # the route DKM must beat -- would be a poorly-scaled function of age at init. These
        # are persistent buffers so they ride in the checkpoint and are reused verbatim at
        # fine-tune (INV-AGESTD): re-deriving from PIC would put a child at ~0 (PIC's own
        # mean) rather than ~-3 relative to the adult corpus, silently changing the feature.
        # age_years fed to psi stays RAW -- only this demographic channel is standardized.
        if not (age_sd > 0):
            raise ValueError(f"age_sd must be > 0, got {age_sd}")
        self.register_buffer("age_mean", torch.tensor(float(age_mean), dtype=torch.float32),
                             persistent=True)
        self.register_buffer("age_sd", torch.tensor(float(age_sd), dtype=torch.float32),
                             persistent=True)
        self.age_mean.requires_grad_(False)
        self.age_sd.requires_grad_(False)

        # D12 via the Fourier module, restated here so it fails before any allocation.
        if (2 * int(age_M)) % 2 != 0:
            raise AssertionError(f"[D12] age_emb_dim must be even, got {2 * int(age_M)}")

        # -- frozen code embeddings ------------------------------------------ #
        table = self._load_embedding_table(embedding_path, embedding_table)
        if table.shape[0] != self.num_codes + 2:
            raise AssertionError(
                f"[HARD] embedding_table.shape[0] must equal len(code_vocab) + 2 = "
                f"{self.num_codes + 2} exactly, got {table.shape[0]}"
            )
        self.register_buffer("embedding_table", table.float(), persistent=True)
        self.embedding_table.requires_grad_(False)
        self.embedding_dim = int(table.shape[1])

        age_kwargs = dict(age_M=age_M, age_p_min=age_p_min, age_p_max=age_p_max,
                          age_hidden=age_hidden, gen_final_bias=gen_final_bias,
                          center_delta_alpha=center_delta_alpha)

        self.encoder = Encoder(
            self.embedding_dim, self.d_model, n_layers=self.n_layers,
            use_residual=self.use_residual, use_layernorm=self.use_layernorm,
            use_ffn=self.use_ffn, ffn_mult=self.ffn_mult, n_heads=self.n_heads, s=self.s,
            tau_max=tau_max, generator_mode=self.cfg.kernel_generator_mode,
            use_out_proj=self.n_heads > 1, **age_kwargs,
        )
        self.pooling = AttentionPooling(
            self.d_model, s=self.s, tau_max=tau_max,
            generator_mode=self.cfg.kernel_generator_mode, **age_kwargs,
        )

        # Additive arm: the generator's output is concatenated to h (draft), NOT added as a
        # 1024-d delta to the code embeddings (the older internal guide's variant).
        self.additive_age = (
            AgeConditioner(out_dim=self.s, M=age_M, p_min=age_p_min, p_max=age_p_max,
                           hidden_dim=age_hidden, mode="real", final_bias=gen_final_bias,
                           center_delta_alpha=center_delta_alpha)
            if self.cfg.additive_head else None
        )

        self.demo_proj = nn.Sequential(nn.Linear(self.demo_dim, self.demo_hidden), nn.GELU())
        # head_in_max is the width the WIDEST arm (additive) uses. Every arm draws its head
        # weights at this width and slices, so the shared columns are bit-identical and
        # xavier's fan_in is arm-independent. See reinit_non_age_parameters_.
        self.head_in_max = self.d_model + self.demo_hidden + self.s
        head_in = self.d_model + self.demo_hidden + (self.s if self.cfg.additive_head else 0)
        self.head_in = head_in
        # D9 -- the fine-tune head operates on the pooled h, exactly as pretraining does.
        # There is no return_repr_only path, so pooling-site age parameters cannot be
        # gradient-dead at fine-tune time.
        if task not in {"pretrain", "classification"}:
            raise ValueError(f"task must be 'pretrain' or 'classification', got {task!r}")
        self.task = task
        out_dim = self.num_codes if task == "pretrain" else 1
        self.head = PredictionHead(head_in, out_dim, hidden_dim=self.head_in_max,
                                   final_bias=-7.0 if task == "pretrain" else 0.0)

        self.reinit_non_age_parameters_(self.seed)

    # ---- construction helpers --------------------------------------------- #
    @staticmethod
    def _load_embedding_table(path, table) -> torch.Tensor:
        if table is not None:
            return table.detach().clone()
        if path is None:
            raise ValueError("one of embedding_path / embedding_table is required")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing embedding file: {p}")
        obj = torch.load(p, map_location="cpu")
        t = obj["embeddings"] if isinstance(obj, dict) else obj
        if t.ndim != 2:
            raise ValueError(f"expected a 2-D embedding table, got {tuple(t.shape)}")
        return t

    def _param_generator(self, name: str) -> torch.Generator:
        """A generator seeded from ``(seed, parameter name)``.

        A single shared generator walked in name order would let a *shape* change in one
        parameter shift every draw after it: the ``additive`` arm has a wider head, so
        ``pooling.q_base`` would come out different from the other arms' and the "shared"
        backbone would not be shared. Seeding per parameter makes each one independent of
        every other one's shape. ``crc32`` is used rather than ``hash`` because Python's
        string hash is salted per process.
        """
        import zlib
        h = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
        return torch.Generator().manual_seed((int(self.seed) * 1_000_003 + h) % (2 ** 63 - 1))

    @torch.no_grad()
    def reinit_non_age_parameters_(self, seed: int) -> None:
        """Deterministically re-initialise every non-age trainable parameter.

        Constructing the age modules consumes RNG draws, so a naive ``manual_seed(s)`` gives
        different *shared* parameters per arm and ``INV-ZERO-A`` would fail for a reason that
        has nothing to do with the age pathway. Re-initialising in name-sorted order from
        per-parameter generators makes the shared backbone bit-identical across all four
        arms, including ``additive``.
        """
        self.seed = int(seed)
        age_ids = {id(p) for p in self.age_parameters()}

        owners: dict[str, tuple[nn.Module, str, nn.Parameter]] = {}
        for mname, mod in self.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                owners[f"{mname}.{pname}" if mname else pname] = (mod, pname, p)

        head_first = self.head.net[0]
        for full in sorted(owners):
            mod, pname, p = owners[full]
            if id(p) in age_ids or not p.requires_grad:
                continue
            if isinstance(mod, nn.LayerNorm):
                p.fill_(1.0) if pname == "weight" else p.zero_()
            elif isinstance(mod, nn.Linear):
                if pname == "weight":
                    if mod is head_first:
                        # Draw at the widest arm's width, then slice. Without this,
                        # additive's wider input changes xavier's fan_in and therefore the
                        # scale of EVERY weight in this layer -- an uncontrolled difference
                        # in function space between arms that INV-ZERO-B cannot see, since
                        # it only checks that the concat columns contribute zero given zero
                        # input.
                        fan_in, fan_out = self.head_in_max, p.shape[0]
                        bound = math.sqrt(6.0 / (fan_in + fan_out))
                        full_w = torch.empty(p.shape[0], self.head_in_max, dtype=p.dtype)
                        full_w.uniform_(-bound, bound,
                                        generator=self._param_generator(full))
                        # The extra s columns get a NORMAL draw, never zero: with the
                        # generator's final layer already zero-init, zeroing them too makes
                        # both gradients vanish permanently (dL/dW_c = 0 because g = 0, and
                        # dL/dg = 0 because W_c = 0), so the additive pathway could never
                        # start. This mirrors the kernel arm, where a nonzero alpha_base
                        # plays the role of W_c.
                        p.copy_(full_w[:, : p.shape[1]])
                    else:
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
                        bound = math.sqrt(6.0 / (fan_in + fan_out))
                        p.uniform_(-bound, bound, generator=self._param_generator(full))
                else:
                    p.zero_()
            # Raw parameters (alpha_base, q_base) are handled by their owning module below.

        for mname, mod in sorted(self.named_modules(), key=lambda kv: kv[0]):
            if mod is not self and hasattr(mod, "reset_raw_parameters_"):
                mod.reset_raw_parameters_(self._param_generator(mname))

        self.head.apply_final_bias_()

    # ---- declared parameter sets (D6) -------------------------------------- #
    def age_parameters(self) -> list[nn.Parameter]:
        seen, out = set(), []
        for _, mod in self.named_modules():
            if mod is self or not hasattr(mod, "age_parameters"):
                continue
            for p in mod.age_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def head_parameters(self) -> list[nn.Parameter]:
        seen, out = set(), []
        for _, mod in self.named_modules():
            if mod is self or not hasattr(mod, "head_parameters"):
                continue
            for p in mod.head_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    # ---- tau_max: one source of truth -------------------------------------- #
    def kernel_sites(self) -> list[tuple[str, nn.Module]]:
        return list(self.encoder.kernel_sites()) + [("pooling", self.pooling)]

    @property
    def tau_max(self) -> float:
        vals = {float(site.kernel.tau_max) for _, site in self.kernel_sites()}
        if len(vals) != 1:
            raise AssertionError(f"[INV-TMAX] kernel sites disagree on tau_max: {sorted(vals)}")
        return vals.pop()

    @torch.no_grad()
    def set_tau_max(self, value: float) -> None:
        for _, site in self.kernel_sites():
            site.kernel.tau_max.fill_(float(value))

    @torch.no_grad()
    def set_age_standardization(self, mean: float, sd: float) -> None:
        if not (sd > 0):
            raise ValueError(f"age_sd must be > 0, got {sd}")
        self.age_mean.fill_(float(mean))
        self.age_sd.fill_(float(sd))

    def standardize_demo_age(self, demo: torch.Tensor) -> torch.Tensor:
        """Standardize channel 0 (age) of a demographic vector; leave the rest untouched."""
        out = demo.clone()
        out[..., 0] = (out[..., 0] - self.age_mean) / self.age_sd
        return out

    def reset_clamp_stats(self) -> None:
        for _, site in self.kernel_sites():
            site.kernel.reset_clamp_stats()

    # ---- parameter accounting ---------------------------------------------- #
    def parameter_report(self) -> dict[str, int]:
        age_ids = {id(p) for p in self.age_parameters()}
        head_ids = {id(p) for p in self.head_parameters()}
        backbone = sum(p.numel() for p in self.parameters()
                       if p.requires_grad and id(p) not in age_ids and id(p) not in head_ids)
        age = sum(p.numel() for p in self.age_parameters())
        head = sum(p.numel() for p in self.head_parameters())
        encoder_total = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        return {
            "backbone": backbone,
            "age": age,
            "head": head,
            "frozen_embedding": int(self.embedding_table.numel()),
            "total_trainable": backbone + age + head,
            "encoder_trainable": encoder_total,
            "age_share_of_encoder": (age / encoder_total) if encoder_total else 0.0,
        }

    def config_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "use_residual": self.use_residual,
            "use_layernorm": self.use_layernorm,
            "use_ffn": self.use_ffn,
            "ffn_mult": self.ffn_mult,
            "use_out_proj": self.n_heads > 1,
            "n_heads": self.n_heads,
            "s": self.s,
            "tau_max": self.tau_max,
            "fourier": {"M": self.age_hparams["M"], "p_min": self.age_hparams["p_min"],
                        "p_max": self.age_hparams["p_max"], "log_age": True,
                        "age_emb_dim": 2 * self.age_hparams["M"]},
            "age_hidden": self.age_hparams["hidden"],
            "demo_dim": self.demo_dim,
            "demo_channels": list(self.demo_channels),
            "demo_hidden": self.demo_hidden,
            "race_encoding": self.race_encoding,
            "age_standardization": {"mean": float(self.age_mean), "sd": float(self.age_sd),
                                    "applies_to": "demographic channel 0 only; age_years stays raw"},
            "embedding_dim": self.embedding_dim,
            "injection": self.cfg.kernel_injection,
            "masking": self.cfg.masking,
            "pooling": self.cfg.pooling,
            "gen_final_bias": self.gen_final_bias,
            "center_delta_alpha": self.center_delta_alpha,
            "head_in": self.head_in,
            "head_out": self.head.out_dim,
            "head_final_bias": self.head.final_bias,
            "task": self.task,
        }

    # ---- forward ------------------------------------------------------------ #
    def _check_batch(self, batch: dict[str, torch.Tensor]) -> None:
        if "age_years" not in batch:
            raise AssertionError(
                "[INV-DEMO-SPLIT] age must arrive as batch['age_years'], not out of demographics"
            )
        demo = batch["demographics"]
        if demo.shape[-1] != self.demo_dim:
            raise AssertionError(
                f"[INV-DEMO-SPLIT] demographics must have {self.demo_dim} channels "
                f"{self.demo_channels}, got {demo.shape[-1]}"
            )
        for key in ("code_indices", "timestamps_days", "attention_mask"):
            if key not in batch:
                raise AssertionError(f"[HARD] batch is missing required key {key!r}")
        if "lengths" in batch:
            # lengths and the mask must agree; the pooling last-index and tau_to_now both
            # rely on it. Cheap HARD check.
            if not bool(torch.equal(batch["lengths"].to(batch["attention_mask"].device),
                                    batch["attention_mask"].sum(dim=1).long())):
                raise AssertionError("[HARD] batch['lengths'] disagrees with attention_mask")

    def forward(self, batch: dict[str, torch.Tensor], *, need_diagnostics: bool = False) -> dict:
        self._check_batch(batch)
        code_indices = batch["code_indices"]
        attention_mask = batch["attention_mask"]
        age_years = batch["age_years"]
        demographics = batch["demographics"]

        # tau is computed HERE, on the batch's device, from timestamps_days -- not shipped
        # from the collate. See data.tau_from_timestamps and the note in _pad_common.
        tau, tau_to_now = tau_from_timestamps(
            batch["timestamps_days"], attention_mask, batch.get("lengths"))

        x = self.embedding_table[code_indices]
        e = self.encoder(x, tau, attention_mask, age_years)

        last = self.pooling.last_valid_index(attention_mask)
        rows = torch.arange(code_indices.shape[0], device=code_indices.device)
        age_last = age_years[rows, last]

        if need_diagnostics:
            h, pool_attn, pool_log_w = self.pooling(
                e, tau_to_now, attention_mask, age_last, need_weights=True)
        else:
            h = self.pooling(e, tau_to_now, attention_mask, age_last)
            pool_attn = pool_log_w = None

        demo_last = self.standardize_demo_age(demographics[rows, last])
        parts = [h, self.demo_proj(demo_last)]
        age_delta = None
        if self.additive_age is not None:
            age_delta = self.additive_age(age_last)  # [B, s]
            parts.append(age_delta)
        logits = self.head(torch.cat(parts, dim=-1))

        out = {"h": h, "age_last": age_last}
        if self.task == "pretrain":
            out["code_logits"] = logits
        else:
            out["logits"] = logits.squeeze(-1)
        if need_diagnostics:
            out.update({"pool_attn": pool_attn, "pool_log_w": pool_log_w,
                        "age_delta": age_delta, "e": e})
        return out

    @torch.no_grad()
    def extract_representations(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Read-only patient representations for the frozen linear probe.

        Returns two vectors per patient (see the probe asymmetry note):

        * ``h_pool`` -- ``AttentionPooling`` output, before demographic combination and
          before any arm-specific concatenation. Primary probe input.
        * ``h_head`` -- exactly what the prediction head sees, minus the demographic
          sub-vector. For ``additive`` this is ``concat(h_pool, age_delta)``; for the
          other arms it equals ``h_pool``.

        Does not run the prediction head. Must not be called from ``train.py`` or
        ``train_finetune.py`` (D9 / INV-PROBE-FROZEN): a training path that skipped the
        head would leave pooling-site age parameters gradient-dead.
        """
        self._check_batch(batch)
        code_indices = batch["code_indices"]
        attention_mask = batch["attention_mask"]
        age_years = batch["age_years"]

        tau, tau_to_now = tau_from_timestamps(
            batch["timestamps_days"], attention_mask, batch.get("lengths"))
        x = self.embedding_table[code_indices]
        e = self.encoder(x, tau, attention_mask, age_years)

        last = self.pooling.last_valid_index(attention_mask)
        rows = torch.arange(code_indices.shape[0], device=code_indices.device)
        age_last = age_years[rows, last]

        h_pool = self.pooling(e, tau_to_now, attention_mask, age_last)
        if self.additive_age is not None:
            h_head = torch.cat([h_pool, self.additive_age(age_last)], dim=-1)
        else:
            h_head = h_pool
        return {"h_pool": h_pool, "h_head": h_head, "age_last": age_last}


def _smoke() -> None:
    from model_new import diagnostics
    from model_new.arms import ARMS
    from model_new.optim import build_param_groups

    torch.manual_seed(0)
    v, b, l, d_in = 40, 3, 6, 32
    table = torch.randn(v + 2, d_in)
    t = (torch.rand(b, l).cumsum(dim=1) * 40.0).double()
    mask = torch.ones(b, l, dtype=torch.bool)
    # tau is computed inside forward now; the batch ships timestamps + lengths.
    batch = {
        "code_indices": torch.randint(0, v + 2, (b, l)),
        "timestamps_days": t,
        "lengths": mask.sum(dim=1).long(),
        "attention_mask": mask,
        "age_years": torch.rand(b, l) * 80.0,
        "demographics": torch.rand(b, l, 9),
        "target_codes": (torch.rand(b, v) < 0.05).float(),
    }

    rows, ref, ref_enc = [], None, None
    for arm in ARMS:
        m = DKMModel(num_codes=v, embedding_table=table, arm=arm, seed=0, d_model=16,
                     age_hidden=8, demo_hidden=8)
        with torch.no_grad():
            out = m(batch)
        rep = m.parameter_report()
        groups, _ = build_param_groups(m, 1e-4, 1e-3, 1e-3)
        age_ids = {id(q) for q in m.age_parameters()}
        enc = torch.cat([p.detach().reshape(-1) for _, p in sorted(m.encoder.named_parameters())
                         if id(p) not in age_ids])
        if arm == "vanilla":
            ref, ref_enc = out["code_logits"], enc
        same = f"{float((out['code_logits'] - ref).abs().max()):.2e}"
        enc_same = f"{float((enc - ref_enc).abs().max()):.2e}"
        rows.append(f"{arm:<16} backbone={rep['backbone']:>7,} age={rep['age']:>5,} "
                    f"head={rep['head']:>6,} total={rep['total_trainable']:>7,} "
                    f"d_logits={same} d_encoder={enc_same} groups={[len(g['params']) for g in groups]}")
    rows.append("")
    rows.append("d_logits  == 0 for ALL FOUR arms is INV-ZERO-A: the head is drawn at the widest")
    rows.append("arm's width and sliced, so shared columns match and additive's extra s columns")
    rows.append("contribute zero at init (generator final layer zero-init). d_encoder == 0 too.")
    diagnostics.print_block("model.py smoke", rows)


if __name__ == "__main__":
    _smoke()
