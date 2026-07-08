"""core.models.jepa — Cell-JEPA-style pretraining (UNIFIED_BUILD_PLAN.md §7e).

Developer 2's star deliverable. Implements the *correct* recipe (the plan's §1
correction #2), and refuses the two collapse paths the prior plan fell into:

  * The student sees VALUE-MASKED single cells; the EMA **teacher** sees the
    unmasked cell with a **stop-gradient**; a predictor head maps the student's
    pooled embedding toward the teacher's; the alignment loss is **cosine**, plus a
    reconstruction term.  (NOT plain-L2, NOT a pseudobulk MLP, and the EMA teacher
    is NEVER dropped.)
  * It operates at **single-cell resolution** (<=600 HVG/cell), masking expression
    *values within a cell* — it does not pseudobulk.
  * The student is the **same ``GeneTokenEncoder`` class the causal model uses**, so
    ``student.state_dict()`` initializes ``CausalCisTransFormer.encoder`` (§7f).

A collapse guard monitors the std of teacher embeddings; if it trends to 0 the run
escalates (VICReg penalty + higher EMA momentum) and, if it truly collapses, fails
loud rather than shipping a degenerate checkpoint.

Runtime discipline: ``pretrain_jepa`` runs the §6 **measure-then-extrapolate gate**
(time N warm steps, project to the full schedule) and applies the §10 fallback
(fewer cells / fewer steps / smaller d_model) before committing to the overnight run.

Testability: everything numeric here is exercised on tiny random ``(gene_id, value)``
tensors with the reference encoder (``tests/test_jepa.py``) — no real cells needed.
The GPU entry point ``python -m core.models.jepa`` is what ``gpu_queue.py submit
jepa`` (G4) invokes.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder_api import MASK_VALUE_SENTINEL, EncoderOutput

# ---------------------------------------------------------------------------
# Encoder class resolution: prefer Developer 1's real GeneTokenEncoder once it
# exists; fall back to the reference so this module is standalone-runnable now.
# ---------------------------------------------------------------------------
def _default_encoder_cls() -> type[nn.Module]:
    try:
        from .gene_tokens import GeneTokenEncoder  # Developer 1, appears at core-frozen
        return GeneTokenEncoder
    except Exception:
        from ._reference_gene_tokens import ReferenceGeneTokenEncoder
        return ReferenceGeneTokenEncoder


# ---------------------------------------------------------------------------
# 1. Masking (single-cell, values only)  — §7e "mask EXPRESSION VALUES only"
# ---------------------------------------------------------------------------
def mask_values(
    values: Tensor,
    frac: float = 0.5,
    sentinel: float = MASK_VALUE_SENTINEL,
    key_padding_mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Mask a fraction of expression VALUES within each cell.

    Only ``values`` change (gene_ids are untouched — masking is on expression, not
    identity). Padding positions are never selected. Returns ``(masked_values, mask)``
    where ``mask`` is a bool tensor, True at masked positions (the reconstruction
    targets).
    """
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    # Generate on the generator's device (kept on CPU for cross-device reproducibility
    # — MPS generators are unreliable), then move to the values' device.
    gen_device = generator.device if generator is not None else values.device
    rand = torch.rand(values.shape, generator=generator, device=gen_device).to(values.device)
    mask = rand < frac
    if key_padding_mask is not None:
        mask = mask & (~key_padding_mask)
    masked = values.masked_fill(mask, sentinel)
    return masked, mask


# ---------------------------------------------------------------------------
# 2. Losses
# ---------------------------------------------------------------------------
def cosine_alignment_loss(pred: Tensor, target: Tensor) -> Tensor:
    """1 - cosine(pred, stop_grad(target)), averaged over the batch (§7e)."""
    target = target.detach()
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def vicreg_penalty(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """VICReg variance + covariance penalty (collapse guard, §7e / §10).

    Variance term hinges each feature's std up toward ``gamma`` (fights dimensional
    collapse); covariance term decorrelates features. The *invariance* term is the
    JEPA cosine loss itself, so it is not included here. Applied to the student's
    embeddings (the ones carrying gradient).
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)                    # [d, d]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss + cov_loss


# ---------------------------------------------------------------------------
# 3. The JEPA module (student + EMA teacher + predictor)
# ---------------------------------------------------------------------------
class Predictor(nn.Module):
    """MLP predictor head mapping student.pooled -> teacher.pooled space (§7e)."""

    def __init__(self, d_model: int, hidden_ratio: float = 2.0):
        super().__init__()
        hidden = int(d_model * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CellJEPA(nn.Module):
    """Student encoder + stop-gradient EMA teacher + predictor head.

    The student is trained; the teacher is an exponential moving average of the
    student with ``requires_grad_(False)`` and ``eval()`` mode (stable, dropout-free
    targets). ``ema_update`` is called by the trainer *after* each ``optimizer.step``.
    """

    def __init__(
        self,
        student: nn.Module,
        d_model: int,
        w_jepa: float = 1.0,
        w_rec: float = 0.5,
        w_vicreg: float = 0.0,
        mask_frac: float = 0.5,
        sentinel: float = MASK_VALUE_SENTINEL,
    ):
        super().__init__()
        if not hasattr(student, "value_head"):
            raise TypeError("student encoder must expose a `value_head` for the JEPA reconstruction term")
        self.student = student
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()                      # no dropout / stable targets
        self.predictor = Predictor(d_model)
        self.d_model = d_model
        self.w_jepa = float(w_jepa)
        self.w_rec = float(w_rec)
        self.w_vicreg = float(w_vicreg)
        self.mask_frac = float(mask_frac)
        self.sentinel = float(sentinel)

    # -- keep the teacher frozen + eval even if .train() is called on the parent --
    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        gene_ids: Tensor,
        values: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        masked_values, mask = mask_values(
            values, self.mask_frac, self.sentinel, key_padding_mask, generator
        )
        # student sees the MASKED cell (attn_mask=None during pretraining)
        s_out: EncoderOutput = self.student(gene_ids, masked_values, key_padding_mask=key_padding_mask)
        pred = self.predictor(s_out.pooled)

        # teacher sees the UNMASKED cell, under stop-gradient (no_grad + frozen + detach)
        with torch.no_grad():
            t_out: EncoderOutput = self.teacher(gene_ids, values, key_padding_mask=key_padding_mask)
        e_t = t_out.pooled

        loss_jepa = cosine_alignment_loss(pred, e_t)

        # reconstruction of the TRUE masked values from the student's masked-input tokens
        pred_vals = self.student.value_head(s_out.tokens).squeeze(-1)   # [B, L]
        if mask.any():
            loss_rec = F.mse_loss(pred_vals[mask], values[mask])
        else:
            loss_rec = values.new_zeros(())

        loss_vic = vicreg_penalty(s_out.pooled) if self.w_vicreg > 0 else values.new_zeros(())

        total = self.w_jepa * loss_jepa + self.w_rec * loss_rec + self.w_vicreg * loss_vic

        # collapse monitor: spread of teacher embeddings across the batch (§7e).
        with torch.no_grad():
            teacher_std = e_t.std(dim=0).mean() if e_t.shape[0] > 1 else e_t.new_zeros(())
        return {
            "loss": total,
            "loss_jepa": loss_jepa.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_vicreg": loss_vic.detach(),
            "teacher_std": teacher_std.detach(),
            "n_masked": int(mask.sum().item()),
        }

    @torch.no_grad()
    def ema_update(self, m: float) -> None:
        """teacher <- m * teacher + (1 - m) * student, for params and buffers."""
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bt, bs in zip(self.teacher.buffers(), self.student.buffers()):
            if bt.dtype.is_floating_point:
                bt.mul_(m).add_(bs.detach(), alpha=1.0 - m)
            else:
                bt.copy_(bs)

    def student_state_dict(self) -> dict:
        """State dict of the STUDENT encoder — the payload that initializes the
        causal model's encoder (§7f). Keys match the encoder submodule names."""
        return self.student.state_dict()


# ---------------------------------------------------------------------------
# 4. Schedules
# ---------------------------------------------------------------------------
def momentum_schedule(step: int, total_steps: int, base: float = 0.996, final: float = 0.999) -> float:
    """Cosine EMA-momentum ramp base -> final over training (BYOL-style, §7e)."""
    if total_steps <= 1:
        return final
    progress = min(step, total_steps) / total_steps
    return final - (final - base) * (math.cos(math.pi * progress) + 1.0) / 2.0


def lr_schedule(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.05) -> float:
    """Linear warmup then cosine decay to 0."""
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# 5. Config + fallbacks (§10)
# ---------------------------------------------------------------------------
@dataclass
class JEPAConfig:
    # data / model dimensions
    n_genes: int = 3000              # HVG_N; the encoder's gene vocabulary
    max_genes_per_cell: int = 600    # §7e Cell-JEPA per-cell cap
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    n_proxy: int = 32
    # optimization
    steps: int = 20_000
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    # JEPA hyper-parameters
    mask_frac: float = 0.5
    w_jepa: float = 1.0
    w_rec: float = 0.5
    w_vicreg: float = 0.0
    ema_base: float = 0.996
    ema_final: float = 0.999
    # runtime gate (§6) + collapse guard (§10)
    gate_steps: int = 200
    budget_seconds: float = 12 * 3600.0     # overnight slot
    auto_fallback: bool = True
    collapse_std_floor: float = 1e-3
    collapse_warn_ratio: float = 0.1        # warn if std < ratio * initial std
    # bookkeeping
    n_cells: int = 1_000_000
    seed: int = 42
    log_every: int = 100
    ckpt_every: int = 2000
    ckpt_path: Optional[str] = None         # defaults to CHECKPOINTS_DIR/jepa.pt
    device: str = "auto"
    fallback_level: int = 0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def as_fallback(self) -> "JEPAConfig":
        """Next §10 fallback tier: halve cells / steps / d_model / batch (the plan's
        256->128 example is exactly one tier). Small floors keep it sane without
        clamping back *up* for already-small configs. (``batch_size`` takes effect
        only when the caller rebuilds the loader — see ``main``.)"""
        return replace(
            self,
            n_cells=max(1_000, self.n_cells // 2),
            steps=max(1, self.steps // 2),
            d_model=max(16, self.d_model // 2),
            batch_size=max(2, self.batch_size // 2),
            fallback_level=self.fallback_level + 1,
        )


# ---------------------------------------------------------------------------
# 6. Errors + reports
# ---------------------------------------------------------------------------
class JEPACollapse(RuntimeError):
    """Raised when the teacher embedding collapses despite escalation — fail loud
    rather than checkpointing a degenerate encoder."""


@dataclass
class GateResult:
    seconds_per_step: float
    projected_seconds: float
    over_budget: bool


@dataclass
class JEPAReport:
    steps_run: int
    final_loss: float
    teacher_std_history: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    escalations: int = 0
    ckpt_path: Optional[str] = None
    config: Optional[dict] = None
    fallback_applied: bool = False
    projected_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# 7. Building blocks for the training loop
# ---------------------------------------------------------------------------
def build_jepa(config: JEPAConfig, encoder: Optional[nn.Module] = None) -> CellJEPA:
    """Construct a CellJEPA (student + EMA teacher + predictor) from config.

    ``encoder`` lets tests inject a tiny reference student; production passes None so
    the resolved ``GeneTokenEncoder`` is built at ``config`` scale.
    """
    if encoder is None:
        cls = _default_encoder_cls()
        encoder = cls(
            n_genes=config.n_genes,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_proxy=config.n_proxy,
        )
    d_model = getattr(encoder, "d_model", config.d_model)
    return CellJEPA(
        encoder, d_model,
        w_jepa=config.w_jepa, w_rec=config.w_rec, w_vicreg=config.w_vicreg,
        mask_frac=config.mask_frac,
    )


def _infinite(batches: Iterable) -> Iterator:
    """Cycle an iterable forever (a fresh iterator each pass)."""
    while True:
        empty = True
        for b in batches:
            empty = False
            yield b
        if empty:
            raise ValueError("data loader yielded no batches")


def _to_device(batch, device: str):
    gene_ids, values, kpm = batch
    gene_ids = gene_ids.to(device)
    values = values.to(device)
    if kpm is not None:
        kpm = kpm.to(device)
    return gene_ids, values, kpm


# ---------------------------------------------------------------------------
# 8. The training loop (with the §6 gate + §10 collapse guard)
# ---------------------------------------------------------------------------
def pretrain_jepa(
    data_loader: Iterable,
    config: JEPAConfig,
    encoder: Optional[nn.Module] = None,
    log_fn: Callable[[str], None] = print,
    _max_fallbacks: int = 2,
) -> JEPAReport:
    """Pretrain a Cell-JEPA student and checkpoint it for causal-encoder init.

    Sequence (§6, §7e, §10):
      1. Build student + EMA teacher + predictor + AdamW.
      2. **Gate**: run ``gate_steps`` real steps, time them, project to ``steps``. If
         projected > ``budget_seconds`` and ``auto_fallback``, log and restart at the
         next fallback tier (``as_fallback``). Gate steps are real training, not
         wasted.
      3. Run the remaining steps with cosine LR + EMA momentum ramp, monitoring the
         teacher-embedding std. On a downward collapse trend: escalate (turn on/raise
         VICReg, bump EMA momentum). On true collapse (< floor after escalation):
         raise ``JEPACollapse``.
      4. Checkpoint the STUDENT encoder to ``ckpt_path`` (default CHECKPOINTS_DIR/
         jepa.pt) periodically and at the end.
    """
    device = config.resolved_device()
    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed)  # CPU generator (MPS-safe; see mask_values)

    jepa = build_jepa(config, encoder).to(device)
    jepa.train()
    opt = torch.optim.AdamW(
        [p for p in jepa.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )

    batches = _infinite(data_loader)
    report = JEPAReport(steps_run=0, final_loss=float("nan"), config=asdict(config))

    # ---- measure-then-extrapolate gate (§6) --------------------------------
    gate_n = min(config.gate_steps, config.steps)
    t0 = time.perf_counter()
    step = 0
    initial_std: Optional[float] = None
    ema_momentum_floor = 0.0

    def run_step(step_idx: int) -> dict:
        gene_ids, values, kpm = _to_device(next(batches), device)
        for g in opt.param_groups:
            g["lr"] = lr_schedule(step_idx, config.steps, config.lr, config.warmup_frac)
        opt.zero_grad(set_to_none=True)
        out = jepa(gene_ids, values, key_padding_mask=kpm, generator=gen)
        out["loss"].backward()
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(jepa.parameters(), config.grad_clip)
        opt.step()
        m = max(momentum_schedule(step_idx, config.steps, config.ema_base, config.ema_final), ema_momentum_floor)
        jepa.ema_update(m)
        return out

    for step in range(gate_n):
        out = run_step(step)
        if initial_std is None:
            initial_std = float(out["teacher_std"])
    gate_elapsed = time.perf_counter() - t0
    sps = gate_elapsed / max(1, gate_n)
    projected = sps * config.steps
    report.projected_seconds = projected
    gate = GateResult(seconds_per_step=sps, projected_seconds=projected, over_budget=projected > config.budget_seconds)
    log_fn(
        f"[jepa][gate] {gate_n} steps in {gate_elapsed:.1f}s "
        f"({sps*1000:.1f} ms/step) -> projected {projected/3600:.2f}h for {config.steps} steps "
        f"(budget {config.budget_seconds/3600:.1f}h)"
    )
    if gate.over_budget and config.auto_fallback and config.fallback_level < _max_fallbacks:
        fb = config.as_fallback()
        log_fn(
            f"[jepa][gate] OVER BUDGET -> applying fallback tier {fb.fallback_level} "
            f"(d_model {config.d_model}->{fb.d_model}, steps {config.steps}->{fb.steps}, "
            f"n_cells {config.n_cells}->{fb.n_cells})"
        )
        rep = pretrain_jepa(data_loader, fb, encoder=encoder, log_fn=log_fn, _max_fallbacks=_max_fallbacks)
        rep.fallback_applied = True
        return rep

    # ---- full training -----------------------------------------------------
    for step in range(gate_n, config.steps):
        out = run_step(step)
        if step % config.log_every == 0 or step == config.steps - 1:
            _log_and_monitor(step, out, report, log_fn)
            initial_std, ema_momentum_floor = _collapse_guard(
                out, initial_std, config, jepa, report, log_fn, ema_momentum_floor
            )
        if config.ckpt_every and step > 0 and step % config.ckpt_every == 0:
            _save_checkpoint(jepa, config, step, report)

    report.steps_run = config.steps
    report.final_loss = float(out["loss"].detach())
    report.ckpt_path = _save_checkpoint(jepa, config, config.steps, report)
    log_fn(f"[jepa] done: {config.steps} steps, final loss {report.final_loss:.4f}, ckpt {report.ckpt_path}")
    return report


def _log_and_monitor(step, out, report: JEPAReport, log_fn):
    tstd = float(out["teacher_std"])
    loss = float(out["loss"].detach())
    report.teacher_std_history.append((step, tstd))
    report.loss_history.append((step, loss))
    log_fn(
        f"[jepa] step {step}: loss {loss:.4f} "
        f"(jepa {float(out['loss_jepa']):.4f} rec {float(out['loss_rec']):.4f} "
        f"vic {float(out['loss_vicreg']):.4f}) teacher_std {tstd:.4f}"
    )


def _collapse_guard(out, initial_std, config: JEPAConfig, jepa: CellJEPA, report, log_fn, ema_floor):
    """Escalate on a downward collapse trend; fail loud on true collapse (§7e/§10)."""
    tstd = float(out["teacher_std"])
    if initial_std is None or initial_std <= 0:
        return max(initial_std or 0.0, tstd), ema_floor
    if tstd < config.collapse_std_floor:
        # already escalated at least once and still collapsed -> abort.
        if report.escalations > 0:
            raise JEPACollapse(
                f"teacher embedding collapsed: std {tstd:.2e} < floor {config.collapse_std_floor:.1e} "
                f"after {report.escalations} escalation(s)"
            )
    if tstd < config.collapse_warn_ratio * initial_std:
        report.escalations += 1
        jepa.w_vicreg = max(jepa.w_vicreg, 0.04 * report.escalations)
        ema_floor = min(0.9995, max(ema_floor, config.ema_final))
        log_fn(
            f"[jepa][collapse-guard] teacher_std {tstd:.2e} << initial {initial_std:.2e}; "
            f"escalation #{report.escalations}: w_vicreg -> {jepa.w_vicreg:.3f}, ema_floor -> {ema_floor:.4f}"
        )
    return initial_std, ema_floor


def _resolve_ckpt_path(config: JEPAConfig) -> Path:
    if config.ckpt_path:
        return Path(config.ckpt_path)
    from core import contract
    contract.ensure_dirs()
    return contract.checkpoint_path("jepa")


def _save_checkpoint(jepa: CellJEPA, config: JEPAConfig, step: int, report: JEPAReport) -> str:
    path = _resolve_ckpt_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": jepa.student_state_dict(),
            "predictor_state_dict": jepa.predictor.state_dict(),
            "config": asdict(config),
            "step": step,
            "d_model": jepa.d_model,
            "teacher_std_history": report.teacher_std_history,
        },
        path,
    )
    return str(path)


# ---------------------------------------------------------------------------
# 9. GPU entry point — `gpu_queue.py submit jepa` (G4) calls this.
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    from core import contract

    p = argparse.ArgumentParser(description="Cell-JEPA pretraining (G4)")
    p.add_argument("--steps", type=int, default=JEPAConfig.steps)
    p.add_argument("--d-model", type=int, default=JEPAConfig.d_model)
    p.add_argument("--batch-size", type=int, default=JEPAConfig.batch_size)
    p.add_argument("--n-cells", type=int, default=JEPAConfig.n_cells)
    p.add_argument("--max-genes", type=int, default=JEPAConfig.max_genes_per_cell)
    p.add_argument("--budget-hours", type=float, default=12.0)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(argv)

    cfg = JEPAConfig(
        n_genes=contract.HVG_N,
        steps=args.steps,
        d_model=args.d_model,
        batch_size=args.batch_size,
        n_cells=args.n_cells,
        max_genes_per_cell=args.max_genes,
        budget_seconds=args.budget_hours * 3600.0,
        ckpt_path=args.ckpt,
        device=args.device,
    )
    # Real cells come from Developer 1's subsampler output in CELLS_DIR (§7e / §2).
    from .jepa_data import build_cell_loader

    loader = build_cell_loader(cfg)
    report = pretrain_jepa(loader, cfg)
    print(f"[jepa] checkpoint: {report.ckpt_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "mask_values", "cosine_alignment_loss", "vicreg_penalty",
    "Predictor", "CellJEPA",
    "momentum_schedule", "lr_schedule",
    "JEPAConfig", "JEPACollapse", "GateResult", "JEPAReport",
    "build_jepa", "pretrain_jepa", "main",
]
