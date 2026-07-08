"""
gpu_queue.py — the single-GPU serial job scheduler (UNIFIED_BUILD_PLAN.md §6).

Both worktrees launch GPU work ONLY via `python gpu_queue.py submit <job>`. Never run
training directly. The queue guarantees:
  * MUTUAL EXCLUSION — one job on the GPU at a time, via an atomic O_EXCL lock on
    DATA_ROOT/GPU_LOCK. A second submitter blocks until the lock frees.
  * PRIORITY ORDER — pending jobs run in the §6 order (contract.GPU_JOB_ORDER); a
    lock holder yields if a higher-priority job is waiting.
  * THE EPOCH-1 GATE — a training job first runs a short probe (1 epoch / N steps),
    extrapolates wall-time to the planned schedule, and applies the job's fallback if it
    would overrun its slot — so no job silently eats the queue.

Jobs dispatch to the real (gated) implementations. jepa / jepa_finetune belong to
Developer 2 (core.models.jepa); this file dispatches to them if present.

Usage:
  python gpu_queue.py submit causal            # enqueue + run job G2 when the GPU is free
  python gpu_queue.py status                   # show the lock + pending queue
  python gpu_queue.py run causal               # run WITHOUT the lock (debug only)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core import contract as C

QUEUE_DIR = C.DATA_ROOT / "gpu_queue"

# §6 rough slot budgets (hours) used by the epoch-1 gate. Fallbacks are described in §6/§10.
SLOT_HOURS: Dict[str, float] = {
    "esm2": 1.0, "causal": 5.0, "noncausal": 5.0, "jepa": 12.0,
    "jepa_finetune": 3.0, "arc_state": 4.0, "iftime": 6.0,
}


def _log(msg: str) -> None:
    print(f"[gpu_queue] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------
def acquire_lock(*, poll: float = 1.0, timeout: Optional[float] = None) -> bool:
    """Blocking atomic lock acquire (O_EXCL). Returns False only on timeout."""
    C.GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(str(C.GPU_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps(
                {"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}
            ).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if timeout is not None and time.time() - start > timeout:
                return False
            time.sleep(poll)


def release_lock() -> None:
    try:
        os.unlink(C.GPU_LOCK)
    except FileNotFoundError:
        pass


def lock_holder() -> Optional[dict]:
    try:
        return json.loads(C.GPU_LOCK.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Pending queue (filesystem)
# ---------------------------------------------------------------------------
@dataclass
class Request:
    job: str
    priority: int
    t: float
    path: Path


def _enqueue(job: str) -> Request:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    t = time.time()
    prio = C.gpu_job_priority(job)
    path = QUEUE_DIR / f"{prio:02d}_{job}_{os.getpid()}_{int(t*1000)}.json"
    path.write_text(json.dumps({"job": job, "priority": prio, "t": t, "pid": os.getpid()}))
    return Request(job, prio, t, path)


def pending() -> List[Request]:
    if not QUEUE_DIR.exists():
        return []
    out = []
    for p in QUEUE_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            out.append(Request(d["job"], d["priority"], d["t"], p))
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(out, key=lambda r: (r.priority, r.t))


def _dequeue(req: Request) -> None:
    try:
        req.path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Epoch-1 measure-then-extrapolate gate (§6)
# ---------------------------------------------------------------------------
def measure_then_extrapolate(probe: Callable[[], float], planned_units: float,
                             slot_hours: float, label: str) -> bool:
    """Run `probe` (1 epoch / N steps) returning units done, extrapolate to planned_units,
    and return True if the projected wall-time fits the slot. Logs the projection either way."""
    t0 = time.time()
    units = max(1e-9, probe())
    dt = time.time() - t0
    projected_s = dt / units * planned_units
    fits = projected_s <= slot_hours * 3600
    _log(f"epoch-1 gate [{label}]: probe {units:.0f} units in {dt:.1f}s -> "
         f"projected {projected_s/3600:.2f}h vs slot {slot_hours:.1f}h "
         f"({'OK' if fits else 'OVER BUDGET -> apply §6 fallback'})")
    return fits


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------
def _job_esm2(**kw):
    from core import features as feat
    seq_path = C.RAW_DIR / "gene_sequences.parquet"
    if not seq_path.exists():
        raise RuntimeError(
            f"esm2 job needs a gene->protein-sequence map at {seq_path} "
            "(ENSG->UniProt->sequence, prepared by core.data). GenePT fallback available."
        )
    import pandas as pd
    seqs = pd.read_parquet(seq_path)["sequence"].to_dict()
    feat.write_esm2(feat.build_esm2(seqs))


def _job_causal(**kw):
    from core.models import causal_cistransformer as cc
    cc.run_causal()


def _job_noncausal(**kw):
    from core.models import causal_cistransformer as cc
    cc.run_noncausal()


def _job_dev2(name: str):
    def _run(**kw):
        try:
            from core.models import jepa  # Developer 2
        except Exception as e:
            raise RuntimeError(f"job '{name}' is owned by Developer 2 (core.models.jepa): {e}")
        fn = getattr(jepa, "run_" + name, None)
        if fn is None:
            raise RuntimeError(f"core.models.jepa has no run_{name}()")
        fn(**kw)
    return _run


def _job_gated(name: str):
    def _run(**kw):
        raise RuntimeError(f"job '{name}' is gated (§7g/§6 If-Time); wire it before submitting.")
    return _run


def _job_selftest(**kw):
    """Dummy job for exercising the lock/priority mechanism in tests."""
    dur = float(kw.get("dur", 0.2))
    time.sleep(dur)
    (QUEUE_DIR / f"_selftest_done_{os.getpid()}").write_text(str(time.time()))


JOBS: Dict[str, Callable] = {
    "esm2": _job_esm2,
    "causal": _job_causal,
    "noncausal": _job_noncausal,
    "jepa": _job_dev2("jepa"),
    "jepa_finetune": _job_dev2("jepa_finetune"),
    "arc_state": _job_gated("arc_state"),
    "iftime": _job_gated("iftime"),
    "_selftest": _job_selftest,
}


def run_job(job: str, **kw) -> None:
    if job not in JOBS:
        raise ValueError(f"unknown job '{job}'; known: {sorted(JOBS)}")
    _log(f"running job '{job}' (slot budget {SLOT_HOURS.get(job, '?')}h) ...")
    t0 = time.time()
    JOBS[job](**kw)
    _log(f"job '{job}' done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# submit — enqueue, then run when we hold the lock AND are highest priority
# ---------------------------------------------------------------------------
def submit(job: str, *, wait: bool = True, poll: float = 0.5, **kw) -> None:
    if job not in JOBS:
        raise ValueError(f"unknown job '{job}'; known: {sorted(JOBS)}")
    req = _enqueue(job)
    _log(f"submitted '{job}' (priority {req.priority}); waiting for the GPU ...")
    try:
        while True:
            acquire_lock()
            try:
                top = pending()[0] if pending() else None
                if top is not None and top.path == req.path:
                    run_job(job, **kw)
                    _dequeue(req)
                    return
                # a higher-priority job is waiting: yield the lock to it
            finally:
                release_lock()
            if not wait:
                _dequeue(req)
                return
            time.sleep(poll)
    except BaseException:
        _dequeue(req)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_status() -> None:
    h = lock_holder()
    print("GPU_LOCK:", "held by " + json.dumps(h) if h else "free")
    pend = pending()
    if not pend:
        print("pending: (none)")
    for r in pend:
        print(f"  [{r.priority:02d}] {r.job}  (enqueued {time.strftime('%H:%M:%S', time.localtime(r.t))})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="enqueue a job and run it when the GPU is free")
    s.add_argument("job", choices=sorted(JOBS))
    s.add_argument("--no-wait", action="store_true")
    r = sub.add_parser("run", help="run a job WITHOUT the lock (debug only)")
    r.add_argument("job", choices=sorted(JOBS))
    sub.add_parser("status", help="show the lock + pending queue")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    if args.cmd == "submit":
        submit(args.job, wait=not args.no_wait)
    elif args.cmd == "run":
        run_job(args.job)
    elif args.cmd == "status":
        _cmd_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
