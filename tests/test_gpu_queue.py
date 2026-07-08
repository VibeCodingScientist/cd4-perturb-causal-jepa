"""
Tests for the single-GPU serial scheduler (gpu_queue.py): atomic lock mutual exclusion,
§6 priority ordering of the pending queue, and an end-to-end submit of the self-test job.

Run: CD4_DATA_ROOT=$(mktemp -d) python3.12 tests/test_gpu_queue.py
"""
import os
import sys
import time
import tempfile
import threading

os.environ.setdefault("CD4_DATA_ROOT", tempfile.mkdtemp(prefix="cd4-gpuq-"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import contract as C
import gpu_queue as gq


def test_lock_mutual_exclusion():
    C.ensure_dirs()
    assert gq.acquire_lock(timeout=1.0) is True
    h = gq.lock_holder()
    assert h and h["pid"] == os.getpid()
    # a second acquire must block and then time out while the lock is held
    assert gq.acquire_lock(timeout=0.5, poll=0.05) is False
    gq.release_lock()
    assert gq.lock_holder() is None
    # now it can be acquired again
    assert gq.acquire_lock(timeout=1.0) is True
    gq.release_lock()
    print("PASS  lock mutual exclusion + release")


def test_priority_ordering():
    # clear queue
    for p in gq.QUEUE_DIR.glob("*.json"):
        p.unlink()
    # enqueue out of priority order; pending() must return §6 order (esm2 < causal < jepa)
    gq._enqueue("jepa")      # priority 3
    gq._enqueue("esm2")      # priority 0
    gq._enqueue("noncausal") # priority 2
    gq._enqueue("causal")    # priority 1
    order = [r.job for r in gq.pending()]
    assert order == ["esm2", "causal", "noncausal", "jepa"], order
    for p in gq.QUEUE_DIR.glob("*.json"):
        p.unlink()
    print("PASS  pending queue is ordered by §6 priority:", order)


def test_submit_end_to_end():
    for f in gq.QUEUE_DIR.glob("_selftest_done_*"):
        f.unlink()
    gq.submit("_selftest", dur=0.1)
    # the job ran (dropped its done marker) and the lock + queue are clean afterwards
    done = list(gq.QUEUE_DIR.glob("_selftest_done_*"))
    assert done, "self-test job did not run"
    assert gq.lock_holder() is None, "lock not released after submit"
    assert not gq.pending(), "queue not drained after submit"
    print("PASS  submit end-to-end: job ran, lock released, queue drained")


def test_concurrent_submits_serialize():
    """Two concurrent submitters must not run their jobs at the same time."""
    for f in gq.QUEUE_DIR.glob("_run_*"):
        f.unlink()
    intervals = []
    lock = threading.Lock()

    def worker(tag):
        def job(**kw):
            t0 = time.time()
            time.sleep(0.3)
            with lock:
                intervals.append((t0, time.time()))
        gq.JOBS["_probe_" + tag] = job
        gq.submit("_probe_" + tag)

    # register temp jobs sharing the lowest priority (unknown -> sorts last, same bucket)
    threads = [threading.Thread(target=worker, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(intervals) == 2
    (s1, e1), (s2, e2) = sorted(intervals)
    assert e1 <= s2 + 1e-3, f"jobs overlapped on the GPU: {intervals}"
    print("PASS  concurrent submits serialize (no overlap)")


if __name__ == "__main__":
    test_lock_mutual_exclusion()
    test_priority_ordering()
    test_submit_end_to_end()
    test_concurrent_submits_serialize()
    print("\nGPU QUEUE TESTS PASSED")
