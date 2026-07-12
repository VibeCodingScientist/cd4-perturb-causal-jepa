"""predictability_audit — package the seven pre-registered probes + the predictability budget + the
do-operator positive control into one dataset **predictability scorecard**.

This is an evaluation/methods reframe of already-validated content (see PREDICTABILITY_AUDIT.md). It does
NOT retrain or fit anything and does NOT modify the frozen `core.eval`. `run_audit()` reads the committed
gate CSVs (ground truth) and assembles a scorecard: each probe's score, its degree/label-preserving null,
its position relative to the measured noise floor, and its pre-registered verdict — with the do-operator
C2 as the signal-detection anchor (a null means "no signal," not "no sensitivity").
"""
from .scorecard import run_audit, PROBES  # noqa: F401
