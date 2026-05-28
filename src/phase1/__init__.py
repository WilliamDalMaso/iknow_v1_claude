"""Phase 1 extraction package.

Behavior-preserving decomposition of the historically monolithic
``phase1_extract.py``. Modules here are extracted one concern at a time, each
step verified against ``tests/test_phase1_regression.py`` so the Douglass
phase1_v3 output is provably unchanged. ``phase1_extract.py`` remains the entry
point and re-imports these helpers, so all existing call sites keep working.
"""
