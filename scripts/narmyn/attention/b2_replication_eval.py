#!/usr/bin/env python3
"""Run Codex's B2 evaluation (scripts/run_b2_token_evaluation.py) unchanged, with one fix.

The independent replication of the B2a/S4/B2b token package (protocol Section 65)
uses Codex's committed scripts and plan as they are. The only change: COCO-Cap
class names contain a space ("dining table", "potted plant") while the token keys
and the plan use a hyphen, so ``class_tokens`` is called with the hyphenated
name. VOC-Sim names are already hyphenated, so VOC-Sim is unaffected.

    python scripts/narmyn/attention/b2_replication_eval.py <run_b2_token_evaluation.py arguments>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_b2_token_evaluation as evaluation  # noqa: E402

_original = evaluation.class_tokens


def class_tokens(pipe, classname: str, *args, **kwargs):
    return _original(pipe, classname.replace(" ", "-"), *args, **kwargs)


evaluation.class_tokens = class_tokens

if __name__ == "__main__":
    evaluation.main()
