#!/usr/bin/env python3
"""Summarise measure_cost.py JSON files (A100, V100) into comparisons/cost/result.json (medians, p10/p90)."""
import json
import sys
from pathlib import Path

import numpy as np

EXT = Path.home() / "TRDP/external/n_side"
HERE = Path(__file__).resolve().parent
STAGES = ("gen_plain", "gen_capture", "runtime_extras", "fine_map", "hier_map", "aggregate", "crf_fine", "crf_brace")
PEAKS = ("gen_plain_peak_mib", "gen_capture_total_peak_mib", "fine_map_peak_mib", "hier_map_peak_mib")

out = {}
for name in sys.argv[1:]:
    J = json.loads((EXT / f"cost/{name}.json").read_text()); I = J["images"]
    s = {k: np.array([r[k] for r in I]) for k in STAGES + PEAKS}
    med = {k: float(np.median(v)) for k, v in s.items()}
    pct = {k: [float(np.percentile(v, 10)), float(np.percentile(v, 90))] for k, v in s.items()}
    tot_b = s["gen_capture"] + s["hier_map"] + s["crf_brace"]; tot_f = s["gen_capture"] + s["fine_map"] + s["crf_fine"]
    out[name] = {"meta": J["meta"], "n": len(I), "median": med, "p10_p90": pct,
                 "capture_overhead_median": float(np.median(s["gen_capture"] - s["gen_plain"])),
                 "brace_total_median": float(np.median(tot_b)), "fine_only_total_median": float(np.median(tot_f)),
                 "brace_gpu_median": float(np.median(s["gen_capture"] + s["hier_map"])),
                 "fine_gpu_median": float(np.median(s["gen_capture"] + s["fine_map"])),
                 "peak_mib_max": float(max(s[k].max() for k in PEAKS)),
                 "images_equal_plain": int(sum(r["image_equal_plain"] for r in I)),
                 "v0_maxdiff_max": float(max(r["v0_maxdiff"] for r in I)), "v3_maxdiff_max": float(max(r["v3_maxdiff"] for r in I))}
print(json.dumps(out, indent=1))
(HERE / "result.json").write_text(json.dumps(out, indent=1))
