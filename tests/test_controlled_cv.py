from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.segmentation import controlled_cv as cv
from src.segmentation.attention_aggregation import attention_to_mask, normalize_map
from src.segmentation.metrics import BinarySegmentationMetrics

ROOT = Path(__file__).resolve().parents[1]
CLASSES = tuple(f"class{i:02d}" for i in range(20))


def synthetic_results(per_image_iou: dict[str, np.ndarray], images_per_class: int = 10) -> cv.Results:
    """Results whose pooled class IoU equals the mean of the given per-image IoUs.

    Every image has 100 annotated pixels, ``tp = 100 * iou``, ``fn = 100 - tp``,
    no false positives, and the same counts at every threshold.
    """
    names = tuple(per_image_iou)
    n = len(CLASSES) * images_per_class
    counts = np.zeros((len(names), n, len(cv.THRESHOLDS), 4), dtype=np.int64)
    for c, name in enumerate(names):
        tp = np.rint(100 * per_image_iou[name]).astype(np.int64)
        counts[c, :, :, 0] = tp[:, None]
        counts[c, :, :, 2] = (100 - tp)[:, None]
        counts[c, :, :, 3] = 900
    classes = tuple(CLASSES[i // images_per_class] for i in range(n))
    folds = np.array([i % images_per_class % cv.FOLD_COUNT for i in range(n)])
    return cv.Results(names, cv.THRESHOLDS, tuple(f"img-{i}" for i in range(n)), classes, folds,
                      counts, np.full((len(names), n), 0.5), {"synthetic": True})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.maps = torch.rand((6, 64, 64), generator=generator) ** 3
        self.maps[3] = 0.25  # constant map: normalization must return zeros
        self.target = (torch.rand((96, 96), generator=generator) > 0.6).numpy()
        self.target[:40, :40] = True

    def test_counts_match_attention_to_mask(self):
        for normalization in ("minmax", "percentile_01_99"):
            counts, _ = cv.score_maps(self.maps, self.target, normalization)
            for c in range(self.maps.shape[0]):
                for t, threshold in enumerate(cv.THRESHOLDS):
                    pred = attention_to_mask(self.maps[c], self.target.shape, threshold, normalization).numpy()
                    truth = self.target
                    expected = [(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum(), (~pred & ~truth).sum()]
                    np.testing.assert_array_equal(counts[c, t], expected, err_msg=f"{normalization} c={c} t={threshold}")

    def test_normalization_matches_normalize_map(self):
        for normalization in ("minmax", "percentile_01_99"):
            batch = cv.normalize_batch(self.maps, normalization)
            for c in range(self.maps.shape[0]):
                torch.testing.assert_close(batch[c], normalize_map(self.maps[c], normalization), rtol=0, atol=0)

    def test_average_precision_matches_attention_analysis(self):
        from src.segmentation.attention_analysis import _average_precision, _target_at_size

        target = np.zeros((512, 512), bool)
        target[100:300, 150:400] = True
        _, ap = cv.score_maps(self.maps, target, "percentile_01_99")
        small = _target_at_size(target).flatten().numpy()
        torch.testing.assert_close(cv.target_at_size(target).flatten(), torch.from_numpy(small))
        for c in range(self.maps.shape[0]):
            scores = normalize_map(self.maps[c], "percentile_01_99").flatten().numpy()
            self.assertAlmostEqual(ap[c], _average_precision(scores, small), places=12)

    def test_empty_target_gives_nan_ap(self):
        _, ap = cv.score_maps(self.maps, np.zeros((64, 64), bool))
        self.assertTrue(np.isnan(ap).all())

    def test_score_masks_repeats_counts(self):
        masks = np.stack([self.target, ~self.target])
        counts = cv.score_masks(masks, self.target)
        self.assertEqual(counts.shape, (2, len(cv.THRESHOLDS), 4))
        np.testing.assert_array_equal(counts[0, 0], [self.target.sum(), 0, 0, (~self.target).sum()])
        np.testing.assert_array_equal(counts[:, 0], counts[:, 2])


class MetricTests(unittest.TestCase):
    def test_miou_and_image_iou_match_binary_segmentation_metrics(self):
        generator = torch.Generator().manual_seed(1)
        names = ("a", "b")
        n = 30
        classes = tuple(CLASSES[i % 3] for i in range(n))
        maps = torch.rand((len(names), 64, 64), generator=generator)
        counts = np.zeros((len(names), n, len(cv.THRESHOLDS), 4), dtype=np.int64)
        metrics = {name: BinarySegmentationMetrics() for name in names}
        for i in range(n):
            target = (torch.rand((64, 64), generator=generator) > 0.5).numpy()
            noisy = maps + 0.3 * torch.rand(maps.shape, generator=generator)
            counts[:, i], _ = cv.score_maps(noisy, target)
            for c, name in enumerate(names):
                metrics[name].update(classes[i], attention_to_mask(noisy[c], target.shape, 0.45, "minmax"), target)
        results = cv.Results(names, cv.THRESHOLDS, tuple(map(str, range(n))), classes,
                             np.arange(n) % 5, counts, np.zeros((2, n)), {})
        for c, name in enumerate(names):
            expected = metrics[name].result()
            self.assertAlmostEqual(cv.miou(results, c), expected["miou"], places=12)
            self.assertAlmostEqual(cv.image_ious(results, c).mean(), expected["mean_image_iou"], places=12)
            for k, v in cv.class_ious(results, c).items():
                self.assertAlmostEqual(v, expected["per_class"][k]["iou"], places=12)

    def test_paired_bootstrap(self):
        a = np.linspace(0.5, 0.9, 200)
        out = cv.paired_bootstrap(a, a - 0.01, resamples=2000)
        self.assertAlmostEqual(out["mean"], 0.01)
        self.assertLessEqual(out["ci_low"], 0.01 + 1e-12)
        self.assertEqual(out["better"], 200)
        self.assertEqual(cv.paired_bootstrap(a, a - 0.01, resamples=2000), out)


class StorageTests(unittest.TestCase):
    def test_writer_roundtrip_resume_and_guards(self):
        names = ("cfg/a", "cfg/b", "cfg/c")
        folds = {f"img-{i}": i % 5 for i in range(6)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.npz"
            writer = cv.ShardWriter(path, names, folds, {"family": "test"})
            rng = np.random.default_rng(0)
            stored = []
            for i in range(3):
                counts = rng.integers(0, 1000, size=(3, 3, 4))
                writer.add(f"img-{i}", "cat", counts, rng.random(3))
                stored.append(counts)
            with self.assertRaises(ValueError):
                writer.add("img-0", "cat", stored[0])
            with self.assertRaises(KeyError):
                writer.add("img-99", "cat", stored[0])
            with self.assertRaises(ValueError):
                writer.add("img-3", "cat", stored[0][:2])

            resumed = cv.ShardWriter(path, names, folds, {"family": "test"})
            self.assertEqual(resumed.completed, {"img-0", "img-1", "img-2"})
            resumed.add("img-3", "dog", stored[1])
            resumed.finish()
            with self.assertRaises(ValueError):
                cv.ShardWriter(path, names[:2], folds, {"family": "test"})
            with self.assertRaises(ValueError):
                cv.ShardWriter(path, names, folds, {"family": "other"})

            loaded = cv.load_results([path])
            self.assertEqual(loaded.configurations, names)
            self.assertEqual(loaded.image_ids, ("img-0", "img-1", "img-2", "img-3"))
            self.assertEqual(loaded.classes, ("cat", "cat", "cat", "dog"))
            np.testing.assert_array_equal(loaded.folds, [0, 1, 2, 3])
            np.testing.assert_array_equal(loaded.counts[:, 0], stored[0])
            np.testing.assert_array_equal(loaded.counts[:, 3], stored[1])
            self.assertEqual(loaded.complete, (True,))
            self.assertTrue(loaded.commits)
            records = list(loaded.records())
            self.assertEqual(len(records), 3 * 4)
            self.assertEqual(set(records[0]), {"image_id", "class", "fold", "configuration", "threshold",
                                               "tp", "fp", "fn", "tn", "image_iou", "ap"})

    def test_environment_is_recorded_per_session_and_merged(self):
        folds = {f"img-{i}": 0 for i in range(4)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.npz"
            writer = cv.ShardWriter(path, ("x",), folds, {})
            writer.add("img-0", "cat", np.ones((1, 3, 4)))
            env = cv.load_results([path]).environments
            self.assertEqual(len(env), 1)
            for key in ("torch", "torch_cuda", "gpu", "node", "commit", "diffusers", "python"):
                self.assertIn(key, env[0])
            self.assertEqual(env[0]["torch"], torch.__version__)
            # Same environment on resume: no new entry.
            cv.ShardWriter(path, ("x",), folds, {}).add("img-1", "cat", np.ones((1, 3, 4)))
            self.assertEqual(len(cv.load_results([path]).environments), 1)
            # Different environment on resume: a second entry, settings check unaffected.
            original = cv.runtime_environment
            cv.runtime_environment = lambda: {**original(), "torch": "0.0.0-test"}
            try:
                cv.ShardWriter(path, ("x",), folds, {}).add("img-2", "cat", np.ones((1, 3, 4)))
            finally:
                cv.runtime_environment = original
            envs = cv.load_results([path]).environments
            self.assertEqual([e["torch"] for e in envs], [torch.__version__, "0.0.0-test"])

    def test_shards_without_environment_still_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.npz"
            np.savez_compressed(
                path, configurations=np.array(["x"]), image_ids=np.array(["img-0"]), classes=np.array(["cat"]),
                folds=np.array([0], dtype=np.int8), thresholds=np.array(cv.THRESHOLDS),
                counts=np.ones((1, 1, 3, 4), dtype=np.int32), ap=np.zeros((1, 1), dtype=np.float32),
                settings=np.array("{}"), commits=np.array(["abc"]), complete=np.array(True),
            )
            loaded = cv.load_results([path])
            self.assertEqual(loaded.environments, ())
            self.assertEqual(loaded.commits, ("abc",))

    def test_load_rejects_mismatched_or_duplicate_shards(self):
        folds = {f"img-{i}": 0 for i in range(3)}
        with tempfile.TemporaryDirectory() as tmp:
            a = cv.ShardWriter(Path(tmp) / "a.npz", ("x", "y"), folds, {})
            a.add("img-0", "cat", np.ones((2, 3, 4)))
            b = cv.ShardWriter(Path(tmp) / "b.npz", ("x", "z"), folds, {})
            b.add("img-1", "cat", np.ones((2, 3, 4)))
            with self.assertRaises(ValueError):
                cv.load_results([Path(tmp) / "a.npz", Path(tmp) / "b.npz"])
            c = cv.ShardWriter(Path(tmp) / "c.npz", ("x", "y"), folds, {})
            c.add("img-0", "cat", np.ones((2, 3, 4)))
            with self.assertRaises(ValueError):
                cv.load_results([Path(tmp) / "a.npz", Path(tmp) / "c.npz"])


class FoldTests(unittest.TestCase):
    def test_make_folds_is_stratified_balanced_and_deterministic(self):
        rows = [{"image_id": f"img-{i}", "classname": CLASSES[i % 20], "annotated": "True"} for i in range(490)]
        rows += [{"image_id": "img-x", "classname": CLASSES[0], "annotated": "False"}]
        first = cv.make_folds(rows)
        self.assertEqual(first, cv.make_folds(list(reversed(rows))))
        self.assertEqual(len(first), 490)
        sizes = Counter(r["fold"] for r in first)
        self.assertLessEqual(max(sizes.values()) - min(sizes.values()), 1)
        per_class = Counter((r["classname"], r["fold"]) for r in first)
        self.assertLessEqual(max(per_class.values()) - min(per_class.values()), 1)

    def test_committed_fold_file(self):
        path = ROOT / "configs" / "voc_sim_folds.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 490)
        self.assertEqual(len({r["image_id"] for r in rows}), 490)
        self.assertEqual(Counter(r["fold"] for r in rows), Counter({str(k): 98 for k in range(5)}))
        per_class = Counter((r["classname"], r["fold"]) for r in rows)
        self.assertEqual(len({r["classname"] for r in rows}), 20)
        self.assertTrue(set(per_class.values()) <= {4, 5})


class DecisionTests(unittest.TestCase):
    def setUp(self):
        n = len(CLASSES) * 10
        base = np.full(n, 0.60)
        conditional = base + 0.007
        conditional[np.isin(np.arange(n) // 10, range(15, 20))] = 0.60  # only 15 classes improve
        harmful = base + 0.010
        harmful[:10] = 0.50  # one class loses 0.10
        self.results = synthetic_results({
            "ref": base,
            "promote": base + 0.02,
            "promote_close": base + 0.019,
            "conditional": conditional,
            "harmful": harmful,
            "tiny": base + 0.003,
        })

    def assess(self, name):
        return cv.assess(self.results, self.results.index("ref"), self.results.index(name))

    def test_verdicts(self):
        self.assertEqual(self.assess("promote")["verdict"], "promote")
        self.assertEqual(self.assess("conditional")["verdict"], "conditional")
        self.assertEqual(self.assess("conditional")["classes_improved"], 15)
        harmful = self.assess("harmful")  # +0.01 on 19 classes, -0.10 on one: mean +0.0045
        self.assertAlmostEqual(harmful["delta"], 0.0045)
        self.assertEqual(harmful["verdict"], "marginal")
        self.assertEqual(self.assess("tiny")["verdict"], "marginal")

    def test_conditional_requires_no_large_class_loss(self):
        n = len(CLASSES) * 10
        values = np.full(n, 0.61)
        values[:10] = 0.54  # one class loses 0.06 > 0.05
        results = synthetic_results({"ref": np.full(n, 0.6), "cand": values})
        out = cv.assess(results, 0, 1)
        self.assertAlmostEqual(out["delta"], 0.0065)
        self.assertEqual(out["classes_improved"], 19)
        self.assertEqual(out["verdict"], "marginal")
        values[:10] = 0.56  # loss 0.04: now conditional
        results = synthetic_results({"ref": np.full(n, 0.6), "cand": values})
        self.assertEqual(cv.assess(results, 0, 1)["verdict"], "conditional")

    def test_tie_prefers_more_inclusive(self):
        candidates = ["promote", "promote_close", "tiny"]
        self.assertEqual(cv.decide(self.results, "ref", candidates)["chosen"], "promote")
        chosen = cv.decide(self.results, "ref", candidates, {"promote": 1, "promote_close": 2})["chosen"]
        self.assertEqual(chosen, "promote_close")
        self.assertEqual(cv.decide(self.results, "ref", ["tiny"])["chosen"], "ref")


class CrossValidationTests(unittest.TestCase):
    def test_consistent_gain_is_frozen_and_oof_matches(self):
        n = len(CLASSES) * 10
        results = synthetic_results({"ref": np.full(n, 0.6), "better": np.full(n, 0.63)})
        out = cv.cross_validate(results, "ref", ["better"])
        self.assertEqual(out["frozen"], "better")
        self.assertEqual(out["frozen_votes"], 5)
        self.assertTrue(out["oof_guard_passed"])
        self.assertAlmostEqual(out["out_of_fold_miou_selection"], 0.63)
        self.assertAlmostEqual(out["out_of_fold_miou_reference"], 0.60)
        self.assertFalse(out["calibration_dependent"])

    def test_fold_specific_gain_is_not_frozen(self):
        n = len(CLASSES) * 10
        values = np.full(n, 0.6)
        folds = np.array([i % 10 % 5 for i in range(n)])
        values[np.isin(folds, [0, 1, 2])] += 0.01
        values[np.isin(folds, [3, 4])] -= 0.03
        results = synthetic_results({"ref": np.full(n, 0.6), "noisy": values})
        out = cv.cross_validate(results, "ref", ["noisy"])
        self.assertEqual(out["frozen"], "ref")
        self.assertEqual(out["selection_counts"], {"ref": 5})
        self.assertAlmostEqual(out["out_of_fold_miou_selection"], out["out_of_fold_miou_reference"])

    def test_gain_in_one_fold_only_is_rejected_by_oof_guard(self):
        """A gain confined to fold 0 wins the 4-of-5 vote but not the OOF guard.

        The four training sets that contain fold 0 all choose it, yet every
        held-out fold scores exactly like the reference, so the reference is
        kept (protocol Section 5a step 3).
        """
        n = len(CLASSES) * 10
        values = np.full(n, 0.6)
        folds = np.array([i % 10 % 5 for i in range(n)])
        values[folds == 0] += 0.06
        results = synthetic_results({"ref": np.full(n, 0.6), "lucky": values})
        out = cv.cross_validate(results, "ref", ["lucky"])
        self.assertEqual(out["selection_counts"], {"lucky": 4, "ref": 1})
        self.assertEqual(out["vote_winner"], "lucky")
        self.assertTrue(out["vote_passed"])
        self.assertAlmostEqual(out["oof_gain"], 0.0)
        self.assertFalse(out["oof_guard_passed"])
        self.assertEqual(out["frozen"], "ref")
        self.assertIn("OOF guard failed", out["freeze_reason"])

    def test_vote_and_guard_both_needed(self):
        n = len(CLASSES) * 10
        folds = np.array([i % 10 % 5 for i in range(n)])
        # Real but small held-out gain everywhere: vote 5/5 and positive OOF gain.
        results = synthetic_results({"ref": np.full(n, 0.6), "steady": np.full(n, 0.62)})
        out = cv.cross_validate(results, "ref", ["steady"])
        self.assertEqual(out["frozen"], "steady")
        self.assertTrue(out["vote_passed"] and out["oof_guard_passed"])
        self.assertEqual(out["freeze_reason"], "fold vote and OOF guard passed")
        # +0.04 on folds 0-1, -0.01 on folds 2-4: the three training sets that
        # hold out a losing fold promote it (+0.015), the two that hold out a
        # winning fold do not (+0.0025), so it wins only 3 of 5 votes.
        values = np.full(n, 0.6)
        values[np.isin(folds, [0, 1])] += 0.04
        values[np.isin(folds, [2, 3, 4])] -= 0.01
        out = cv.cross_validate(synthetic_results({"ref": np.full(n, 0.6), "split": values}), "ref", ["split"])
        self.assertEqual(out["selection_counts"], {"split": 3, "ref": 2})
        self.assertFalse(out["vote_passed"])
        self.assertEqual(out["frozen"], "ref")
        self.assertIn("fewer than 4 fold votes", out["freeze_reason"])

    def test_selector_script_end_to_end(self):
        n = len(CLASSES) * 10
        synthetic = synthetic_results({"ref": np.full(n, 0.6), "better": np.full(n, 0.63)})
        with tempfile.TemporaryDirectory() as tmp:
            folds = dict(zip(synthetic.image_ids, synthetic.folds.tolist()))
            for name in CLASSES[:20]:
                idx = [i for i, c in enumerate(synthetic.classes) if c == name]
                writer = cv.ShardWriter(Path(tmp) / f"{name}.npz", synthetic.configurations, folds, {"t": 1})
                for i in idx:
                    writer.add(synthetic.image_ids[i], name, synthetic.counts[:, i], synthetic.ap[:, i])
                writer.finish()
            spec = Path(tmp) / "spec.json"
            spec.write_text(json.dumps({"screens": [{"name": "S", "reference": "ref", "prefix": ""}]}))
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "select_controlled_cv.py"),
                            "--package-dir", tmp, "--spec", str(spec)], check=True, capture_output=True, cwd=ROOT)
            folds_out = json.loads((Path(tmp) / "fold_results.json").read_text())
            self.assertEqual(folds_out["screens"]["S"]["frozen"], "better")
            self.assertTrue((Path(tmp) / "summary.json").exists())
            report = (Path(tmp) / "report.md").read_text()
            self.assertIn("| S |", report)
            self.assertIn("fold vote and OOF guard passed", report)


if __name__ == "__main__":
    unittest.main()
