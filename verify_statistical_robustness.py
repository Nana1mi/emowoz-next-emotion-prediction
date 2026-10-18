"""Synthetic checks only; no real experimental results or training."""
import itertools
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import analyze_statistical_robustness as audit


class StatisticalAuditTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([0, 1, 2, 3, 4, 5, 6, 0, 4, 2, 1, 6])
        self.pred = np.array([0, 0, 2, 4, 4, 0, 6, 1, 2, 2, 1, 6])
        self.groups = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3])
        self.counts = audit.sufficient(self.truth, self.pred, self.groups, 4)

    def assert_metrics(self, counts, truth, pred):
        actual = audit.metric(counts)
        self.assertAlmostEqual(actual[0], f1_score(truth, pred, labels=list(range(7)), average="macro", zero_division=0))
        self.assertAlmostEqual(actual[1], accuracy_score(truth, pred))
        self.assertAlmostEqual(actual[2], f1_score(np.isin(truth, [1, 2, 4]), np.isin(pred, [1, 2, 4]), zero_division=0))
        np.testing.assert_allclose(actual[3:], f1_score(truth, pred, labels=list(range(7)), average=None, zero_division=0))

    def test_metrics_match_sklearn(self):
        self.assert_metrics(self.counts.sum(axis=0), self.truth, self.pred)

    def test_duplicate_whole_dialogues_matches_row_expansion(self):
        weights = np.array([2, 0, 3, 1])
        indices = np.repeat(np.arange(len(self.truth)), weights[self.groups])
        self.assert_metrics(weights @ self.counts, self.truth[indices], self.pred[indices])

    def test_dialogue_swap_matches_actual_prediction_swap(self):
        pred_b = self.pred[::-1].copy()
        stats_b = audit.sufficient(self.truth, pred_b, self.groups, 4)
        for swapped in itertools.product([False, True], repeat=4):
            swapped = np.array(swapped)
            a = np.where(swapped[self.groups], pred_b, self.pred)
            b = np.where(swapped[self.groups], self.pred, pred_b)
            signs = 1 - 2 * swapped.astype(int)
            middle = ((self.counts + stats_b) / 2).sum(axis=0)
            delta = signs @ ((self.counts - stats_b) / 2)
            self.assert_metrics(middle + delta, self.truth, a)
            self.assert_metrics(middle - delta, self.truth, b)

    def test_identical_model_has_zero_paired_interval_and_p_one(self):
        stats = np.stack([self.counts, self.counts], axis=1)[:, :, None, :]
        stats = np.repeat(stats, 3, axis=2)
        fixed, crossed, _ = audit.resample(stats, 200, np.random.default_rng(9), 31)
        np.testing.assert_array_equal(fixed[:, 0], fixed[:, 1])
        np.testing.assert_array_equal(crossed[:, 0], crossed[:, 1])
        self.assertEqual(audit.permutation(stats, [(0, 1)], 200, np.random.default_rng(10), 31), [1.0])

    def test_bootstrap_is_reproducible(self):
        stats = self.counts[:, None, None, :]
        a = audit.resample(stats, 100, np.random.default_rng(22), 17)
        b = audit.resample(stats, 100, np.random.default_rng(22), 17)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_holm(self):
        np.testing.assert_allclose(audit.holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
        np.testing.assert_allclose(audit.holm([1, 1, 1]), [1, 1, 1])

    def test_zero_support_convention(self):
        np.testing.assert_array_equal(audit.metric(np.zeros(24)), np.zeros(10))

    def test_three_seed_two_sided_resolution(self):
        signs = np.array(list(itertools.product([-1, 1], repeat=3)))
        effect = np.array([0.01, 0.02, 0.03])
        p = np.mean(np.abs((signs * effect).mean(axis=1)) >= abs(effect.mean()) - 1e-12)
        self.assertEqual(p, 0.25)

    def test_real_source_subsets_match_independent_sklearn(self):
        report_path = audit.ROOT / "统计稳健性审计_20261015/统计结果_all.json"
        if not report_path.exists():
            self.skipTest("Run the real audit first for independent subset verification")
        outputs = {scope: audit.read(audit.ROOT / f"统计稳健性审计_20261015/统计结果_{scope}.json")
                   for scope in ["all", "multiwoz", "dialmage"]}
        for name in [model["model"] for model in outputs["all"]["models"]]:
            stage, prefix = name.split("/")
            for seed_i, seed in enumerate(audit.SEEDS):
                path = audit.ROOT / "runs" / stage / "models" / f"{prefix}_seed{seed}" / "test_predictions.jsonl"
                with path.open(encoding="utf-8") as stream:
                    rows = [audit.json.loads(line) for line in stream]
                for scope, output in outputs.items():
                    subset = rows if scope == "all" else [r for r in rows if r["domain"] == scope]
                    y = [r["label"] for r in subset]
                    pred = [audit.LABELS.index(r["predicted_emotion"]) for r in subset]
                    result = next(m for m in output["models"] if m["model"] == name)
                    self.assertAlmostEqual(f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0),
                                           result["metrics"]["macro_f1"]["by_seed"][seed_i], places=12)

    def test_saved_distributions_match_reported_intervals(self):
        directory = audit.ROOT / "统计稳健性审计_20261015"
        if not (directory / "完成核验.json").exists():
            self.skipTest("Run the real audit first for saved-distribution verification")
        for scope in ["all", "multiwoz", "dialmage"]:
            output = audit.read(directory / f"统计结果_{scope}.json")
            with np.load(directory / f"bootstrap分布_{scope}.npz") as arrays:
                self.assertEqual(arrays["models"].tolist(), [m["model"] for m in output["models"]])
                self.assertEqual(arrays["metrics"].tolist(), audit.METRICS)
                for key, field in [("fixed_seed_mean", "dialogue_percentile_ci95"),
                                   ("dialogue_seed_sensitivity", "dialogue_seed_resampling_sensitivity_ci95")]:
                    quantiles = np.quantile(arrays[key], [0.025, 0.975], axis=0)
                    for i, model in enumerate(output["models"]):
                        for j, metric in enumerate(audit.METRICS):
                            np.testing.assert_allclose(quantiles[:, i, j], model["metrics"][metric][field],
                                                       rtol=0, atol=5e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
