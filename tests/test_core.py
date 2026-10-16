import unittest

from yolo_tuner.core import Config, Deadline, choose_presets, memory_batch_estimate, pareto, successive_halving
from yolo_tuner.offline import predict, quantile_upper


def row(config, epoch, accuracy=0.5, seconds=100, memory=1000, status="ok"):
    from dataclasses import asdict
    return {"config": asdict(config), "epochs_completed": epoch, "status": status,
            "map50_proxy": accuracy, "estimated_full_seconds": seconds, "peak_reserved_mb": memory}


class CoreTests(unittest.TestCase):
    def test_memory_proposal_is_bounded_and_noise_safe(self):
        self.assertEqual(memory_batch_estimate([(1, 1000), (2, 1500)], 6000), 4)
        self.assertEqual(memory_batch_estimate([(1, 1500), (2, 1400)], 6000), 2)
        self.assertEqual(memory_batch_estimate([(1, 1000), (2, 1500)], 1800), 1)

    def test_pareto_and_empty_feasible_set(self):
        config = Config("a")
        good = row(config, 1)
        dominated = row(config, 1, 0.4, 150, 1200)
        tradeoff = row(config, 1, 0.7, 200, 1600)
        self.assertEqual(pareto([good, dominated, tradeoff]), [good, tradeoff])
        self.assertEqual(choose_presets([good], 50), ({}, []))
        self.assertEqual(choose_presets([good])[0]["Balanced"], good)

    def test_interrupted_rung_retains_equal_fidelity_cohort(self):
        configs = [Config(str(i)) for i in range(4)]
        def evaluate(c, epochs, remaining):
            if epochs == 2 and c.model == "1":
                return row(c, 1, status="timeout")
            return row(c, epochs, 0.9 - 0.1 * int(c.model))
        cohort, history, complete = successive_halving(configs, [1, 2, 4], evaluate, Deadline(60))
        self.assertFalse(complete)
        self.assertEqual(len(cohort), 4)
        self.assertEqual({r["epochs_completed"] for r in cohort}, {1})
        self.assertEqual(len(history), 6)

    def test_failed_first_rung_returns_no_recommendation(self):
        cohort, _, _ = successive_halving([Config("a")], [1],
                                         lambda c, e, _: row(c, 0, status="oom"), Deadline(10))
        self.assertEqual(cohort, [])

    def test_successive_halving_uses_epochs_not_wall_time(self):
        calls = []
        def evaluate(c, e, _):
            calls.append(e)
            return row(c, e, float(c.model) / 10)
        cohort, _, complete = successive_halving([Config(str(i)) for i in range(4)], [1, 2, 4], evaluate, Deadline(10))
        self.assertTrue(complete)
        self.assertEqual(calls, [1, 1, 1, 1, 2, 2, 4])
        self.assertEqual(cohort[0]["config"]["model"], "3")

    def test_accumulation_does_not_claim_exact_effective_batch(self):
        c = Config("a", batch=20)
        self.assertEqual(c.payload()["effective_batch_after_warmup"], 60)

    def test_conformal_finite_sample_correction(self):
        self.assertEqual(quantile_upper(list(range(9)), 0.1), 8)
        with self.assertRaises(ValueError):
            quantile_upper([1, 2], 0.1)

    def test_deadline_validation_and_clock(self):
        now = [0]
        d = Deadline(5, clock=lambda: now[0])
        now[0] = 7
        self.assertEqual(d.remaining, 0)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Deadline(value)


if __name__ == "__main__":
    unittest.main()
