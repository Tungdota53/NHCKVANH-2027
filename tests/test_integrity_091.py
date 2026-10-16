"""Integrity/timing regressions. Temporary fixtures are not ML evidence."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.integrity import sha
from research_pipeline.v2.file_integrity import VerifiedFileCache, content_hash
from research_pipeline.v2.schema import lock_protocol
from research_pipeline.v2.report import make_report
from research_pipeline.v2 import workflow
from research_pipeline.v2.budget import FinalizationReserve
import test_research_v2 as fixtures


class Integrity091Tests(unittest.TestCase):
    def test_finalization_estimate_uses_measured_checkpoint_bandwidth(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            path.write_bytes(b"x" * (4 * 1024 ** 2))
            clock = iter([0, 0.02])
            budget = FinalizationReserve(5, 1, clock=lambda: next(clock))
            budget.observe({"best_checkpoint": str(path), "checkpoint_sha256": sha(path)})
            self.assertEqual(budget.attempts, 1)
            self.assertEqual(len(budget.bulk_rates), 1)
            self.assertLess(budget.seconds(), 6)
            self.assertFalse(budget.summary()["upper_bound_guaranteed"])

    def test_finalization_calibration_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            path.write_bytes(b"changed fixture")
            budget = FinalizationReserve(5, 1)
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                budget.observe({"best_checkpoint": str(path), "checkpoint_sha256": "wrong"})

    def test_streaming_hash_matches_sha256_without_read_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large.bin"
            value = b"streaming fixture" * 180000
            path.write_bytes(value)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read forbidden")):
                self.assertEqual(sha(path), hashlib.sha256(value).hexdigest())

    def test_cache_skips_bytes_but_forced_boundary_rereads(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture"
            path.write_bytes(b"one")
            expected = sha(path)
            cache = VerifiedFileCache()
            cache.verify(path, expected)
            cache.verify(path, expected)
            self.assertEqual((cache.reads, cache.hits), (1, 1))
            cache.verify(path, expected, force=True)
            self.assertEqual((cache.reads, cache.hits), (2, 1))
            self.assertFalse(cache.stats()["anti_tamper_guarantee"])

    def test_modified_content_is_rejected_even_at_same_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture"
            path.write_bytes(b"one")
            expected = sha(path)
            cache = VerifiedFileCache()
            cache.verify(path, expected)
            old = path.stat()
            path.write_bytes(b"two")
            # Even preserved timestamps cannot evade the mandatory full boundary.
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "content changed"):
                cache.verify(path, expected, force=True)

    def test_model_identity_cache_invalidates_normal_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            path.write_bytes(b"not actual weights")
            first = content_hash(path)
            self.assertEqual(first, content_hash(path))
            path.write_bytes(b"different file length")
            self.assertNotEqual(first, content_hash(path))

    def policy(self):
        return {"version": 2, "integrity": "stat_cached_between_trials_full_hash_boundaries",
                "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
                "data_loader": {"workers": "auto", "cache_mb": 64, "pin_memory": "auto"}}

    def test_new_report_counts_primary_publication_once_and_old_is_unchanged(self):
        helper = fixtures.V2Tests()
        old = helper.protocol()
        modern = lock_protocol({k: v for k, v in {**old, "execution_policy": self.policy()}.items()
                                if k not in {"protocol_sha256", "schema_version", "adapter", "config_ids"}})
        state = {"results": {"0": [], "1": []}, "probes": []}
        events = [{"kind": "session_started", "payload": {}, "sha256": "fixture"},
                  {"kind": "selection_committed", "payload": {"session_seconds": 10}, "sha256": "fixture"},
                  {"kind": "publication_finished", "payload": {"session_seconds": 12}, "sha256": "fixture"}]
        self.assertEqual(make_report(old, state, events)["online_session_seconds"], 10)
        report = make_report(modern, state, events)
        self.assertEqual(report["online_session_seconds"], 12)
        self.assertFalse(report["hard_deadline_guaranteed"])

    def test_execution_policy_rejects_invalid_loader_settings(self):
        helper = fixtures.V2Tests()
        for update in ({"workers": -1}, {"workers": True}, {"cache_mb": float("nan")}, {"pin_memory": "yes"}):
            policy = self.policy()
            policy["data_loader"].update(update)
            with self.assertRaises((ValueError, TypeError)):
                helper.protocol(execution_policy=policy)

    def test_source_map_supports_installed_package_without_root_launcher(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "research_pipeline" / "v2").mkdir(parents=True)
            (root / "yolo_tuner").mkdir()
            (root / "research_pipeline" / "v2" / "__main__.py").write_text("pass", encoding="utf-8")
            with patch.object(workflow, "ROOT", root):
                mapping = workflow.source_map()
            self.assertEqual(list(mapping), ["research_pipeline/v2/__main__.py"])

    def test_cached_live_verification_still_full_hashes_boundaries(self):
        helper = fixtures.V2Tests()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            helper.fixture(folder)
            import json
            p = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
            image = folder / "image.bin"
            image.write_bytes(b"real content for hash fixture")
            manifest = {"splits": {"train": [{"image": str(image), "sha256": sha(image)}], "val": [], "test": []}}
            from yolo_tuner.pipeline import write_json
            from yolo_tuner.research import digest
            write_json(folder / "dataset.json", manifest)
            p["dataset_sha256"] = digest(manifest)
            p["protocol_sha256"] = digest({k: v for k, v in p.items() if k != "protocol_sha256"})
            cache = VerifiedFileCache()
            with patch.object(workflow, "runtime", return_value={}):
                workflow.verify_live(folder, p, cache, force=True)
                workflow.verify_live(folder, p, cache)
                self.assertEqual((cache.reads, cache.hits), (1, 1))
                workflow.verify_live(folder, p, cache, force=True)
                self.assertEqual(cache.reads, 2)

    def test_recovered_probe_cannot_bypass_saved_exploration_charge(self):
        import json
        from yolo_tuner.pipeline import write_json
        from yolo_tuner.research import digest
        helper = fixtures.V2Tests()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _, ledger, state = helper.fixture(folder)
            protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
            protocol["scheduler"]["policy_version"] = 2
            write_json(folder / "protocol.json", protocol)
            state["probe_budget_spent_seconds"] = 0.25
            write_json(folder / "state.json", state)
            trial = folder / "trials" / "recovered_probe"
            trial.mkdir()
            result = {"mode": "search", "index": 0, "attempt_id": "recovered-probe", "status": "ok"}
            write_json(trial / "parent_result.json", result)
            ledger.emit("trial_started", attempt_id=result["attempt_id"])
            ledger.emit("trial_completed", attempt_id=result["attempt_id"], directory=str(trial),
                        parent_sha256=sha(trial / "parent_result.json"))
            workflow.reconcile_state(folder)
            recovered = json.loads((folder / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(recovered["probes"]), 1)
            self.assertGreaterEqual(recovered["probe_budget_spent_seconds"],
                                    protocol["budget_seconds"] * protocol["scheduler"]["probe_fraction"])
            self.assertIn("conservatively exhausted", recovered["probe_budget_recovery"])


if __name__ == "__main__":
    unittest.main()
