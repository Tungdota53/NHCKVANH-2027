from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from yolo_tuner.cli import parser
from yolo_tuner.pipeline import language_policy,optimize,select_presets
from yolo_tuner.runner import resolve_device
from yolo_tuner.vision_worker import weighted_epoch_time


class PortableTests(unittest.TestCase):
    def test_time_extrapolation_accounts_for_expensive_source_under_sampling(self):
        estimate=weighted_epoch_time(10,10,100,{"small":[2,8],"large":[2,2]},
                                     {"small":10,"large":90})
        self.assertAlmostEqual(estimate,152.5)
        self.assertGreater(estimate,100)  # The naive subset-only scaling misses I/O imbalance.

    def test_vocabulary_does_not_read_validation_or_test(self):
        manifest={"splits":{"train":[{"captions":["xin chào"]}],"val":[{"captions":["SECRET"]}],"test":[]}}
        vocabulary,_=language_policy(manifest)
        self.assertIn("xin",vocabulary)
        self.assertNotIn("secret",vocabulary)

    def test_no_cuda_no_mps_automatically_chooses_cpu(self):
        with patch("torch.cuda.is_available",return_value=False),patch("torch.backends.mps.is_available",return_value=False):
            self.assertEqual(resolve_device(),"cpu")

    def test_failed_cuda_probe_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available",return_value=True),patch("torch.ones",side_effect=RuntimeError("driver")), \
             patch("torch.backends.mps.is_available",return_value=False):
            self.assertEqual(resolve_device(),"cpu")

    def test_auto_selects_mps_when_available(self):
        with patch("torch.cuda.is_available",return_value=False),patch("torch.backends.mps.is_available",return_value=True), \
             patch("torch.ones"),patch("torch.mps.synchronize"):
            self.assertEqual(resolve_device(),"mps")

    def test_plan_only_saves_config_and_uses_available_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"raw";data.mkdir()
            for i in range(4):
                Image.new("RGB",(32,32),(i*20,0,0)).save(data/f"{i}.png")
            args=parser().parse_args(["auto","--data",str(data),"--plan-only","--output",str(root/"output")])
            hw={"device":"cpu","gpu":None,"fp16":False,"bf16":False,"ml_runtime_ready":False,
                "ram_available_mb":500,"logical_cpus":2,"vram_free_mb":None}
            with patch("yolo_tuner.pipeline.runtime_info",return_value=hw),redirect_stdout(io.StringIO()):
                self.assertEqual(optimize(args),0)
            file=next((root/"output").glob("*/selected.json"))
            selection=json.loads(file.read_text(encoding="utf-8"))
            self.assertEqual(selection["status"],"recommended_unverified")
            self.assertEqual(selection["task"],"reconstruction")
            self.assertEqual(selection["device"],"cpu")
            self.assertFalse(selection["trainable"])
            self.assertTrue((file.parent/"REPORT.md").exists())

    def test_generalized_pareto_does_not_confuse_caption_loss_with_map(self):
        def result(q,t,m):
            return {"status":"ok","resource_feasible":True,"quality_score":q,"estimated_full_seconds":t,"memory_mb":m}
        a,b,c=result(0.8,10,100),result(0.7,20,120),result(0.9,30,200)
        presets,front=select_presets([a,b,c])
        self.assertNotIn(b,front)
        self.assertIs(presets["Accuracy"],c)
        self.assertIs(presets["Time"],a)


if __name__=="__main__":
    unittest.main()
