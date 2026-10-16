import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from yolo_tuner.cli import parser
from yolo_tuner.research import Objective, aggregate, default_space, run_search, study
from yolo_tuner.runner import hardware
from yolo_tuner.verification import content_identity, hardware_identity, summarize_evidence, verification_outcome


class VerificationTests(unittest.TestCase):
    def row(self,key,q,complete=True):
        return {"config_id":key,"complete":complete,"quality_mean":q,"quality_min":q,
                "time_mean":10,"time_max":10,"memory_max":20}

    def test_verify_parser_does_not_mutate_research_defaults(self):
        p=parser()
        research=p.parse_args(["research","--data","example"])
        verified=p.parse_args(["verify-config","--data","example"])
        self.assertEqual(research.methods,["default","random","successive_halving","pareto_racing"])
        self.assertEqual(research.space,"compact")
        self.assertEqual(research.search_budget,2)
        self.assertEqual(verified.methods,["adaptive_verified"])
        self.assertEqual(verified.space,"expanded")
        self.assertEqual(verified.search_budget,10)

    def test_verify_help_has_online_cost_and_only_production_method(self):
        p=parser()
        output=io.StringIO()
        with patch("sys.stdout",output), self.assertRaises(SystemExit) as stopped:
            p.parse_args(["verify-config","--help"])
        self.assertEqual(stopped.exception.code,0)
        help_text=" ".join(output.getvalue().split())
        self.assertIn("Online minutes per training seed",help_text)
        self.assertIn("no offline oracle",help_text)
        self.assertNotIn("offline reference is separate",help_text)
        with patch("sys.stderr",io.StringIO()), self.assertRaises(SystemExit):
            p.parse_args(["verify-config","--data","example","--methods","random"])
        self.assertEqual(p.parse_args(["research","--data","example","--methods","random"]).methods,["random"])

    def test_late_starter_is_not_eliminated_by_proxy(self):
        now=[0]; calls=[]
        configs=[{"lr0":0.001},{"lr0":0.0003},{"lr0":0.003}]
        def evaluate(index,epochs,remaining,full):
            calls.append((index,epochs,full)); now[0]+=1
            q=(0.99 if index==1 else 0.8) if full else (0.01 if index==1 else 0.9)
            return {"status":"ok","epochs_completed":epochs,"quality_score":q,
                    "estimated_full_seconds":10,"memory_mb":20,"worker_wall_seconds":1}
        decision=run_search("adaptive_verified",configs,evaluate,Objective("quality"),100,[1,2,4],8,17,clock=lambda:now[0])
        self.assertEqual(decision["selected"],"c01")
        self.assertEqual({i for i,e,f in calls if f},{0,1,2})
        self.assertTrue(decision["exhaustive_full_data"])
        self.assertFalse(decision["projection_is_bound"])

    def test_insufficient_budget_never_certifies_proxy(self):
        now=[0]
        def evaluate(index,epochs,remaining,full):
            self.assertFalse(full); now[0]+=1
            return {"status":"ok","epochs_completed":epochs,"quality_score":0.9,
                    "estimated_full_seconds":100,"memory_mb":20,"worker_wall_seconds":1}
        result=run_search("adaptive_verified",[{"lr0":0.001},{"lr0":0.0003}],evaluate,Objective(),5,[1,2],8,17,clock=lambda:now[0])
        self.assertIsNone(result["selected"])
        self.assertFalse(result["exhaustive_full_data"])
        self.assertEqual(result["unverified_configs"],["c00","c01"])

    def test_lr_stratification_retains_all_candidates_without_quality_pruning(self):
        now=[0]; called=[]
        configs=[{"lr0":lr} for lr in (0.001,0.001,0.0003,0.0003)]
        def evaluate(index,epochs,remaining,full):
            now[0]+=1
            if full:
                called.append(configs[index]["lr0"])
            return {"status":"ok","epochs_completed":epochs,"quality_score":0.4,
                    "estimated_full_seconds":10,"memory_mb":20,"worker_wall_seconds":1}
        result=run_search("adaptive_verified",configs,evaluate,Objective(),25,[1,2],8,17,clock=lambda:now[0])
        self.assertEqual(result["strategy"],"lr_stratified_full_confirmation")
        self.assertNotEqual(called[0],called[1])

    def test_partial_space_is_not_an_optimality_certificate(self):
        complete,best,status=verification_outcome([self.row("c00",0.8),self.row("c01",0.9,False)],Objective("quality"))
        self.assertFalse(complete)
        self.assertEqual(best["config_id"],"c00")
        self.assertEqual(status,"fully_trained_best_observed")
        self.assertEqual(verification_outcome([self.row("c00",0.8,False)],Objective())[2],"no_verified_feasible_selection")

    def test_complete_space_can_certify_only_declared_empirical_optimum(self):
        complete,best,status=verification_outcome([self.row("c00",0.8),self.row("c01",0.9)],Objective("quality"))
        self.assertTrue(complete)
        self.assertEqual(best["config_id"],"c01")
        self.assertEqual(status,"empirical_optimal_in_declared_space")

    def test_dataset_identity_ignores_paths_but_not_labels_or_splits(self):
        data={"task":"classification","names":["a","b"],"splits":{"train":[{"image":"old","sha256":"abc","class_id":0}],"val":[]}}
        moved={**data,"splits":{"train":[{"image":"new","sha256":"abc","class_id":0}],"val":[]}}
        relabeled={**data,"splits":{"train":[{"image":"old","sha256":"abc","class_id":1}],"val":[]}}
        resplit={**data,"splits":{"train":[],"val":data["splits"]["train"]}}
        self.assertEqual(content_identity(data),content_identity(moved))
        self.assertNotEqual(content_identity(data),content_identity(relabeled))
        self.assertNotEqual(content_identity(data),content_identity(resplit))

    def test_cpu_and_cuda_on_same_host_are_not_two_gpus(self):
        rows=[{"protocol_sha256":"a","reference_complete":True,"dataset_content_id":"one",
               "hardware":hardware_identity({"device":"0","gpu_uuid":"real-gpu","gpu":"3050"}),"runtime":{}},
              {"protocol_sha256":"b","reference_complete":True,"dataset_content_id":"two",
               "hardware":hardware_identity({"device":"cpu","cpu":"same host CPU"}),"runtime":{}}]
        report=summarize_evidence(rows+rows)
        self.assertEqual(report["distinct_physical_cuda_gpus"],1)
        self.assertEqual(report["complete_runs"],2)
        self.assertEqual(report["unseen_gpu_validation"],"not_established")
        self.assertFalse(report["universal_optimality_guarantee"])

    def test_unknown_gpu_identity_is_not_counted_as_a_new_physical_gpu(self):
        rows=[{"protocol_sha256":"a","reference_complete":True,"dataset_content_id":"one",
               "hardware":hardware_identity({"device":"0","gpu":"generic name"}),"runtime":{}}]
        self.assertEqual(summarize_evidence(rows)["distinct_physical_cuda_gpus"],0)

    def mock_cuda(self,device_uuid):
        return [patch("yolo_tuner.runner.resolve_device",return_value="0"),
                patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.set_device"),
                patch("torch.cuda.mem_get_info",return_value=(1000000000,2000000000)),
                patch("torch.cuda.get_device_capability",return_value=(8,6)),
                patch("torch.cuda.get_device_properties",return_value=SimpleNamespace(uuid=device_uuid)),
                patch("torch.cuda.get_device_name",return_value="actual CUDA adapter"),
                patch("torch.cuda.is_bf16_supported",return_value=True)]

    def test_cuda_logical_index_is_resolved_by_uuid_not_physical_index(self):
        from contextlib import ExitStack
        target="11111111-2222-3333-4444-555555555555"
        with ExitStack() as stack:
            for mock in self.mock_cuda(target):
                stack.enter_context(mock)
            nvml=stack.enter_context(patch("yolo_tuner.runner._nvml_gpu",return_value={"gpu_uuid":"GPU-"+target,"gpu_identity_verified":True}))
            info=hardware("0")
        nvml.assert_called_once_with(0,device_uuid="GPU-"+target)
        self.assertEqual(info["gpu_uuid"],"GPU-"+target)

    def test_no_cuda_uuid_does_not_guess_nvml_physical_adapter(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for mock in self.mock_cuda(None):
                stack.enter_context(mock)
            nvml=stack.enter_context(patch("yolo_tuner.runner._nvml_gpu"))
            info=hardware("0")
        nvml.assert_not_called()
        self.assertIsNone(info["gpu_uuid"])
        self.assertFalse(hardware_identity(info)["identity_verified"])

    def test_nvml_failure_preserves_training_but_not_physical_identity_claim(self):
        from contextlib import ExitStack
        target="11111111-2222-3333-4444-555555555555"
        with ExitStack() as stack:
            for mock in self.mock_cuda(target):
                stack.enter_context(mock)
            stack.enter_context(patch("yolo_tuner.runner._nvml_gpu",side_effect=RuntimeError("NVML unavailable")))
            info=hardware("0")
        self.assertTrue(info["cuda_training_ready"])
        self.assertFalse(info["gpu_identity_verified"])

    def test_expanded_space_matches_cpu_gpu_hyperparameters_when_precision_locked(self):
        dataset={"counts":{"train":1000}}
        a=default_space("classification",{"device":"cpu","ram_available_mb":100,"fp16":False},dataset,8,"expanded")
        b=default_space("classification",{"device":"0","ram_available_mb":5000,"fp16":True},dataset,8,"expanded")
        self.assertEqual([{**c,"precision":"fp32"} for c in a],[{**c,"precision":"fp32"} for c in b])
        self.assertEqual(len(a),6)

    def test_online_verification_does_not_run_a_hidden_offline_oracle(self):
        hw={"device":"cpu","ml_runtime_ready":True,"ram_available_mb":1000,"fp16":False,"bf16":False}
        manifest={"source":"mock-not-benchmark","task":"classification","validation_ready":True,
                  "splits":{"train":[],"val":[],"test":[]},"counts":{"train":10,"val":5,"test":0},"names":["a","b"]}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"session"; folder.mkdir(); calls=[]
            class FakeRunner:
                def __init__(self,path,*args):
                    assert "reference" not in Path(path).parts
                    self.index=0
                def run(self,config,epochs,seconds,mode):
                    calls.append((config,epochs,mode))
                    return {"status":"ok","epochs_completed":epochs,"quality_score":0.5,
                            "estimated_full_seconds":1,"worker_wall_seconds":1,"memory_mb":100,"config":config}
            args=SimpleNamespace(command="verify-config",data="mock",task="auto",device="cpu",seeds=[17,29,43],methods=["adaptive_verified"],split_seed=101,
                                 full_epochs=8,rungs=[1,2,4],train_subset=10,val_subset=5,trial_timeout=30,
                                 search_budget=1,timeout_hours=1,output=temp,grid=None,max_memory=None,space="expanded",precision=None,
                                 objective="quality",min_quality=0,max_time=None,time_scale=600,memory_scale=1024,weights=[0.6,0.25,0.15])
            with patch("yolo_tuner.pipeline.new_session",return_value=folder),patch("yolo_tuner.pipeline.runtime_info",return_value=hw),patch("yolo_tuner.adapters.discover",return_value=manifest),patch("yolo_tuner.adapters.subset",return_value=manifest),patch("yolo_tuner.pipeline.VisionRunner",FakeRunner),patch("yolo_tuner.research.cli_ui.trial"),patch("builtins.print"):
                self.assertEqual(study(args),0)
            report=json.loads((folder/"study.json").read_text(encoding="utf-8"))
            self.assertTrue(report["reference_complete"])
            self.assertEqual(report["offline_reference_seconds"],0)
            self.assertEqual(sum(mode=="train" for c,e,mode in calls),18)
            self.assertFalse(report["independent_method_regret_available"])
            self.assertEqual({d["ordering_seed"] for d in report["decisions"]},{101})


if __name__=="__main__":
    unittest.main()
