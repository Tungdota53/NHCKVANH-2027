import unittest
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from yolo_tuner.research import Objective, aggregate, hypervolume, run_search, score_decisions, nondominated, digest, verify_protocol, deployment_scope_changes, resume, reference_cost, study, verify_dataset_files, verify_source


class ResearchTests(unittest.TestCase):
    def row(self, key, q, t, m):
        return {"config_id":key,"complete":True,"quality_mean":q,"quality_min":q,
                "time_mean":t,"time_max":t,"memory_max":m}

    def test_objective_has_fixed_scales_not_oracle_fitted_normalization(self):
        obj=Objective(weights=(0.6,0.25,0.15),time_scale=100,memory_scale=100)
        row=self.row("c00",0.8,20,10)
        self.assertAlmostEqual(obj.value(row),0.48-0.05-0.015)
        with self.assertRaises(ValueError):
            Objective(time_scale=0)
        with self.assertRaises(ValueError):
            Objective(weights=(0,0,0))

    def test_feasibility_uses_worst_observed_seed_not_mean_only(self):
        obj=Objective(min_quality=0.7,max_time=10,max_memory=100)
        row=self.row("c00",0.8,9,80)
        self.assertTrue(obj.feasible(row))
        self.assertFalse(obj.feasible({**row,"quality_min":0.6}))
        self.assertFalse(obj.feasible({**row,"time_max":11}))
        self.assertFalse(obj.feasible({**row,"complete":False,"time_mean":None}))

    def test_aggregate_requires_all_distinct_full_seeds(self):
        def result(seed,q):
            return {"training_seed":seed,"full_completed":True,"best_quality":q,"quality_score":0.1,
                    "worker_wall_seconds":10,"memory_mb":20}
        one=aggregate("c00",[result(17,0.8)],[17,29,43])
        self.assertFalse(one["complete"])
        full=aggregate("c00",[result(17,0.8),result(29,0.6),result(43,0.7)],[17,29,43])
        self.assertTrue(full["complete"])
        self.assertAlmostEqual(full["quality_mean"],0.7)
        self.assertAlmostEqual(full["quality_sd"],0.1)
        self.assertEqual(full["quality_min"],0.6)
        self.assertFalse(aggregate("c00",[result(17,0.8),result(17,0.8),result(29,0.6),result(43,0.7)],[17,29,43])["complete"])

    def test_hypervolume_exact_union_with_shared_reference(self):
        obj=Objective(time_scale=1,memory_scale=1)
        a=self.row("a",0.5,0.5,0.5)
        b=self.row("b",1.0,0.75,0.75)
        self.assertAlmostEqual(hypervolume([a],obj),0.125)
        self.assertAlmostEqual(hypervolume([a,b],obj),0.125+0.0625-0.03125)
        self.assertEqual(hypervolume([],obj),0)
        self.assertEqual(hypervolume([self.row("c",0.8,2,2)],obj),0)

    def test_regret_is_full_outcome_gap_and_infeasible_is_not_fake_zero(self):
        obj=Objective("quality",min_quality=0.6)
        reference=[self.row("c00",0.7,10,20),self.row("c01",0.9,12,30),self.row("c02",0.5,8,10)]
        decisions=[{"method":"default","seed":1,"selected":key,"finalists":[key],"history":[],"search_seconds":0}
                   for key in ("c00","c01","c02")]
        best,scores,_=score_decisions(decisions,reference,obj)
        self.assertEqual(best["config_id"],"c01")
        self.assertAlmostEqual(scores[0]["objective_regret"],0.2)
        self.assertEqual(scores[1]["objective_regret"],0)
        self.assertFalse(scores[2]["feasible"])
        self.assertIsNone(scores[2]["objective_regret"])

    def test_default_does_not_query_evaluator_or_oracle(self):
        def forbidden(*args):
            raise AssertionError("Default must not consult outcomes")
        decision=run_search("default",[{},{}],forbidden,Objective(),60,[1,2],4,17)
        self.assertEqual(decision["selected"],"c00")
        self.assertEqual(decision["history"],[])

    def test_interrupted_rung_keeps_equal_fidelity_cohort(self):
        now=[0]
        def evaluate(index,epochs,remaining,full):
            now[0]+=1
            return {"status":"timeout" if epochs==2 else "ok","epochs_completed":epochs,
                    "quality_score":0.4+index*0.1,"estimated_full_seconds":10,"memory_mb":10,"worker_wall_seconds":1}
        result=run_search("successive_halving",[{},{}],evaluate,Objective(),60,[1,2],4,17,clock=lambda:now[0])
        self.assertEqual(result["selected"],"c01")
        self.assertEqual(result["history"][-1]["status"],"timeout")

    def test_random_uses_full_data_and_never_counts_timeout_as_full(self):
        now=[0]; calls=[]
        def evaluate(index,epochs,remaining,full):
            calls.append(full); now[0]+=10
            return {"status":"ok" if len(calls)==1 else "timeout","epochs_completed":epochs,
                    "quality_score":0.5,"estimated_full_seconds":10,"memory_mb":10,"worker_wall_seconds":10}
        result=run_search("random",[{},{}],evaluate,Objective(),15,[1,2],4,17,clock=lambda:now[0])
        self.assertTrue(all(calls))
        self.assertEqual(len(result["finalists"]),1)

    def test_protocol_edit_is_detected(self):
        protocol={"full_epochs":8,"configs":[{}]}
        protocol["protocol_sha256"]=digest(protocol)
        verify_protocol(protocol)
        with self.assertRaises(ValueError):
            verify_protocol({**protocol,"full_epochs":20})

    def test_source_change_cannot_silently_mix_implementations(self):
        with self.assertRaisesRegex(ValueError,"source changed"):
            verify_source({"source_sha256":{"research.py":"0"*64}})

    def test_dataset_bytes_change_invalidates_recorded_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"fixture.bin"; path.write_bytes(b"initial")
            manifest={"splits":{"train":[{"image":str(path),"sha256":hashlib.sha256(b"initial").hexdigest()}]}}
            verify_dataset_files(manifest)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"image changed"):
                verify_dataset_files(manifest)

    def test_random_proxy_is_a_separate_matched_fidelity_control(self):
        now=[0]; calls=[]
        def evaluate(index,epochs,remaining,full):
            calls.append((epochs,full)); now[0]+=1
            return {"status":"ok","epochs_completed":epochs,"quality_score":0.5+index*0.1,
                    "estimated_full_seconds":10,"memory_mb":10,"worker_wall_seconds":1}
        result=run_search("random_proxy",[{},{}],evaluate,Objective("quality"),10,[1,2,4],8,17,clock=lambda:now[0])
        self.assertEqual(calls,[(4,False),(4,False)])
        self.assertEqual(result["selected"],"c01")
        self.assertIn("proxy",result["selection_basis"])

    def test_proxy_error_compares_same_seed_not_aggregate_mean(self):
        truth=self.row("c00",0.8,20,40)
        truth["runs"]=[{"training_seed":17,"full_completed":True,"best_quality":0.6,"worker_wall_seconds":10,"memory_mb":20},
                       {"training_seed":29,"full_completed":True,"best_quality":1.0,"worker_wall_seconds":30,"memory_mb":40}]
        decision={"method":"successive_halving","seed":17,"selected":"c00","finalists":["c00"],"search_seconds":1,
                  "history":[{"config_id":"c00","complete":True,"full_data":False,"quality":0.5,"estimated_time":15,"memory":10,"status":"ok"}]}
        _,scores,_=score_decisions([decision],[truth],Objective())
        self.assertAlmostEqual(scores[0]["proxy_quality_mae"],0.1)
        self.assertAlmostEqual(scores[0]["time_mape"],0.5)
        self.assertAlmostEqual(scores[0]["memory_relative_error"],0.5)

    def test_changed_epoch_or_runtime_loses_optimality_scope(self):
        config={"epochs":8,"batch":4,"feature_cache":"old"}
        manifest={"source":"fixed","splits":{}}
        hw={"device":"cpu","torch_version":"2.6"}
        protocol={"full_epochs":8,"training_seeds":[17,29,43],"configs":[config],"hardware":hw,"dataset_sha256":digest(manifest)}
        protocol["protocol_sha256"]=digest(protocol)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"study.json"
            path.write_text(json.dumps({"protocol":protocol,"empirical_best":"c00","reference_complete":True}),encoding="utf-8")
            selection={"research_provenance":str(path),"full_epochs":8,"seed":17}
            self.assertEqual(deployment_scope_changes(selection,{**config,"feature_cache":"relocated"},manifest,hw),[])
            selection["full_epochs"]=20
            self.assertIn("epoch horizon",deployment_scope_changes(selection,config,manifest,hw))
            self.assertIn("hardware/runtime: torch_version",deployment_scope_changes(selection,config,manifest,{**hw,"torch_version":"2.7"}))

    def test_resume_only_missing_reference_and_keeps_failed_cost(self):
        hw={"device":"cpu","torch_version":"2.6"}
        manifest={"splits":{"train":[],"val":[]},"names":[],"task":"classification"}
        protocol={"configs":[{"batch":1},{"batch":2}],"training_seeds":[17],"full_epochs":8,
                  "split_seed":17,"dataset_sha256":digest(manifest),"hardware":hw}
        protocol["protocol_sha256"]=digest(protocol)
        decisions=[{"method":"default","seed":17,"selected":"c00"}]
        calls=[]
        class FakeRunner:
            def __init__(self,folder,*args):
                Path(folder).mkdir(parents=True,exist_ok=True)
            def run(self,config,epochs,seconds,mode):
                calls.append((config,epochs,mode))
                return {"status":"ok","epochs_completed":epochs,"worker_wall_seconds":2}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            files={"protocol.json":protocol,"dataset.json":manifest,"decisions.locked.json":decisions,
                   "status.json":{"phase":"full_reference","decisions_sha256":digest(decisions)},
                   "reference.partial.json":{"c00":[{"training_seed":17,"full_completed":True}],
                                             "c01":[{"training_seed":17,"full_completed":False,"status":"timeout","worker_wall_seconds":10}]}}
            for name,value in files.items():
                (folder/name).write_text(json.dumps(value),encoding="utf-8")
            with patch("yolo_tuner.pipeline.runtime_info",return_value=hw),patch("yolo_tuner.pipeline.VisionRunner",FakeRunner),patch("yolo_tuner.research.cli_ui.trial"),patch("yolo_tuner.research.publish",return_value=0) as final:
                self.assertEqual(resume(SimpleNamespace(run=temp,timeout_hours=1,trial_timeout=30)),0)
            self.assertEqual(calls,[({"batch":2},8,"train")])
            rows=final.call_args.args[3]
            self.assertTrue(rows["c01"][0]["full_completed"])
            self.assertEqual(reference_cost(rows["c01"][0]),12)
            self.assertEqual(json.loads((folder/"decisions.locked.json").read_text(encoding="utf-8")),decisions)

    def test_incomplete_search_cannot_resume_with_reset_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); protocol={"configs":[{}]}
            protocol["protocol_sha256"]=digest(protocol)
            (folder/"protocol.json").write_text(json.dumps(protocol),encoding="utf-8")
            (folder/"status.json").write_text(json.dumps({"phase":"search"}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"not locked"):
                resume(SimpleNamespace(run=temp,timeout_hours=1,trial_timeout=30))

    def test_study_locks_decisions_before_any_full_reference(self):
        hw={"device":"cpu","ml_runtime_ready":True,"ram_available_mb":1000,"fp16":False,"bf16":False}
        manifest={"source":"mock-protocol-unit-test-not-real-data","task":"classification","validation_ready":True,
                  "splits":{"train":[],"val":[],"test":[]},"counts":{"train":10,"val":5,"test":0},"names":["a","b"]}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"session"; folder.mkdir()
            reference_started=[]
            class FakeRunner:
                def __init__(self,path,*args):
                    reference_started.append(Path(path))
                    assert (folder/"decisions.locked.json").exists()
                    assert not (folder/"study.json").exists()
                def run(self,config,epochs,seconds,mode):
                    return {"status":"ok","epochs_completed":epochs,"quality_score":0.5,
                            "worker_wall_seconds":10,"memory_mb":100,"config":config}
            args=SimpleNamespace(data="mock",task="auto",device="cpu",seeds=[17,29,43],methods=["default"],split_seed=17,
                                 full_epochs=8,rungs=[1,2,4],train_subset=10,val_subset=5,trial_timeout=30,
                                 search_budget=1,timeout_hours=1,output=temp,grid=None,max_memory=None,
                                 objective="quality",min_quality=0,max_time=None,time_scale=600,memory_scale=1024,weights=[0.6,0.25,0.15])
            with patch("yolo_tuner.pipeline.new_session",return_value=folder),patch("yolo_tuner.pipeline.runtime_info",return_value=hw),patch("yolo_tuner.adapters.discover",return_value=manifest),patch("yolo_tuner.adapters.subset",return_value=manifest),patch("yolo_tuner.pipeline.VisionRunner",FakeRunner),patch("yolo_tuner.research.cli_ui.trial"),patch("builtins.print"):
                self.assertEqual(study(args),0)
            self.assertEqual(len(reference_started),12)
            report=json.loads((folder/"study.json").read_text(encoding="utf-8"))
            self.assertTrue(report["reference_complete"])
            self.assertEqual(report["summaries"][0]["objective_regret_mean"],0)
            self.assertEqual(report["offline_reference_seconds"],120)
            self.assertTrue((folder/"selected.json").exists())


if __name__=="__main__":
    unittest.main()
