import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_pipeline.cli import parser
from research_pipeline.methods import search, stratified_order
from research_pipeline.report import sensitivity, summarize
from research_pipeline.runner import actual_matches
from research_pipeline.study import exclusive_run, grid, model_snapshot, require_idle, full_reference
from research_pipeline.integrity import sha, verify_raw, verify_live, hardware_changes
from yolo_tuner.research import Objective, digest, nondominated


class DetectionResearchTests(unittest.TestCase):
    def config(self, **kwargs):
        return {"model": "local-model.yaml", "imgsz": 320, "batch": 2, "precision": "fp32",
                "accumulation": 8, "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, **kwargs}

    def protocol(self, mode="independent_research", environment="uncontended_preflight"):
        return {"configs": [self.config(), self.config(lr0=0.0003)], "training_seeds": [17, 29, 43],
                "objective": {"name": "balanced"}, "mode": mode,
                "measurement_environment": environment, "weight_scenarios": {"balanced": [0.6,0.25,0.15]},
                "hardware": {"device": "cpu"}}

    def results(self, complete=True):
        return {f"c{i:02d}": [{"training_seed": s, "full_completed": True, "status": "ok",
                  "quality_score": 0.6+i*0.1, "worker_wall_seconds": 10, "memory_mb": 50}
                 for s in ([17,29,43] if complete or i == 0 else [17])] for i in range(2)}

    def test_detection_parser_extends_without_mutating_legacy(self):
        self.assertEqual(parser().parse_args(["research","--data","x","--task","detection"]).task,"detection")
        self.assertEqual(parser().parse_args(["verify-config","--data","x"]).methods,["adaptive_verified"])
        from yolo_tuner.cli import parser as old
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                old().parse_args(["research","--data","x","--task","detection"])

    def test_actual_config_checks_every_reported_parameter(self):
        config=self.config(); actual={k:v for k,v in config.items() if k!="model"}
        self.assertTrue(actual_matches(config,actual))
        for k,v in {"batch":1,"imgsz":640,"precision":"fp16","accumulation":1,
                    "optimizer":"SGD","lr0":0.01,"weight_decay":0.01}.items():
            self.assertFalse(actual_matches(config,{**actual,k:v}),k)
        self.assertFalse(actual_matches(config,{}))

    def test_positive_weight_optimum_is_nondominated(self):
        rows=[{"config_id":"a","quality_mean":0.5,"quality_min":0.5,"time_mean":20,"time_max":20,"memory_max":50},
              {"config_id":"b","quality_mean":0.6,"quality_min":0.6,"time_mean":10,"time_max":10,"memory_max":40}]
        objective=Objective()
        self.assertGreater(objective.value(rows[1]),objective.value(rows[0]))
        self.assertEqual([r["config_id"] for r in nondominated(rows)],["b"])

    def test_strata_retains_all_indices_and_covers_lrs_first(self):
        configs=[self.config(lr0=v) for v in (0.001,0.001,0.0003,0.0003)]
        output=stratified_order([0,1,2,3],configs)
        self.assertEqual(output,[0,2,1,3])
        self.assertEqual(set(output),set(range(4)))

    def ablation(self, method):
        now=[0]; calls=[]
        def evaluator(index,epoch,remaining,full):
            calls.append((index,full)); now[0]+=1
            return {"status":"ok","epochs_completed":epoch,"quality_score":0.2+index*0.1,
                    "worker_wall_seconds":1,"memory_mb":50,"estimated_full_seconds":1}
        result=search(method,[self.config(),self.config(lr0=0.0003)],evaluator,Objective(),100,[1,2],4,17,clock=lambda:now[0])
        return result,calls

    def test_no_profile_ablation_has_only_full_runs_and_charges_all(self):
        decision,calls=self.ablation("adaptive_no_profile")
        self.assertTrue(all(full for _,full in calls))
        self.assertEqual(decision["search_seconds"],len(calls))
        self.assertEqual(decision["selected"],"c01")

    def test_no_strata_ablation_profiles_then_confirms(self):
        decision,calls=self.ablation("adaptive_no_strata")
        self.assertEqual([full for _,full in calls],[False,False,True,True])
        self.assertEqual(decision["search_seconds"],4)

    def test_full_reference_is_not_allowed_before_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"locked"):
                full_reference(Path(temp),{}, {},None,{})

    def test_contention_blocks_benchmark_but_explicit_smoke_is_possible(self):
        with patch("research_pipeline.study.competing_jobs",return_value=[{"pid":123}]):
            with self.assertRaisesRegex(ValueError,"Other ML jobs"):
                require_idle()
            self.assertEqual(require_idle(True),[{"pid":123}])

    def test_partial_grid_only_chooses_complete_config_not_higher_partial(self):
        report=summarize(self.protocol("online_verification"),[],self.results(False),100)
        self.assertEqual(report["verification_status"],"fully_trained_best_observed")
        self.assertEqual(report["empirical_best"],"c00")
        self.assertIsNone(report["candidates"][1]["utility"])
        self.assertEqual(report["candidates"][1]["state"],"unresolved")

    def test_complete_online_grid_has_no_free_oracle_or_regret(self):
        report=summarize(self.protocol("online_verification"),[],self.results(),100)
        self.assertEqual(report["verification_status"],"empirical_optimal_in_declared_space")
        self.assertEqual(report["offline_reference_seconds"],0)
        self.assertEqual(report["online_confirmation_seconds"],60)
        self.assertEqual(report["scores"],[])
        self.assertIsNone(report["recommendation"]["test_map50_95"])
        self.assertIsNone(report["recommendation"]["predicted_memory_mb"])
        self.assertEqual(report["recommendation"]["memory_kind"],"process RSS; NOT VRAM")

    def test_contended_measurements_never_export_certified_selection(self):
        report=summarize(self.protocol(environment="contended_smoke"),[],self.results(),100)
        self.assertIsNone(report["empirical_best"])
        self.assertEqual(report["verification_status"],"contended_smoke_not_certified")
        self.assertEqual(report["recommendation"]["full_training_verification"],"smoke_only_contended")

    def test_no_complete_seed_config_does_not_select_proxy(self):
        results=self.results()
        for key in results:
            results[key]=results[key][:1]
        report=summarize(self.protocol(),[],results,100)
        self.assertIsNone(report["empirical_best"])
        self.assertIsNone(report["recommendation"])

    def test_memory_feasibility_uses_worst_observed_seed(self):
        p=self.protocol(); p["objective"]={"max_memory":70}
        results=self.results(); results["c01"][-1]["memory_mb"]=80
        report=summarize(p,[],results,100)
        self.assertEqual(report["empirical_best"],"c00")
        self.assertEqual(report["candidates"][1]["state"],"confirmed_constraint_failure")

    def test_sensitivity_does_not_mutate_primary_objective(self):
        rows=[{"config_id":"a","quality_mean":0.9,"quality_min":0.9,"time_mean":100,"time_max":100,"memory_max":80},
              {"config_id":"b","quality_mean":0.8,"quality_min":0.8,"time_mean":10,"time_max":10,"memory_max":50}]
        objective=Objective(); before=objective.weights
        output=sensitivity(rows,objective,{"accuracy":[1,0,0],"time":[0,1,0]})
        self.assertEqual([r["selected"] for r in output],["a","b"])
        self.assertEqual(objective.weights,before)

    def test_local_model_snapshot_has_identical_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"model.yaml"; source.write_text("test model bytes")
            result=Path(model_snapshot(str(source),Path(temp)/"snapshots"))
            self.assertEqual(source.read_bytes(),result.read_bytes())
            self.assertNotEqual(source,result)

    def test_missing_checkpoint_does_not_trigger_network_download(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"local"):
                model_snapshot("not-downloaded.pt",Path(temp))

    def test_installed_architecture_scale_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(model_snapshot("yolo11s.yaml",Path(temp)))
            import yaml
            self.assertEqual(yaml.safe_load(file.read_text())["scale"],"s")

    def test_grid_rejects_auto_optimizer_and_nonintegral_accumulation(self):
        hw={"device":"cpu","fp16":False}
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"grid.json"
            args=SimpleNamespace(precision="fp32",models=None,grid=str(file),space="compact")
            for bad in ({"optimizer":"auto"},{"accumulation":1.5},{"imgsz":65},{"precision":"bf16","batch":0}):
                file.write_text(json.dumps([bad]))
                with self.assertRaises(ValueError):
                    grid(args,hw,Path(temp))

    def test_active_parent_lock_is_not_removed(self):
        import os
        import psutil
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"run.lock"
            file.write_text(json.dumps({"pid":os.getpid(),"create_time":psutil.Process().create_time()}))
            with self.assertRaisesRegex(ValueError,"still running"):
                with exclusive_run(Path(temp)):
                    self.fail("must not acquire live lock")
            self.assertTrue(file.exists())

    def test_active_worker_blocks_even_without_parent_lock(self):
        import os
        import psutil
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"process.json"
            file.write_text(json.dumps({"pid":os.getpid(),"create_time":psutil.Process().create_time(),"status":"running"}))
            with self.assertRaisesRegex(ValueError,"worker is still running"):
                with exclusive_run(Path(temp)):
                    self.fail("must not restart beside live orphan worker")

    def raw_fixture(self, folder):
        protocol=self.protocol(); protocol["source_files"]={}; protocol["full_epochs"]=8
        manifest={"task":"detection","splits":{}}
        protocol["dataset_sha256"]=digest(manifest)
        protocol["protocol_sha256"]=digest(protocol)
        trial=folder/"reference"/"c00"/"seed_17"/"trial_0001"; trial.mkdir(parents=True)
        checkpoint=trial/"best.pt"; checkpoint.write_bytes(b"unit-test-placeholder-not-a-real-model")
        config=protocol["configs"][0]
        request={"config":config,"seed":17,"trial_epochs":8,"full_epochs":8,"mode":"train","resume":None,
                 "manifest":str(folder/"dataset.json"),"data":str(folder/"full_data"/"data.yaml")}
        raw={"config":config,"status":"ok","quality_score":0.6,"best_quality":0.6,"memory_mb":50,"epochs_completed":8,
             "actual":{k:v for k,v in config.items() if k!="model"},"best_checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint)}
        parent={**raw,"worker_wall_seconds":10,"log":str(trial/"worker.log")}
        decisions=[{"method":"default","seed":17,"history":[]}]
        files={folder/"dataset.json":manifest,folder/"decisions.locked.json":decisions,
               trial/"request.json":request,trial/"result.json":raw,trial/"parent_result.json":parent}
        for path,value in files.items():
            path.write_text(json.dumps(value))
        rows=[{"config_id":"c00","runs":[{**parent,"training_seed":17,"full_completed":True}]},
              {"config_id":"c01","runs":[]}]
        return protocol,decisions,rows,trial

    def test_raw_audit_checks_original_actual_config_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); protocol,decisions,rows,trial=self.raw_fixture(folder)
            self.assertEqual(verify_raw(folder,protocol,decisions,rows),1)
            raw=json.loads((trial/"result.json").read_text()); raw["best_quality"]=0.99
            (trial/"result.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,"Raw result changed"):
                verify_raw(folder,protocol,decisions,rows)

    def test_raw_audit_rejects_hidden_subset_in_full_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); protocol,decisions,rows,trial=self.raw_fixture(folder)
            req=json.loads((trial/"request.json").read_text()); req["data"]=str(folder/"subset_data"/"data.yaml")
            (trial/"request.json").write_text(json.dumps(req))
            with self.assertRaisesRegex(ValueError,"Actual data request"):
                verify_raw(folder,protocol,decisions,rows)

    def test_raw_audit_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); protocol,decisions,rows,trial=self.raw_fixture(folder)
            (trial/"best.pt").write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError,"Checkpoint"):
                verify_raw(folder,protocol,decisions,rows)

    def test_exported_annotation_hash_is_checked_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); label=folder/"label.txt"; label.write_text("original label")
            manifest={"splits":{}}
            protocol={"source_files":{},"model_files":{},"runtime_versions":{},"dataset_sha256":digest(manifest),
                      "data_files":{"label.txt":sha(label)}}
            protocol["protocol_sha256"]=digest(protocol)
            with patch("research_pipeline.integrity.runtime",return_value={}):
                verify_live(protocol,manifest,folder=folder)
                label.write_text("changed label")
                with self.assertRaisesRegex(ValueError,"Exported training data"):
                    verify_live(protocol,manifest,folder=folder)

    def test_independent_study_locks_decisions_before_six_reference_jobs(self):
        from research_pipeline.study import study
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"run"; folder.mkdir()
            model=folder/"model.yaml"; model.write_text("unit placeholder")
            configs=[self.config(model=str(model)),self.config(model=str(model),lr0=0.0003)]
            manifest={"source":"mock-unit-test-not-real-data","task":"detection","validation_ready":True,
                      "seed":17,"names":["person"],"splits":{"train":[],"val":[],"test":[]},
                      "counts":{"train":10,"val":5,"test":0}}
            args=parser().parse_args(["research","--data","mock","--task","detection","--methods","default"])
            calls=[]
            def fake(folder_arg,protocol,manifest_arg,index,epochs,seconds,seed,location,full):
                self.assertTrue((folder/"decisions.locked.json").exists())
                self.assertTrue(full)
                calls.append((index,seed))
                return {"config":configs[index],"status":"ok","epochs_completed":epochs,"quality_score":0.5+index*0.1,
                        "memory_mb":50,"worker_wall_seconds":10,"log":str(location/"fake.log")}
            def fake_export(data,destination):
                destination.mkdir(parents=True)
                (destination/"data.yaml").write_text("unit mock export, not a real dataset")
                return destination/"data.yaml"
            with patch("research_pipeline.study.competing_jobs",return_value=[]),patch("research_pipeline.study.new_session",return_value=folder),\
                 patch("research_pipeline.study.runtime_info",return_value={"device":"cpu","ml_runtime_ready":True,"ram_available_mb":1000}),\
                 patch("research_pipeline.study.grid",return_value=configs),patch("research_pipeline.study.verify_live"),\
                 patch("research_pipeline.study.export_yolo",side_effect=fake_export),\
                 patch("research_pipeline.study.evaluate",side_effect=fake):
                self.assertEqual(study(args,manifest),0)
            self.assertEqual(len(calls),6)
            report=json.loads((folder/"study.json").read_text())
            self.assertEqual(report["offline_reference_seconds"],60)
            self.assertEqual(report["empirical_best"],"c01")
            self.assertTrue(report["reference_complete"])

    def test_unverified_same_named_gpu_is_not_same_hardware_evidence(self):
        hw={"device":"0","gpu":"same GPU name","gpu_uuid":None,"gpu_identity_verified":False}
        self.assertIn("physical GPU identity unverified",hardware_changes(hw,dict(hw)))

    def test_same_verified_gpu_and_runtime_has_no_scope_change(self):
        hw={"device":"0","gpu_uuid":"GPU-real-test-identity","gpu_identity_verified":True,"host_label":"host-a"}
        self.assertEqual(hardware_changes(hw,dict(hw)),[])

    def test_cpu_host_move_is_detected_even_with_identical_cpu_name(self):
        hw={"device":"cpu","cpu":"same CPU","host_label":"host-a"}
        self.assertIn("host_label",hardware_changes(hw,{**hw,"host_label":"host-b"}))

    def test_training_evidence_requires_both_loss_and_optimizer_update(self):
        from research_pipeline.runner import training_evidence_complete
        self.assertTrue(training_evidence_complete({"optimizer_state_created":True,"training_loss_finite":True}))
        for evidence in ({},{"optimizer_state_created":False,"training_loss_finite":True},
                         {"optimizer_state_created":True,"training_loss_finite":False}):
            self.assertFalse(training_evidence_complete(evidence))

    def test_finite_losses_rejects_missing_and_nan_or_inf(self):
        from research_pipeline.worker import finite_losses
        import torch
        self.assertTrue(finite_losses({"box":torch.tensor(1.0),"cls":torch.tensor(0.3)}))
        self.assertTrue(finite_losses(torch.tensor([1.0,2.0])))
        self.assertFalse(finite_losses(None))
        self.assertFalse(finite_losses({}))
        self.assertFalse(finite_losses([float("nan")]))
        self.assertFalse(finite_losses(torch.tensor([float("inf")])) )

    def test_root_help_exposes_new_commands(self):
        from research_pipeline.cli import main
        output=io.StringIO()
        with patch("sys.argv",["train_config.py","--help"]),patch("sys.stdout",output),self.assertRaises(SystemExit) as stopped:
            main()
        self.assertEqual(stopped.exception.code,0)
        self.assertIn("evaluate-detection",output.getvalue())
        self.assertIn("research-start",output.getvalue())

    def test_audit_accepts_json_null_key_in_selection_counts(self):
        from research_pipeline.cli import audit
        protocol=self.protocol(); protocol.update(full_epochs=4,dataset_source="mock-unit-fixture")
        decisions=[{"method":"adaptive_verified","seed":s,"selected":None if s!=43 else "c01",
                    "finalists":[] if s!=43 else ["c01"],"history":[],"search_seconds":1} for s in [17,29,43]]
        original=json.loads(json.dumps(summarize(protocol,decisions,self.results(),100)))
        self.assertIn("null",original["summaries"][0]["selected_configs"])
        with tempfile.TemporaryDirectory() as temp:
            with patch("research_pipeline.cli.read_run",return_value=(Path(temp),original,0)):
                self.assertEqual(audit(temp),0)

    def test_detection_resume_only_missing_full_jobs_and_preserves_failed_cost(self):
        from research_pipeline.study import resume
        from yolo_tuner.research import reference_cost
        import platform
        protocol={"configs":[self.config(),self.config(lr0=0.0003)],"training_seeds":[17],
                  "full_epochs":4,"split_seed":101,"trial_timeout":30,"hardware":{"device":"cpu","host_label":platform.node()},
                  "measurement_environment":"uncontended_preflight"}
        results={"c00":[{"training_seed":17,"full_completed":True}],
                 "c01":[{"training_seed":17,"full_completed":False,"status":"timeout","worker_wall_seconds":10}]}
        calls=[]
        def fake(folder,proto,manifest,index,epochs,seconds,seed,location,full):
            calls.append((index,seed,epochs,full))
            return {"status":"ok","epochs_completed":epochs,"worker_wall_seconds":2}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name,value in {"protocol.json":protocol,"dataset.json":{},"decisions.locked.json":[],
                               "reference.partial.json":results,"study.json":{"elapsed_seconds":5}}.items():
                (folder/name).write_text(json.dumps(value))
            with patch("research_pipeline.study.verify_live"),patch("research_pipeline.study.require_idle"),\
                 patch("research_pipeline.study.runtime_info",return_value={"device":"cpu","host_label":platform.node()}),\
                 patch("research_pipeline.study.evaluate",side_effect=fake),patch("research_pipeline.study.publish",return_value=0) as published:
                self.assertEqual(resume(SimpleNamespace(run=temp,timeout_hours=1)),0)
            self.assertEqual(calls,[(1,17,4,True)])
            row=published.call_args.args[3]["c01"][0]
            self.assertTrue(row["full_completed"])
            self.assertEqual(reference_cost(row),12)
            metadata=json.loads((folder/"resume.json").read_text())
            self.assertFalse(metadata["search_replayed"])
            self.assertFalse(metadata["downtime_included"])


if __name__ == "__main__":
    unittest.main()
