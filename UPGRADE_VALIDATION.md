# Kiểm chứng nâng cấp nghiên cứu 0.7

Ngày 13/10/2026. Đây là nhật ký triển khai/thực nghiệm, không tự chứng nhận đã hoàn thành toàn bộ nghiên cứu nhiều GPU trong [RESEARCH_UPGRADE_PLAN.md](RESEARCH_UPGRADE_PLAN.md).

## Lõi đã triển khai

- Protocol schema, task-specific metric, content-based candidate IDs, actual configuration/optimizer-step/loss/checkpoint verification.
- Một shared budget giữa các training seeds; incumbent/full-confirmation reserve; probe gating; không quality-prune theo loss sớm. 0.7.1 thêm initial probe có giới hạn và tránh probe cap nhỏ hơn startup quan sát.
- Event ledger hash chain, raw audit, immutable committed state, reconciliation của completed outcomes chưa ghi state; unknown hard-crash cost không được gán 0 để replay.
- CPU process-tree RSS và CUDA reserved/allocated; train/validation/checkpoint phase wall time; NVML diagnostics nullable, không nhầm allocator với total VRAM.
- CLI inspect/doctor/plan-search/optimize/status/audit/resume/benchmark/calibrate/train-v2/evaluate-v2, root/console routing, installer tạo venv mới không overwrite môi trường hiện có.
- Dataset 0.7.1 hỗ trợ normalized JSON manifest, external-path YOLO YAML/COCO; declared groups không được cross-split; detection cần positive train/validation boxes. Unknown true groups/near-duplicate risk ghi rõ.
- Predictor card T/M point fitting + explicit disjoint domain partitions + source/runtime/hardware checks + residual diagnostics. Chưa có validated safety interval hoặc quality surrogate/NEHVI.

Package `yolo_tuner` 0.5 và extension source 0.6 giữ nguyên; root entry/project metadata đổi phiên bản. Snapshots/runs/checkpoints lịch sử không sửa. Source guard vẫn áp dụng, không bypass để live-resume protocol cũ bằng mã mới.

## Tests và smoke thật

162 tests trên runtime CUDA; CPU cùng suite có 1 CUDA-specific test skip vì không có CUDA runtime. Dependency checks 48 packages tương thích ở mỗi runtime; compileall qua. Mock outcomes trong tests chỉ kiểm logic, không là dữ liệu benchmark.

- CPU Fashion-MNIST: `runs/v2_smoke_cpu/20261013_100056_754125`, 1 full worker 2 epoch; optimizer_steps=64, finite loss/actual config/checkpoint đã qua; validation accuracy=0.523333; time=5.11s, peak process-tree RSS=289.29MiB. Audit 1 attempt. Chỉ là execution smoke, không quality certificate.
- CUDA Penn-Fudan: `runs/v2_smoke_gpu/20261013_100219_797095`, 1 full worker 2 epoch; optimizer_steps=16; mAP50-95=0.49332; time=21.78s, reserved=182MiB. NVML device usage peak≈568MiB là diagnostic khác counter, process VRAM unavailable. Audit 1 attempt.
- Captioning ban đầu lỗi strict deterministic adaptive-pool backward trên CUDA: `runs/v2_smoke_caption/20261013_101126_405847`. Không có optimizer update, không cấp selection. Giữ nguyên log/raw failure, không sửa thành Passed.
- Sửa pooling bằng explicit adaptive-cell means, test forward/gradient tương đương CPU float64 và strict CUDA backward qua. `runs/v2_smoke_caption_fixed/20261013_101428_895127`: genuine ViECAP development subset 64/16/16 images, 2 epoch, seed17, tiny spatial-attention encoder. Full worker qua, Q=(generated BLEU4+chrF)/200=0.253595, time=12.39s, reserved=146MiB. Không gọi đó là full ViECAP 2773/944/331 hoặc blind test mới. Audit 1 attempt.

Lượt chuẩn bị caption subset đầu chỉ có external annotations nhưng không local images nên adapter cũ từ chối. Helper đã tạo local hardlinks/copies nhãn/ảnh thật vào output mới; không overwrite output lỗi hoặc dataset nguồn. 0.7.1 còn hỗ trợ trực tiếp external-path manifests.

## Benchmark CPU thật — 12 configs × 3 required seeds × 4 epoch

Artifact: `runs/benchmark_v2_fashion/20261013_102239_404492/BENCHMARK.md` và `BENCHMARK.json`.

Dataset: Fashion-MNIST genuine subset 1000 train/300 val/500 test, source labels/split giữ nguyên; test không dùng. Grid core có controlled LR/batch-accum/image/width contrasts. Shared wall budget 90s mỗi search; 3 search seeds 17/29/43; mỗi feasible choice phải đủ full training seeds 17/29/43. Cold subprocess matched, method order rotated.

| Method | Feasible selections | Conditional regret mean | Mean online wall(s) |
|---|---:|---:|---:|
| Reserve 0.7.0 | 3/3 | 0.034610 | 80.65 |
| Random full | 3/3 | 0.040116 | 85.81 |
| Profile all | 2/3 | 0.029716 trên 2 feasible choices | 85.82 |

Không gọi profile-all tốt hơn dựa trên conditional regret thấp hơn: một lượt không có choice và regret null. Reserve có observed regret thấp hơn Random trong pilot này, nhưng 3 repeats trên 1 domain không đủ statistical superiority hoặc mọi hardware. Repeat43 reserve cũng không chọn được cấu hình tốt nhất đã thấy ở method khác.

Independent reference: **36/36 full outcomes**, mọi C có đủ 3 seeds. Best empirical weighted U: width32/image32/batch16/accum2/FP32/AdamW/LR0.001/decay0.01; val accuracy mean=0.711111±SD0.012620; worker mean=7.3127s; RSS max=295.14MiB; U=0.380386. Runner-up utility margin chỉ 0.000882: không chứng minh expected ranking chắc chắn trên future seeds/timing.

Reference full-worker sum=302.393s; session tới selection commit=346.281s, cost riêng với online search. Regret lấy hai U từ cùng independent reference, không trừ online measurements rồi clip giá trị âm.

### Hai lỗi công cụ đã được ghi rõ, không sửa lịch sử

1. Reference preparation ban đầu từ chối objective vì tuple weights trong memory khác list sau JSON, dù cùng giá trị. Sửa comparison bằng canonical digest; decisions cũ đã khóa không replay/đổi. Reference sau sửa chỉ khác `workflow.py`/`benchmark.py`, **worker/runner/model/metric sources giữ cùng SHA**. Comparison recovery dùng explicit controller-only allowance và công bố source differences; source khác ở training bị từ chối.
2. Standalone recovery helper ban đầu không bật UTF-8 Windows, nên lỗi print tiếng Việt **sau khi 36 workers/selection/report đã commit**. Error/status gốc giữ nguyên, không đổi thành successful parent exit. Raw audit đủ 36; phân tích sử dụng committed outcomes và thời điểm selection, không cộng lại cả session như một lần training nữa. Helper đã sửa UTF-8 và hỗ trợ analysis-only reference. 0.7.1 public report bảo đảm UTF-8 và tách post-commit export failure khỏi interrupted search.

Reference artifact: `reference_after_controller_fix/20261013_104126_958950` trong benchmark folder. `BENCHMARK.json` chứa recovery/source/publication notes. Đây là exploratory development evidence, không final blind benchmark.

## 0.7.1 online detection rộng hơn — đã hoàn tất

`runs/verified_detection_071/20261013_111148_405825`: Penn-Fudan 130/20/20, controlled 6 configs, 3 required seeds, 8 epoch. B ban đầu1200s: 15 full workers +5 probes, actual session933.406s; còn image224 config unresolved theo forecast nên certificate là best observed, không fake full-space.

Resume với additional B900s, audit trước skip; chỉ config thiếu được chạy (1 probe +3 full workers), không replay 15 full workers cũ. **18/18 full confirmations**, **24 original attempts audit**, offline oracle0. Cumulative session1215.781s, worker attempts1185.499s trong đó probes63.623s; total budget được cấp2100s. Không gọi1215s là nằm trong B1200 ban đầu, không giấu additional grant; downtime không được tính như training.

| Trường | Kết quả đo | Phạm vi |
|---|---|---|
| Hardware | RTX3050 Laptop6GB | Một physical GPU UUID |
| Selected | YOLO11n pretrained COCO | C6, E8, seeds17/29/43 |
| Image / batch / precision |160 /4 /FP16 mixed |Actual checked |
| Accumulation / nominal effective batch |4 /16 |Optimizer updates thực sự quan sát |
| Optimizer / initial LR / decay |AdamW /0.0003 /0.0005 |No optimizer auto / no OOM fallback |
| Val mAP50-95 mean±SD |0.699570±0.005916 |Không confidence interval |
| Full worker T mean |56.51s |8epochs/130train/20val, không thời gian cả search |
| M max required seeds |318MiB CUDA reserved |Không total VRAM hoặc RSS |
| U / runner-up margin |0.349614 /0.005290 |Weights/anchors khóa trước |
| Test mAP50-95 /mAP50 |0.684766 /0.886755 |Seed17, FP32 inference,20test images |
| Verification |empirical_optimal_in_declared_space |Không ngoài C/H/P/seeds/objective này |

Test artifact `evaluation/20261013_113512_945206/EVALUATION.md`; không đổi selection theo test. Same-seed validation=0.69605, không so test1seed với mean3seeds như paired3seed estimate. Penn test đã từng exposed ở engineering0.6; global blindness không được chứng nhận. Không quy khác biệt với old4epoch study chỉ cho policy: horizon/grid/initialLR/batch và instrumentation khác.

## Predictor T/M — đã có diagnostics 3-domain độc lập

Card `runs/predictor_v2_cpu.json`; partitions khai báo tại `configs/predictor_v2_partitions.json` trước fitting:

- Fit: audited Fashion reference36 full outcomes.
- Calibration diagnostics: `runs/calibration_v2_cifar/20261013_113609_587025`, 6configs×3seeds×4epoch,18/18full, CPU, no test used.
- Heldout predictor evaluation: `runs/evaluation_v2_mnist/20261013_114036_315958`, 6configs×3seeds×4epoch,18/18full; genuine MNIST1000/300/500, original labels, content-disjoint official-train/official-test indices, archive hashes ở PROVENANCE.json. [Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) là API nguồn. MNIST diagnostics không dùng để retune fitted coefficients/policy.

Một task classification / memory kind process-tree RSS / CPU host/runtime. Body/runner/model/metric source signatures giống nhau; controller/version-only changes không được dùng để giả A/T/M tương đương khi worker sources khác. Có72 audited full resource outcomes nhưng chỉ3dataset/hardware domains, không72independent domains. Source signature/runtime/hardware mismatch vẫn fallback.

| Partition | Full rows | Time MAE(s) | Memory MAE(MiB) | Time underprediction rate |
|---|---:|---:|---:|---:|
| Fit Fashion |36|0.52246|1.10365|41.67%|
| Calibration CIFAR |18|0.63259|1.09557|50.00%|
| Heldout evaluation MNIST |18|0.42850|1.16523|61.11%|

T là full-worker time, M là RSS, không VRAM. Log-linear ridge fit chỉ đọc fit partition; empirical residual p90 tính từ calibration, evaluation chỉ diagnostics. Không gọi các error này là safety/coverage guarantee; clustered configs/seeds và unseenGPU vẫn chưa có chứng minh. Card ghi `coverage_guarantee=False`, `unseen_gpu_validated=False` và không dùng quality label làm target predictor.

CPU domain finite-space findings ở E4: CIFAR selectedwidth64/LR0.001 val0.274444±SD0.019532; MNISTwidth64/LR0.001 val0.725556±SD0.020092. Khác với Fashionwidth32; không suy general quality/convergence từ dataset subsets/horizon4 hoặc gộp scores giữa tasks.

Deployment thật: `runs/deploy_v2_verified/20261013_114409_157425/TRAINING.json`, selected MNIST config retrain CPU1epoch, optimizer/loss/config/checkpoint passed, val0.453333, RSS301.02MiB, worker5.05s. Status `execution_verified_optimality_not_reestablished` vì đổi horizon; không giữ certificateE4 choE1.

GPU no-op resume sau full-space không thêm worker: audit vẫn24attempts, original+additional costs không bị xóa. Additional30s được ghi rõ như một grant riêng; dùng báo cáo latest để xem cumulative clocks.

Đã dùng card thật trong online search mới: `runs/prior_v2_mnist/20261013_114618_639936`, B90s,6C/3seeds/E4. Audit8attempts (2probes+6full workers); chỉ2configs đủ3seeds, chọn width16/LR0.003 val0.663333±SD0.032146 và giữ certificate **best observed**, không toàn-grid optimum. Các config chưa thử có `learned point prior` với `upper_bound_guaranteed=False`; không giả việc có predictor đã tìm được reference-bestwidth64. Đây là execution check + same heldout-domain adaptation theo rule đã khóa, không dữ liệu để refit card hoặc superiority test.

Nguồn19files `yolo_tuner` trong protocol job ViECAP cũ đã đối chiếu SHA, không mismatch. Console entry `train-config --help` hoạt động với bản0.7.1. Cuối lượt không còn ML workers của agent;162tests cuối cùng qua CUDA, CPU skip1CUDA-specific, compile/dependency checks qua.

## Phạm vi chưa hoàn thành

Unseen physical GPU, calibrated safety/coverage intervals, quality surrogate/cost-aware EI/NEHVI, matched full multi-rung baselines, larger convergence studies, independent final domains/statistical power và clean install trên máy khác chưa được xác lập. Installer/CLI chạy được không thay các minh chứng này. Không tuyên bố optimum mọi máy/dataset.
