# Nghiệm thu nâng cấp 0.9.1 — 2026-10-15

## Phạm vi và thay đổi

Thực hiện [kế hoạch](UPGRADE_091_PLAN.md), phương pháp ở [METHOD_091.md](METHOD_091.md). Không sửa dataset, source snapshots, checkpoint hoặc report lịch sử. Legacy `yolo_tuner` giữ nguyên. Đã cài editable 0.9.1 vào hai môi trường bằng uv `--no-deps --no-build-isolation`, không nâng dependency ML.

Đã sửa: OOD predictor fallback, lý do OOD theo ứng viên, SH quota/finalizer reserve, thứ tự model theo descriptor, preset/tie Pareto-safe, kiểm chứng requested/observed loader/microbatch/shape/accumulation, metadata cache có full hash boundaries, SHA streaming, bảo vệ quota probe khi reconcile, module entrypoint/launcher và benchmark-report analysis-only.

Không triển khai: GP/qNEHVI, worker pool, exact checkpoint continuation, benchmark tự phục hồi hard crash thiếu worker cost, tabular adapter, chứng nhận unseenGPU/global optimum. Auto workers là policy bảo thủ, không là autotuner workers tối ưu đã đo.

## Kiểm thử

- Sau hiệu chỉnh controller: **283 tests qua CUDA runtime; CPU 282 qua + 1 CUDA-only skip**. Có hồi quy checkpoint bulk-cost calibration, probe charge độc lập dynamic reserve, reconcile quota và Windows-spawn đọc ảnh thật. Unit fixtures không được tính là thực nghiệm ML.
- Legacy: so SHA256 với protocol `runs/verified_detection_071/20261013_111148_405825/protocol.json`, **19/19 file `yolo_tuner` không đổi**.
- Wheel đầu: import thật từ target ngoài source checkout, version/help/CPU doctor/actual plan-search thành công; từ chối đúng benchmark hard crash thiếu cost, không bịa số liệu. Đây là snapshot cũ hơn source cuối, xem `runs/upgrade_091_packaging/WHEEL_VERIFICATION.md`.
- **Release wheel cuối:** `runs/upgrade_091_packaging/wheels_release/yolo_budget_tuner-0.9.1-py3-none-any.whl`, SHA256 `418c5cd6e74bf9f15a33db8a86ab5b8f7c8b2cb92d3e208a2e24c3371787e62b`. Cài vào target mới, chạy isolated import từ ngoài repository; cả 57 packaged Python source hashes trùng workspace cuối, module version 0.9.1. Không cần root `train_config.py`. Xem `runs/upgrade_091_packaging/WHEEL_RELEASE_VERIFICATION.md`.
- Metadata dependency cuối: `torch>=2.6`, `torchvision>=0.21`, phù hợp nhánh setup đã đo. Không còn quảng cáo hỗ trợ Torch 2.3 thiếu typed-device AMP API mà worker dùng. Hai môi trường editable đã refresh metadata sau khi mọi ML job kết thúc; không cài/nâng ML libraries.
- Wheel checks không thay thế kiểm chứng clean OS/GPU hoặc subprocess khi tùy biến `--target`; training thật bên dưới chạy qua hai môi trường dự án.

### Kiểm toán cuối sau khi kết thúc mọi ML job

Chạy lại toàn bộ tests: CUDA runtime **283/283 pass** (11.802s); CPU **282 pass, 1 CUDA-only skip** (8.361s). Không còn Python training worker chạy nền tại thời điểm nghiệm thu.

Raw audit **17 run roots / 121 attempts**, tất cả qua; không tin report tổng hợp thay bằng chứng gốc:

| Nhóm | Run roots | Attempts đã audit |
|---|---:|---:|
| Pilot benchmark + reference | 4 | 35 |
| Final benchmark + reference | 4 | 49 |
| Cache off/on | 2 | 2 |
| GPU detection | 1 | 9 |
| GPU captioning | 1 | 2 |
| Deployment CPU/detection/caption | 3 | 3 |
| Cold SH fallback | 1 | 12 |
| Prior SH hai rung | 1 | 9 |
| **Tổng** | **17** | **121** |

Cả 58 source hashes của final benchmark và ba deployment execution trùng source workspace cuối. Các run giai đoạn đầu giữ nguyên snapshot riêng, không tự thay hash. Không có `test_history.json` trong các lượt `upgrade_091_*`; test không được mở để lựa chọn hoặc nghiệm thu chất lượng. Các dataset local vốn đã tham gia development từ trước nên không được gọi là blind/unseen domains.

## Training thật mới, không mở test

### CPU cache A/B, Fashion-MNIST local

Hai lượt cùng config/seed17/E2, full train1000/val300, methoddefault1candidate. TorchCPU2.14.1; đây là một cặp kiểm chứng chứ không phải statistical benchmark.

| Cache | Worker wall(s) | Train+val loop(s) | Val accuracy | RSS MiB |
|---|---:|---:|---:|---:|
|0MiB|4.421|1.297|0.523333|288.7|
|64MiB|4.265|1.000|0.523333|305.1|

Loop giảm22.9%, worker chỉ giảm3.5% trong một cặp. Không gán startup/system variation thành speedup đảm bảo. Cacheon có1000trainhits/300valhits, cùng validation metric. Tổng execution lần lượt10.781/10.594s. Trần cache gồm toàn bộ train/valdecodedcopies, không phải toàn bộ RSS.

Raw:

- `runs/upgrade_091_cache_off/20261015_101004_207464`
- `runs/upgrade_091_cache_on/20261015_101028_388696`

### GPU detection và Freeze thật

RTX3050Laptop6GB (UUID ẩn trong bản tài liệu công khai, bản gốc giữ trong artifact cục bộ); Torch2.6.0+cu124/Ultralytics8.4.175. Penn-Fudan local130train/20val/20test; test không mở. Grid3×seeds17/29/43×E2;9/9fulloutcomes thực tế, không là E20 hoặc convergence experiment.

| Model | Freeze | Val mAP50-95 mean | SD | Worker T mean(s) | CUDA reserved max MiB |
|---|---:|---:|---:|---:|---:|
|YOLO11n local pretrained|0|0.451230|0.036825|19.458|182|
|YOLO11n local pretrained|5|0.527083|0.053075|18.0677|180|
|YOLO11s local pretrained|0|0.339467|0.138328|17.1457|458|

Balanced chọn YOLO11n/freeze5 trong **đúng grid/horizon/seeds này**. Không suy ra model tốt nhất nói chung. Freeze actual được xác minh `requires_grad=False` các block yêu cầu; vẫn có2,530,019trainableparameters. Pinning CUDA và workers thực tế được ghi. Primary memory là allocator reserved, không phải totalVRAM.

Execution175.484s/budget300s, workers164.014s, finalverify/audit0.671s;0overrun. Integrity có6651cachehits,1478hashreads ở boundaries, full hash đầu/cuối không bỏ. Audit9attempts qua.

Raw: `runs/upgrade_091_detection/20261015_101052_173416`.

### GPU captioning và cache ảnh dùng chung

ViECAP development local64trainimages/317captions,16valimages/80captions;test không mở. Tinyattentionencoder64px,width16,hidden64,batch8,accum2,FP16;1candidate×2seeds17/29×E2. Hai full outcomes qua verification/audit, không là training production đủ chất lượng.

Cache mỗi seed: train64decode misses/570hits; val16misses/176hits gồm validation generation. Không dùng frozen feature cache cho encoder đang train. Actualshape64×64/microbatch8/accumulation được xác minh.

Execution17.156s/budget180s; meanworker6.711s;CUDAreserved92MiB. Q=(BLEU4+chrF)/200 mean0.141276±SD0.005279. Caption lặp, chỉ2epoch/nhỏ: **không đủ application quality**, không được so như đã nâng BLEU production. Giữ generated-metric/baseline thay vì chỉ loss.

Raw: `runs/upgrade_091_caption/20261015_101413_563069`.

## Benchmark policy và reference độc lập

Pilot đầu hoàn tất: `runs/upgrade_091_benchmark/20261015_101503_243141`, Fashionlocal C4/E2/seeds17/29/43, searchseed17,90s/method, reference12/12fulloutcomes/75.750s. Reserve/SH có6fulloutcomes, random9full+1timeout. Conditional regret trên cùng reference: reserve/SH0.0176188, random0. Không có overrun hoặc false selection. Không chứng minh reserve/SH tốt hơn random.

Pilot chỉ ra estimate quá bảo thủ: random dự phòng31.229s cho finalization nhưng thực tế1.359s, vì trộn throughput ảnh nhỏ với bulkcheckpoint. Cross-config estimate cũng nhân cả non-loop overhead theo width² và bỏ lỡ config rộng dù còn50s. Sau khi giữ nguyên pilot gốc, đã hiệu chỉnh trên controller mới: checkpoint bandwidth đo thật, tách loop/nonloop và fit width exponent từ contrast đã quan sát. Không dùng reference quality để order.

Study cuối đã hoàn tất: `runs/upgrade_091_benchmark_final/20261015_102206_180716`, cùng C/E/S/budget và các tham số còn lại, source mới khóa riêng. Reference chạy riêng đủ **12/12 full outcomes**, tốn 75.766s. Cả ba phương pháp chọn width64/LR0.001, validation accuracy mean 0.622222, SD 0.027756.

| Phương pháp | Full outcomes | Probe | Online(s) | Conditional regret | Overrun(s) |
|---|---:|---:|---:|---:|---:|
| Reserve | 12/12 | 1 | 84.031 | 0 | 0 |
| Random | 12/12 | 0 | 77.359 | 0 | 0 |
| Successive Halving | 12/12 | 0 | 76.766 | 0 | 0 |

Regret tính trên cùng reference với objective đã khóa, không trên chi phí đo của mỗi search. Trong study 90s này, SH ghi `rung_deferred` và dùng full-confirmation fallback; **không có rung SH thực sự để chứng minh early-stopping hiệu quả**. Dự phòng finalization sau hiệu chỉnh 5.0–6.485s, actual 1.234–1.406s. `benchmark-report` đã audit/dựng lại báo cáo, không khởi động training.

Đây là một repeat trên cùng dataset/CPU đã dùng để chẩn đoán pilot: kết quả cho thấy sửa được lãng phí budget ở case local (reserve/SH từ 6 lên 12 full outcomes), **không phải thực nghiệm độc lập chứng minh superiority sau điều chỉnh**, không suy ra thắng Random hoặc transfer sang GPU/dataset mới.

GPU 9/9 và caption 2/2 phía trên dùng worker/backend/data-loader giống source cuối; sau đó controller có thay đổi ở workflow/scheduler/predictors/deployment và thêm `budget.py`. Snapshots/protocol gốc giữ nguyên, không gán lại source hash. Các lượt end-to-end source cuối được ghi riêng bên dưới.

Study0.9.0 trước đó còn thiếu journal cost/reference; giữ nguyên, không đổi thành completed. CLI benchmark-report chỉ dựng lại study đã đủ raw evidence, không làm crash cũ trở thành thành công.

## Kiểm chứng nhánh SH và rủi ro proxy

Lượt cold-start `runs/upgrade_091_sh_rungs/20261015_103044_125452`: budget 180s, probe fraction 0.5. Forecast ban đầu quá bảo thủ để nhận cả rung và giữ finalizer worst-case, nên ghi `rung_deferred`, rồi đo full đủ 12 outcomes trong **106.578s**. Chọn width64/LR0.001. Không gán lượt này thành kiểm chứng hai rung.

Để thực thi nhánh hai rung thật, đã tạo predictor card từ 12 full outcomes của reference local nói trên: `runs/predictor_091_cpu_fashion_fit_only.json`, input ở `configs/predictor_091_engineering_partitions.json`. **Chỉ có fit; calibration và evaluation domains đều bằng 0.** Card tự ghi `empirical_residual_diagnostic_not_validated_safety_interval`, không là safety bound hoặc bằng chứng transfer. Việc dùng dữ liệu đã biết khiến lượt sau chỉ là branch engineering test, không được nhập vào bảng benchmark độc lập.

Lượt `runs/upgrade_091_sh_prior_rungs/20261015_103358_406141` thực sự chạy:

- Rung E1: 4 probe, giữ index [0,1].
- Rung E2: 2 probe, giữ index [1].
- Full confirmation: đủ 3 seeds của index 1, chọn width16/LR0.003; tổng **58.985s**, không overrun.

Validation accuracy mean 0.588889, thấp hơn width64 full-reference 0.622222. **Proxy sớm đã loại cấu hình chất lượng tốt nhất sau full training.** Kết quả chỉ mang nhãn `fully_trained_best_observed`, coverage 1/4 cấu hình full; không được nâng lên optimal trong grid. Đây là minh chứng âm quan trọng: giảm chi phí không bảo đảm bảo toàn ranking/chất lượng cuối, cần đo nhiều domain/seed và báo proxy error chứ không chỉ báo một case thắng.

## End-to-end deployment với source cuối

Ba lượt `train-v2` E2/seed17 chạy lại cấu hình đã chọn, giữ loader policy; đều qua actual-config verification, full end-boundary verification và raw audit, `execution_verified=true`:

| Nhánh | Thư mục có TRAINING.json | Worker(s) | Validation metric |
|---|---|---:|---|
| CPU classification | `runs/upgrade_091_deploy_cpu/20261015_102832_812315` | 6.4 | accuracy 0.523333 |
| GPU detection | `runs/upgrade_091_deploy_gpu/20261015_102853_561075` | 22.5 | mAP50-95 0.54556 |
| GPU captioning | `runs/upgrade_091_deploy_caption/20261015_103008_496564` | 9.6 | Q 0.145009 |

CPU cache-off giữ đúng `workers=0/cache_mb=0/pin_memory=off`; GPU detection giữ freeze5 và kiểm chứng thật; GPU caption dùng pinned transfer/cache như đã khóa. Source scope thay đổi được khai báo rõ; **`optimality_not_reestablished`**. Đây là kiểm chứng chạy lại config, không phải một search mới chứng nhận tối ưu. Cả ba không đánh giá test.

## Giới hạn

Một GPU vật lý/một CPU host không kiểm chứng nhiều GPU. Fixedhorizon ngắn chỉ nghiệm thu engineering, không kết luận chất lượng cuối/convergence. SD không CI. Metadata cache không chống giả mạo. Finalization reserve là heuristic/softdeadline. Worker startup vẫn có chi phí; chưa có workerpool. Resume trial dang dở sẽ restart theo policy hiện tại. Không tự đổi batch/precision rồi giữ chứng nhận cũ.
