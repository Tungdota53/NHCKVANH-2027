# Pipeline nghiên cứu 0.7.1: chạy thật, ngân sách chung, bằng chứng kiểm tra được

Đây là implementation của phần lõi trong [RESEARCH_UPGRADE_PLAN.md](RESEARCH_UPGRADE_PLAN.md). Không đồng nghĩa đã hoàn tất kiểm định nhiều GPU, superiority thống kê hoặc predictor có safety bounds. Kết quả cũ giữ nguyên; `yolo_tuner/*.py` và extension 0.6 không bị viết lại.

Trạng thái/thực nghiệm: [UPGRADE_PROGRESS.md](UPGRADE_PROGRESS.md), [UPGRADE_VALIDATION.md](UPGRADE_VALIDATION.md), [PROTOCOL_V2.md](PROTOCOL_V2.md). Mặc định `--space core`; `--space compact` là grid nhỏ. 0.7.1 hỗ trợ JSON manifest `{task,names,splits}` với image/targets/group_id và external-path native YAML/COCO; kiểm positive boxes và declared-group leakage, không fallback supervised task sang reconstruction.

## Chạy trên máy hiện tại

```powershell
.venv\Scripts\python.exe train_config.py doctor --device auto
.venv\Scripts\python.exe train_config.py inspect --data "D:\scr code\NCHK 2027\ViECAP-Image-Captioning-Vietnamese-master"
.venv\Scripts\python.exe train_config.py optimize --data "D:\scr code\NCHK 2027\ViECAP-Image-Captioning-Vietnamese-master" --device auto --full-epochs 20 --budget-seconds 3600 --seeds 17 29 43
```

Lệnh optimize huấn luyện thật và chỉ chọn từ cấu hình đã full train trên dữ liệu khai báo, đủ horizon và mọi seed. 3600 giây là ngân sách tìm kiếm chung, không thời gian train dự báo, không hứa đủ để xác minh một cấu hình ViECAP. Nếu không đủ hoặc không đạt constraints, chương trình báo không có lựa chọn được xác minh. Bắt đầu từ `inspect`/`plan-search` nếu chưa muốn chạy training.

V2 caption hiện dùng tiny image encoder + spatial-attention GRU, metric câu sinh BLEU-4/chrF. Frozen-pretrained ResNet18 backend của nhánh 0.5 vẫn có qua lệnh cũ; không âm thầm chuyển encoder hoặc tải weights. V2 classification dùng CNN/AdamW với decay 0.01 cố định; tham số/backend chưa hỗ trợ bị reject. Detection dùng YOLO local weights hoặc kiến trúc YAML cài sẵn; ảnh/boxes thật qua YOLO/COCO/VOC/LabelMe adapter. Không hỗ trợ dataset không nhãn bằng cách tự dựng nhãn.

## Detection và cấu hình có kiểm soát

```powershell
.venv\Scripts\python.exe train_config.py optimize --data runs\datasets\penn_fudan_research\data.yaml --task detection --grid configs\detection_v2_controlled.json --device 0 --full-epochs 8 --seeds 17 29 43 --budget-seconds 600
```

Grid controlled có baseline và các đối chứng chỉ đổi image, batch/accum giữ effective batch, LR, model hoặc weight decay. `--space core` tạo khoảng 12 đối chứng hợp lệ theo task; `--grid` nhận tối đa 512 ứng viên hữu hạn. Các trường bắt buộc phù hợp backend; không auto-reduce batch, đổi optimizer hoặc tắt AMP khi lỗi. Model snapshots được hash; mỗi candidate có content ID ổn định khi model được copy sang đường dẫn khác.

## Chốt protocol trước khi chạy

```powershell
.venv\Scripts\python.exe train_config.py plan-search --data DATASET --device auto --space core --full-epochs 20 --budget-seconds 1200
.venv\Scripts\python.exe train_config.py optimize --protocol "ABSOLUTE_RUN_PATH\protocol.json"
```

`--protocol` chỉ chạy plan chưa bắt đầu, không nhận parameter/device/budget override. Preparation của plan riêng được báo riêng; khi chạy optimize trực tiếp từ `--data`, discovery/hardware/hash/export đều nằm trong B online. Muốn đổi điều kiện thì tạo protocol mới; muốn thêm ngân sách cho lượt đã chạy thì resume rõ ràng.

## Xem, audit và resume

```powershell
.venv\Scripts\python.exe train_config.py status --run "RUN_PATH"
.venv\Scripts\python.exe train_config.py audit --run "RUN_PATH"
.venv\Scripts\python.exe train_config.py resume --run "RUN_PATH" --additional-budget-seconds 300
```

Status kiểm PID/create-time, không dùng phase lưu để khẳng định process sống. Audit kiểm request/raw/parent outcome/checkpoint, immutable committed state, source snapshots và hash-chained event ledger; tính lại recommendation từ full outcomes. Đây là integrity/reproducibility, không chống giả mạo có chủ đích.

Resume audit trước khi skip worker, không chạy trùng job đang sống. Chỉ missing seeds/configs được thực thi theo policy; failure/retry cost giữ nguyên, downtime không bị gọi là train cost. Nếu parent chết sau khi outcome đã journal nhưng state chưa lưu, state được reconcile từ parent evidence. Nếu hard crash khiến elapsed cost không biết, chương trình dừng để điều tra, không tự gán cost 0 rồi replay. Resume workflow không phải continuation từ checkpoint; mỗi trial mới train từ initialization theo protocol.

## Thuật toán và hàm mục tiêu

```text
U = wA*mean(validation_quality) - wT*mean(full_worker_time)/Tref - wM*max(memory)/Mref
```

Weights/anchors/constraints khóa trước. CUDA M là reserved allocator MiB; CPU M là sum process-tree RSS MiB, shared pages có thể bị tính lặp; không gọi RSS là VRAM hoặc allocator là toàn GPU memory. NVML device usage/process VRAM/thermal/power là diagnostics có availability rõ. Per-phase peaks là sampling, có thể bỏ sót peak ngắn; toàn worker vẫn có CUDA allocator peak counter.

Policy `reserve` full-confirm một ứng viên ít rủi ro trước, dùng budget chung giữa seeds, giới hạn probe và chỉ khám phá khi còn reserve đủ cho finalist theo forecast. Forecast/margin là heuristic, không upper bound đã calibration. Không quality-prune bằng loss sớm. Nghiệm cuối chỉ từ full registry, không proxy.

Các policy phục vụ đối chứng: `random`, `default`, `profile_all`, `random_proxy`, `quality_sh`, `utility_sh`, `no_reserve`. SH hiện là một rung + promotion nửa cohort + restart full finalizer, không toàn bộ Hyperband/BOHB. Utility-SH dùng quality proxy/full cost forecast đã khóa; không đọc oracle. Mọi policy, kể cả default, trả cost full confirmations. V2 chưa triển khai quality-surrogate acquisition/NEHVI; không gọi reserve là Bayesian optimizer.

## Benchmark nghiên cứu thật

```powershell
.venv-cpu\Scripts\python.exe train_config.py benchmark --data runs\datasets\fashion_research --task classification --device cpu --space core --full-epochs 4 --seeds 17 29 43 --methods reserve random profile_all --search-seeds 17 29 43 --budget-seconds 90 --reference-budget-seconds 900 --output runs\benchmark_v2_fashion
```

Một subprocess lạnh mỗi search, không training đồng thời. Cùng C/D/H/P/objective/memory cap, một shared B mỗi search; method ordering đổi theo repeat. Decisions được khóa/audit trước khi launch independent exhaustive reference. Reference cost riêng; regret lấy cả hai U từ cùng reference. Reference thiếu thì regret null; selection thất bại không được gọi regret 0. Repeats trên một domain không là superiority trên mọi dataset/GPU. Dataset lịch sử chỉ là development, không blind heldout mới.

## Predictor và calibration

```json
{"fit": ["RUN_FIT"], "calibration": ["RUN_CALIBRATION"], "evaluation": ["RUN_HELDOUT"]}
```

```powershell
.venv\Scripts\python.exe train_config.py calibrate --manifest partitions.json --output predictor.json
.venv\Scripts\python.exe train_config.py optimize --data DATASET --predictor predictor.json
```

Card fit từ audited full outcomes, tối thiểu 12 rows. Cấm dataset-content/hardware domain xuất hiện ở nhiều partitions; cấm gộp task hoặc RSS/VRAM. Fit log-linear ridge T/M và residual diagnostics; không fit quality hoặc gán timeout/OOM thành full labels. Hardware/runtime/source lạ hoặc features ngoài range ⇒ không dùng point prior, fallback đo thật.

Residual p90 là diagnostic, không conformal safety interval. Card luôn ghi coverage guarantee=False/unseen GPU validated=False. Chưa có independent multi-domain coverage experiment thì không được xuất “95% an toàn”. Grouped calibration/quality surrogate/nhiều GPU là các bước nghiên cứu tiếp, không phần đã hoàn thành.

## Train và test sau selection

```powershell
.venv\Scripts\python.exe train_config.py train-v2 --selection "RUN_PATH\selected.json" --device auto --output runs\deploy_v2
.venv\Scripts\python.exe train_config.py evaluate-v2 --run "RUN_PATH"
```

Train-v2 thực thi lại config, không search; report ghi scope changes và không tái cấp optimality certificate. Test chỉ sau completed/locked full selection; dùng seed đầu đã khai báo, checkpoint hash đúng. Exposure ghi trước worker, kể cả failure; rerun cùng study cần `--allow-retest`. Test inference FP32, không đổi selection theo test. Historical global test blindness không được tự chứng nhận chỉ vì tạo study mới.

## Cài trên máy Windows khác

```powershell
.\setup-v2.ps1 -Backend cpu
# Hoặc CUDA 12.4 nếu máy/driver tương thích:
.\setup-v2.ps1 -Backend cu124
```

Installer tạo `.venv-v2-cpu`/`.venv-v2-cu124` mới và từ chối overwrite môi trường có sẵn; không đổi `.venv` của job đang chạy. Có network/download/cài dependencies, không phải offline mode. Pair Torch 2.6.0/torchvision 0.21.0 và wheel CPU/cu124 theo [hướng dẫn phiên bản chính thức PyTorch](https://pytorch.org/get-started/previous-versions/#v260); Ultralytics pin 8.4.175. Đây là runtime recipe, không xác nhận mọi GPU/Python/driver tương thích. Script chưa được kiểm thử bằng clean install trên máy vật lý khác; doctor kiểm backend thực tế sau cài.

Linux/macOS dùng venv/pip theo nền tảng và entry `train-config`; MPS/ROCm chưa có máy để kiểm chứng ở dự án này. Trên máy thiếu tài nguyên/backend/nhãn/model, nhận diagnostic hoặc no verified selection, không một cấu hình tối ưu giả.

## Artifacts

- `protocol.json`, `dataset.json`, `subset.json`, model/source snapshots và exported labels.
- `events.jsonl`, immutable `commits/*.json`, `state.json`, từng `trials/attempt_*/request.json/result.json/parent_result.json/worker.log/process.json`.
- `REPORT.md`, `report.json`, `decision.locked.json`; chỉ có `selected.json` khi có full-confirmed feasible choice.
- `BENCHMARK.md/json` và independent reference; `evaluation/` riêng, không lẫn với validation search.

Lệnh lịch sử `research`, `verify-config`, `show-training`, `train`, `evaluate-detection` vẫn route sang 0.5/0.6. Root entry mới khiến source identity của protocol cũ thay đổi: audit lịch sử vẫn dùng snapshots; live resume/evaluation cần đúng frozen source/runtime hoặc protocol mới, không bypass source guard.
