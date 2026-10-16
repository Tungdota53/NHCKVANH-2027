# Tiến độ kiểm chứng nhánh nghiên cứu detection0.6

## Đã hoàn tất

- Kế hoạch7bước tại RESEARCH_PLAN.md; công thức/U/Pareto/proof giới hạn/pseudocode/protocol tại RESEARCH_PROTOCOL.md.
- Adapter YOLO full data/full horizon với actual-config verification, pretrained/architecture snapshot, nguồn/runtime/data hash, isolated worker, checkpoints và raw audit.
- Policy/ablation: default/random/random_proxy/SH/adaptive_verified/adaptive_no_profile/adaptive_no_strata. Sensitivity scenarios được khóa trước, không đổi primary selection.
- CLI chung `python train_config.py`: detection research/verify/show/audit/start/resume/train/evaluate/analysis/transfer; compact tasks chuyển sang runner0.5.
- **126/126 tests** trên cả `.venv` CUDA và `.venv-cpu`; compileall thành công,48packages tương thích ở mỗi runtime. Unit tests có mocked outcomes là kiểm tra logic, không dữ liệu benchmark thật.
- Penn-Fudan nhãn thật:130train/20val/20test,170ảnh/345pedestrians, archive SHA2569095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2; mask-derived boxes, grouped filename blocks, không trùng exact content.

## Smoke thật — không chứng nhận chất lượng

`runs/detection_smoke_06/20261013_075028_178417`:2config×1seed×2epoch, YOLO11n YAML random initialization, FP32, image160/batch4/accum4, LR0.001 và0.0003. Cả2full workers hoàn tất, checkpointbest/last được lưu; Qvalidation mAP50-95 đều0.0. Worker wall30.41/22.34s, peak reserved212MiB. Không có ý nghĩa superiority/convergence từ2epoch/single seed hoặc mAP0.

Đánh dấu `contended_smoke_not_certified` bằng override smoke có chủ ý (máy lúc chạy không có ML job khác được preflight nhận diện); không xuất selected.json đã certified. Audit khớp2originalrequests/results/snapshot/checkpoint bytes. Các số đo chỉ chứng minh luồng YOLO/worker/checkpoint/audit thực sự chạy, không benchmark chất lượng.

## Pilot nghiên cứu thật — đã hoàn tất và audit

`runs/research_detection_06/20261013_075617_022125`:3config×3seed×4epoch; mỗi search policy có60s/seed. Trainsubset64ảnh, val20; full reference130train/20val. Test20ảnh không dùng lựa chọn. Source/protocol/models/weights/scenarios khóa trước, decisions sẽ khóa trước exhaustive reference độc lập.

Grid khóa trước:

| ID | Model / init | Image | Batch | Accum | Optimizer | LR | Weight decay | Precision |
|---|---|---:|---:|---:|---|---:|---:|---|
| c00 | YOLO11n / pretrained COCO | 160 | 4 | 4 | AdamW | 0.001 | 0.0005 | FP16 |
| c01 | YOLO11n / pretrained COCO | 224 | 8 | 2 | AdamW | 0.001 | 0.0005 | FP16 |
| c02 | YOLO11s / pretrained COCO | 160 | 4 | 4 | AdamW | 0.0003 | 0.0005 | FP16 |

Rungs1/2, scheduler horizon4, early stop tắt, warmup0 giữ exact accumulation; nbs16 và explicit optimizer giữ LR/decay không bị auto đổi. Các configs khác nhiều yếu tố cùng lúc, không quy ranking thành causal effect của một hyperparameter riêng. Đây chỉ là pilot3config/4epoch, không toàn bộ16expanded configs/20epoch hoặc thuật toán tối ưu toàn cục.

Đã hoàn tất **9/9 independent full-reference**;21 search decisions khóa trước oracle. Audit khớp **71 unique worker request/result** gồm proxy/online/full-reference và checkpoint/source snapshots. Selected reference **c01**, Pareto gồm cả3config. Selected này từ offline reference, không phải một đáp án online miễn phí trong60s.

| ID | Val mAP50-95 mean±SD | Full worker time mean±SD(s) | Peak reserved max(MiB) | Balanced U |
|---|---:|---:|---:|---:|
| c00 | 0.590007±0.020888 | 26.98±1.19 | 202 | 0.313173 |
| c01 | 0.657613±0.036122 | 22.58±1.43 | 318 | 0.338576 |
| c02 | 0.699323±0.013085 | 29.02±1.95 | 556 | 0.326059 |

Objective khóa0.6/0.25/0.15; Tref600s/Mref1024MiB; marginc01 so runner-upc02 **0.012517**. YOLO11s có Qcao nhất nhưng không phải cân bằng nhất với weights/anchors này. Anchors/trọng số là ưu tiên protocol, không hằng số khách quan hoặc fraction VRAM6GB mặc định; không đổi chúng sau để làm đẹp kết luận.

| Method | Feasible selections | Regret mean±SD (chỉ choices feasible) | Search mean±SD(s) |
|---|---:|---:|---:|
| Default c00 khóa trước | 3/3 | 0.025403±0 | 0 |
| Random full | 3/3 | 0±0 | 60.59±0.09 |
| Random proxy | 3/3 | 0.008345±0.007227 | 32.92±1.37 |
| Successive Halving | 3/3 | 0.008345±0.007227 | 52.05±3.93 |
| Adaptive verified | **1/3** | 0 trên1choice;2regret null | 53.07±11.79 |
| Adaptive no profile | 3/3 | 0.004172±0.007227 | 60.53±0.12 |
| Adaptive no strata | **1/3** | 0 trên1choice;2regret null | 53.63±11.51 |

Random full chọnc01 cả3lượt. Không gọi adaptive có regret0 là ngang Random: nó thất bại lựa chọn2/3lượt. Profile cost và forecast margin tiêu phần lớn cap60s; ablation bỏ profile có tỷ lệ feasible cao hơn trong pilot này. Đây là finding tại3configs/4epoch/60s, không statistical superiority trên mọi budget/dataset. Default là c00 khai báo, không phải toàn bộ Ultralytics factory defaults. SH baseline promotion theoQ, final choice theoU; chưa có scalarized-U SH để tách hoàn toàn scheduling và promotion criterion.

Proxy quality MAE của adaptive **0.249718**, time MAPE21.17%, memory relative error25.56%. Riêng recommendationc01: profile predicted memory260MiB vs measured318MiB, predicted time18.23s vs measured22.58s. Không gọi đây là calibrated predictor hoặc confidence interval. Mean prediction gộp các profile đã hoàn tất của các phương pháp; đó là chẩn đoán sau reference, không thông tin oracle sẵn có cho một search policy.

Sensitivity đã khóa trước: balanced/time-first chọnc01, accuracy-first chọnc02, memory-first chọnc00. Primary selection không thay đổi. Đây là minh chứng trực tiếp cho trade-off, không có một optimum bất biến với mọi ưu tiên.

Elapsed tracked active1178.375s (~19.64phút), offline full-reference worker235.734s; chi phí failed/timeout online có trong decisions. Common setup counter của adapter bắt đầu sau discovery/runtime probe; nó không là toàn bộ CLI cold-start latency. Không giấu full-reference trong cap60s hay gọi20phút study là thời gian training một model.

## Production verification — đã hoàn tất, không có oracle offline

`runs/verified_detection_061/20261013_082149_011738`: cùng dataset/grid/horizon/seeds/objective, adaptive_verified,300s/seed, model/metric/search rule không thay để làm đẹp kết quả pilot. Mọi full confirmations tính online. Đây là kiểm tra cùng miền sau pilot, không validation chuyển dataset mới.

Trước lượt này sửa lỗi audit JSON-null dictionary key, roothelp/console discovery, và thêm guards optimizer-state/finite-training-loss/physical-GPU identity/host-label. Extension0.6.1 có source snapshot khác pilot0.6.0; không sửa snapshot/checksum hoặc số đo pilot. Guard thêm finite-loss capture có overhead, nên thời gian hai lượt không quy hoàn toàn thành hiệu ứng budget. Cài lại editable package không đổi dependencies; base package0.5 vẫn giữ nguyên nguồn.

Đã hoàn tất **9/9 full-online confirmations**, cộng9proxy; audit khớp18originalrequests/results. Mọi trial thành công có optimizer state đã tạo, loss training hữu hạn và actual batch/imgsz/precision/accumulation/optimizer/LR/decay khớp. Full-space verified, vẫn chọn **c01**; validationQ từng seed khớp pilot tới precision do trainer báo. Không dùng offline reference để sửa online choices.

Elapsed tracked353.594s (~5.89phút), full-online worker244.297s, offline0. SelectedU0.338383; margin runner-up0.012606. Tmean23.05s **cho một full worker4epoch/130train/20val**, không cho toàn study hoặc dataset khác. Memory max318MiB CUDA reserved, không toàn VRAM hệ thống. Predicted time14.64s và predicted memory260MiB là heuristic sai lệch, không predictor đã calibration.

## Kết quả dạng recommendation theo yêu cầu đề tài

| Trường | Giá trị thực nghiệm | Phạm vi |
|---|---|---|
| GPU | RTX3050 Laptop6GB | Một GPU vật lý |
| Dataset | Penn-Fudan130train/20val/20test | Filename-grouped split101 |
| Model | YOLO11n pretrained COCO | Nghiệm trong grid3configs |
| Image / batch / precision | 224 / 8 / FP16 mixed precision | Actual settings đã kiểm tra |
| Gradient accumulation | 2 | Nominal effective batch16; minibatch cuối có thể nhỏ hơn |
| Optimizer / LR / weight decay | AdamW /0.001 /0.0005 | Warmup0; initialLR/decay verified |
| Predicted / measured memory | 260 /318MiB | Profile heuristic / peak reserved max3seed; không total VRAM |
| Full training time | 23.05s mean |4epoch cho130train/20val, workerstartup/train/val/checkpoint |
| Validation mAP50-95 | 0.657613±0.036122 | Mean±SD của3seeds; không confidence interval |
| Validation checkpoint seed17 | 0.617160 | Cùng seed với test dưới đây |
| Test mAP50-95 /mAP50 | 0.622469 /0.876560 |20test images, seed17, đánh giá sau lựa chọn |
| OOM / full verification | Không ở các full runs được chọn / Passed | Observed required seeds, không mọi seed/máy tương lai |
| Optimality | empirical_optimal_in_declared_space | C3/H/P/4epoch/seeds17,29,43/weights đã khóa |

Test artifact: `runs/verified_detection_061/20261013_082149_011738/evaluation/20261013_082929_407169/EVALUATION.md`. Đánh giá independent test sau lựa chọn; không đổi model/weights/objective theo test. Inference test dùng validation API mặc định FP32, training là mixedFP16; không quy mọi khác biệt val/test thành domain effect. Test hiện đã exposed; history ghi trước worker, kể cả failure. Không gọi mean validation3seeds và test1seed là paired3seed estimate.

## Deployment và resume đã kiểm tra

Đã chạy `train --selection` từ lựa chọn GPU sang `.venv-cpu`, đồng thời đổi horizon thành1epoch: hoàn tất1/1, optimizer state/finite loss True, source report tại `runs/deploy_detection_061/20261013_083032_641614/training_report.json`. Status **modified_protocol_optimality_not_verified** vì horizon/precision/device/runtime khác; không giữ certificate cũ. Đây là kiểm tra thực thi/cảnh báo, không so chất lượng CPU1epoch với GPU4epoch hoặc coi RSS660.79MiB là VRAM.

`research-resume` trên production đã đủ9/9 không gọi thêm worker, không replay search;18request audit vẫn giữ nguyên. `resume.json` ghi additional budget3.6s và additional active0s tại độ phân giải đồng hồ hiện tại, không gán downtime thành train cost. Unit test riêng của adapter mới xác nhận chỉ missing full job được chạy, failed cost10s+new2s được giữ thành12s; đây là mock test, không giả vờ có benchmark missing-job thực.

`runs/transfer_061/20261013_083043_555402/TRANSFER.md` tổng hợp6complete runs từ4dataset content/splits, **một** physical CUDA UUID. Có78full rows kiểm chứng ở registry (12+12+18+18+9+9); không gộp Q/memory giữa task/backend. GPU chưa thấy vẫn `not_established`.

## Tính toàn vẹn job của người dùng

Job ViECAP20epoch `runs/verified/20261013_072742_478191` tồn tại khi bắt đầu nâng cấp. Không dừng hoặc sửa `yolo_tuner/*.py`; mọi source SHA trong protocol vẫn khớp. Sau đó process không còn sống, log cuối ghi KeyboardInterrupt; search decisions chưa khóa nên không coi job đã hoàn tất hay tự replay budget. Các artifacts vẫn được giữ nguyên. Benchmark detection bắt đầu sau khi preflight không còn nhận diện ML job khác.

## Giới hạn

Chỉ RTX3050 Laptop6GB vật lý đã có. Preflight không bảo đảm cô lập toàn hệ thống/thermal/power; chưa có GPU vật lý khác. Exact duplicate check/filename grouping không chứng minh không còn near-duplicate scene leakage.4epoch/pretrained subset nhỏ không là chứng nhận triển khai hoặc SOTA. Chưa có bằng chứng superiority thống kê; sensitivity3seed không cung cấp confidence interval hiệu chỉnh.

Chưa đối chiếu toàn bộ pretraining COCO với Penn-Fudan để chứng minh không có overlap; nguồn pretrained phải công bố. Test chỉ đánh giá sau lựa chọn, không chuyển test score thành công cụ chọn lại weights/model.
