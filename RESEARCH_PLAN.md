# Kế hoạch nghiên cứu và triển khai — accuracy / memory / training time

Ngày lập: 13/10/2026. Phạm vi: phát triển phương pháp lựa chọn cấu hình trong một tập hữu hạn, không bảo đảm tối ưu toàn cục. Không thay đổi kết quả lịch sử để tạo ưu thế giả.

## Câu hỏi và giả thuyết

Với dataset D, hardware H, protocol P (split, epoch, metric, seed), grid C và ngân sách tìm kiếm B đã khóa, phương pháp nào chọn được cấu hình khả thi có cân bằng A/M/T gần nghiệm tốt nhất trong C, với chi phí tìm kiếm thấp?

Giả thuyết cần kiểm định, chưa phải kết luận: không loại ứng viên theo quality sớm giúp tránh bỏ cấu hình học chậm; phân tầng LR và profile chi phí cải thiện phân bổ ngân sách. Full-confirmation nhiều seed giảm lựa chọn thiếu bằng chứng. Cần đối chứng và ablation; không tự gọi ghép các thành phần này là thuật toán mới có regret bound.

## Các bước, đầu ra và điều kiện nghiệm thu

### P1 — Protocol khoa học [đã triển khai]

- Định nghĩa A: detection mAP50-95 validation checkpoint tốt nhất trong E epoch; caption/classification giữ metric riêng, không gộp chất lượng giữa task.
- T: wall-clock worker, gồm startup/train/validation/checkpoint. M: peak reserved allocator CUDA hoặc RSS CPU; không gọi RSS là VRAM, không gọi allocator là toàn bộ VRAM máy.
- Khóa grid, horizon, split, seeds, objective, anchors, weights, giới hạn và source/runtime trước search. Không dùng test để chọn config/checkpoint/weights.
- Feasible: đủ full-data/full-horizon/mọi seed, actual hyperparameters khớp yêu cầu, không OOM/timeout, đạt constraints. Chưa thử hoặc lỗi là unresolved.
- U = wA*mean(A) - wT*mean(T)/Tref - wM*max(M)/Mref. Công bố weights trước; Pareto và phân tích độ nhạy, không dùng min/max oracle để đổi objective sau.
- Chứng minh giới hạn: với weights dương, nghiệm tối đa U không bị trội trên cùng 3 đại lượng; đo đủ C xác định argmax thực nghiệm. Không suy ra tối ưu mọi máy/seed/config.
- Đầu ra: RESEARCH_PROTOCOL.md, machine-readable protocol và pseudocode.

### P2 — Detection full-reference adapter [đã triển khai; smoke thật đã qua]

- Module riêng `research_pipeline` để không sửa package `yolo_tuner` đang bị lượt ViECAP 20epoch khóa SHA.
- Hỗ trợ YOLO detection local (YOLO/COCO/VOC/LabelMe qua adapter hiện tại), model kiến trúc YAML hoặc weights local; kiểm checksum model, cấm tải weights âm thầm trong trial.
- Grid thật gồm model/imgsz/batch/precision/gradient accumulation/optimizer/LR/weight decay; khóa seed trước khi khởi tạo model. Ghi actual AMP, batch, accumulation và weight-decay optimizer.
- Tắt early stopping, giữ scheduler horizon E giữa proxy/full; không đổi batch khi OOM để giả vờ cùng cấu hình.
- Checkpoint best/last, learning curve, mAP50 và mAP50-95; progress/request/result/log, timeout cleanup.
- Nghiệm thu: default/random/Successive Halving/adaptive full reference chạy được trên detection; OOM và AMP fallback không được gắn Passed.

### P3 — Giải thuật, ablation và công bằng [đã triển khai và đo7methods]

- Giữ policy adaptive_verified hiện tại như heuristic cần kiểm định, không phát minh ưu thế chưa đo.
- Đối chứng Default, Random full, Random cùng proxy fidelity, Successive Halving. Cùng cap B, báo actual cost; full oracle độc lập có chi phí riêng.
- Ablation: bỏ profile (full train theo lịch phân tầng), bỏ phân tầng (profile rồi full theo thứ tự ngẫu nhiên), giữ policy đầy đủ. Phân tích trực tiếp early-quality pruning bằng SH; không thay đổi rule sau khi biết oracle.
- Khóa decisions trước exhaustive reference. Selected production không có oracle miễn phí; chỉ đầy đủ grid mới chứng nhận optimum trong grid.
- Nghiệm thu: regret/feasibility/HV và sai số prediction tính từ raw full outcomes; thiếu oracle không gán regret0; chi phí thất bại không biến mất.

### P4 — Báo cáo có cơ sở chọn cấu hình [pilot/audit/sensitivity đã hoàn tất]

- Danh sách tất cả cấu hình: hoàn tất/không khả thi/chưa xác định, Q/T/M, seed coverage, Pareto, U, margin với runner-up.
- Recommendation: config đầy đủ, predicted vs measured nếu thật sự có predictor/profile, số đo validation riêng với test, OOM, run integrity, full-training verification, optimality scope.
- Dự báo không có calibration ghi heuristic/no interval; không điền 5.4GB, X giờ, mAP0.XXX như số đo thật.
- Phân tích độ nhạy weights predeclared scenarios; không đổi lựa chọn chính để làm đẹp kết quả. Độ lệch chuẩn seed không gọi confidence interval.
- Nghiệm thu: báo cáo JSON + Markdown và CLI có thể tái tạo từ protocol/request/result.

### P5 — Kiểm thử và tích hợp không phá job đang chạy [đã qua126tests ở2runtime]

- Unit tests: math/Pareto, seed coverage, oracle boundary, ablation cost, config actual mismatch, dữ liệu/test leakage, raw integrity, checkpoint identity, timeout/process lifecycle.
- Root `train_config.py` dispatch module mới cho detection; caption và các lệnh cũ giữ package0.5 bất biến. Thêm source guard của cả module mới và helper cũ cho nghiên cứu mới.
- Chạy toàn bộ tests CPU/CUDA, compileall, dependency check; báo riêng mock/smoke/benchmark thật.
- Không thay dependencies/source của job ViECAP hiện tại; không dừng job của người dùng.

### P6 — Thực nghiệm detection thực tế [pilot/production/test/deployment đã hoàn tất]

- Dataset A là ví dụ, không có dataset cụ thể do người dùng chỉ định; chuẩn bị Penn-Fudan pedestrian detection nhãn thật từ nguồn chính thức, checksum archive, split theo filename groups (không có true scene IDs), giữ test ngoài lựa chọn. Không gán box toàn ảnh từ class label để tạo detection giả.
- Pilot nhỏ chỉ xác nhận pipeline, không chứng minh chất lượng/SOTA. Benchmark khóa grid nhỏ, nhiều seeds, đủ full epoch khai báo, nhiều methods; báo tính chất small-domain nếu grid/horizon nhỏ.
- Không chạy benchmark tài nguyên đồng thời job ViECAP đang dùng GPU/CPU. Nếu chưa có cửa sổ máy nhàn, vẫn hoàn tất code/tests/data/protocol, báo benchmark chưa hoàn tất; không tạo số đo thay thế.
- Khi có máy nhàn: chạy full-reference và audit; so phương pháp trên cùng B. Mở rộng nhiều B/horizon/dataset và GPU vật lý khác là thực nghiệm tiếp theo, không tự tuyên bố đã hoàn thành khi chỉ có RTX3050.

### P7 — Tổng kết minh chứng và bàn giao [đã hoàn tất phạm vi pilot]

- Ghi rõ kết quả đạt/không đạt từng giả thuyết, effect/cost, giới hạn và phần còn cần tài nguyên/dataset.
- Hướng dẫn lệnh detection, custom grid, show/audit/resume/analysis/test evaluation. Snapshot không sửa lịch sử.
- Nghiệm thu: người dùng chạy từ dataset detection thật nhận được grid khả thi/Pareto/config cân bằng/bằng chứng; không chỉ một bảng tham số tùy ý.

## Trạng thái môi trường ban đầu

Có job của người dùng: `runs/verified/20261013_072742_478191`, ViECAP20epoch, GPU RTX3050. Bảo toàn package mà job đã khóa; không benchmark đồng thời để chứng nhận thời gian/VRAM. GPU vật lý khác chưa có để thử. Dataset A/B detection chưa được người dùng chỉ định.

## Lịch ghi tiến độ

- 13/10/2026: đã khảo sát mã, phát hiện job ViECAP đang chạy; lập kế hoạch và chọn kiến trúc module nâng cấp độc lập.
- 13/10/2026: P1–P5 đã triển khai,118tests qua CPU/CUDA; giữ nguyên SHA package của job cũ. Penn-Fudan130/20/20 đã chuẩn bị, smoke2fullworkers×2epoch hoàn tất và audit. Sau khi job ViECAP dừng bằng KeyboardInterrupt (không do agent dừng), khởi động independent detection pilot3config×3seed×4epoch,7methods,60s/search/seed.
- 13/10/2026: pilot hoàn tất9/9full-reference, audit71originalrequests, c01balanced optimum, Random full regret0/feasible3/3; adaptive full policy chỉ1/3feasible dưới60s. Không tuyên bố superiority. Đã sửa audit JSON-null key và guards triển khai0.6.1,125tests qua2runtime; chạy thêm production300s/seed hoàn toàn online.
- 13/10/2026: production9/9full-online+9proxy đã xong/audit18requests, offline0, c01vẫn được chọn. Test riêng seed17mAP50-95=0.622469; không chọn lại config. Deployment CPU1epoch thực sự hoàn tất và hạ certificate; no-opresume không trùngworker.126tests qua2runtime; dependency/compile/source-preservation checks qua. Tài liệu DETECTION_RESEARCH.md/DETECTION_VALIDATION.md và registry6runs/4datasets/1GPU đã bàn giao.

## Phạm vi hoàn thành và nghiên cứu tiếp theo

P1–P7 đã hoàn thành cho pilot triển khai/thực nghiệm hiện có. Điều đó không đồng nghĩa đã kiểm định đầy đủ một thuật toán mới có superiority. Dữ liệu cho thấy profile guard under cap60s thường không feasible; giữ finding này, không sửa lịch sử để thắng.

Các bước nghiên cứu rộng hơn chưa thực hiện: nhiềuGPU vật lý/host độc lập, nhiều budget/horizon/grid và dataset nguồn mới, matched scalarized-U SH, nhiều seeds/splits và calibration predictor với test coverage độc lập. Đây là chương trình thực nghiệm tiếp theo cần tài nguyên/thiết kế riêng; không gắn "đã xong" cho phần chưa đo. Chỉoptimum thực nghiệm trongC/D/H/P/seed/objective đã khóa được chứng nhận.
