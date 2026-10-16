# Phương pháp nghiên cứu đã thực thi trong pipeline 0.7.1

## Đầu vào và mục tiêu

Khóa D (pixels/labels/splits/groups), H (runtime/backend/hardware identity), P (metric/E/seeds/preprocessing/checkpoint rule), C hữu hạn, B online. Không dùng test để chọn config, checkpoint, objective hoặc hyperparameters của policy.

```text
U(c) = wA * mean(A_c) - wT * mean(T_c)/Tref - wM * max(M_c)/Mref
```

Weights không âm được normalize; anchors dương khóa trước, không lấy min/max oracle. Constraints dùng min(A), max(T), max(M) qua required seeds. CUDA M là peak reserved allocator, CPU M là peak process-tree RSS; không gộp hai counter trong một ranking. Default min-quality=0 xác minh workflow, không chứng nhận ứng dụng đạt chất lượng.

Với weights dương, nếu d trội c trên mean(A), mean(T), max(M), ít nhất một trục nghiêm, thì U(d)>U(c). Argmax khả thi vì thế thuộc Pareto set. Điều này không chứng minh search heuristic tìm được argmax; khi có zero weight không suy ra strict Pareto trên trục bị bỏ.

## Shared-budget reserve/gated-profiling

1. Validate labels/groups/actual backend; lock protocol/source/model/data, bắt đầu journal từ workflow entry.
2. Theo thứ tự metadata-diversified đã khóa, ưu tiên ứng viên ít rủi ro theo width/image/batch heuristic; không dựa vào oracle accuracy.
3. Nếu có ngân sách probe đủ startup, probe một ứng viên ban đầu, không quét mọi config. Full time forecast có startup và loop scaling; không phải confidence bound.
4. Full-confirm từng required seed của một config trước mở hàng loạt config mới. Ưu tiên hoàn tất seeds thiếu khi resume.
5. Chỉ mở targeted probe nếu còn probe cap và reserve cho confirmations; cap nhỏ hơn observed startup bị bỏ. Không quality-prune config học chậm.
6. Cost projection không fit ⇒ unresolved scheduling, không chứng minh config không khả thi. Mọi worker/probe/failure/retry/preflight/predictor overhead tính budget; teardown grace và overrun đo thật.
7. Audit raw evidence, chọn argmax U từ registry full-data/E/all-seeds, commit immutable decision. Nếu chưa đo đủ C thì best observed; hết C mới empirical finite optimum. No feasible choice ⇒ null selection, không dùng proxy thay thế.

Invariant có điều kiện: b còn lại, r reserve, k upper cost action; k+r+cleanup≤b và actual costs không vượt upper estimates thì sau action còn đủ r+cleanup. Chứng minh bằng phép trừ và quy nạp. Forecast thực tế có thể sai; không biến invariant thành guarantee mọi GPU, safety probability hoặc regret bound.

## Fidelity và task semantics

Detection: validation mAP50-95 checkpoint tốt nhất trong E, nhưng T đo đủ toàn E và final validation/saving. Scheduler horizon E cố định cả probe/full, warmup0/explicit optimizer/nbs=batch×accum, không implicit batch/AMP recovery.

Classification: final-epoch validation accuracy; checkpoint last tương ứng; optimizer AdamW/decay0.01 hiện cố định. Caption: generated validation BLEU4/chrF composite, best checkpoint trong E; tiny encoder, strict deterministic adaptive-cell pooling đã test output/gradient. Không gọi teacher-forced token loss là caption accuracy.

V2 currently single device, workers/cache fixed; không tự suy hỗ trợ distributed/LLM/segmentation từ image adapters. Bootstrap cài môi trường khác không là kiểm chứng tối ưu trên máy khác.

## Đối chứng và minh chứng độc lập

Default/Random full/Random proxy/profile-all/no-reserve và one-rung quality/utility SH cùng B, required-seed finalizer, grid và objective. Default phải trả full-confirmation cost. Cold subprocess và rotated ordering; oracle chạy sau khi decisions đã khóa, có cost riêng.

Regret chỉ với complete independent reference và feasible choice: `U_ref(best) - U_ref(chosen_config)`. Hai utility lấy từ cùng reference; thiếu hoặc thất bại ⇒ null, không 0. Conditional regret luôn báo kèm feasible rate để không giấu survivorship bias.

Case minh họa đã đo trên Fashion: width64 có mean accuracy≈0.713333, width32≈0.711111, nhưng width64 tốn T/RSS hơn. Với weights0.6/0.25/0.15 và anchors600s/1024MiB, width32 có U=0.380386, hơn runner-up chỉ0.000882. Đó là nghiệm thực nghiệm của preferences đã khóa, không chứng minh expected ranking ổn định qua future seeds.

Pilot 12C×3seeds×4epoch có reserve feasible3/3 và conditional regret0.034610, Random3/3 và0.040116; chưa có đủ independent domains/power để kết luận superiority. Profile-all2/3, regret mean trên2choices=0.029716, choice thất bại null. Source/controller repair và publication error sau commit được công bố trong [UPGRADE_VALIDATION.md](UPGRADE_VALIDATION.md), không sửa số đo để làm đẹp kết quả.

## Predictor: giới hạn cần công bố

Fit log-linear ridge T/M từ audited full outcomes; success-only labels, timeout/OOM không là exact full labels. Fit/calibration/evaluation disjoint dataset-content/hardware domains, một task/counter mỗi card. Strict runtime/physical hardware/training-source identity và feature range; outside scope fallback đo thật.

Residual p90/MAE/underprediction là diagnostics, chưa có validated coverage/safety interval. Adaptive choices/clustered trial rows không thỏa một guarantee đơn giản từ nominal95%. Chưa có quality surrogate/cost-aware EI/NEHVI trong release này; toàn bộ quality selection vẫn từ full measurements.

## Tái lập và tính toàn vẹn

protocol.json/source snapshots/model SHA/content IDs, request/raw/parent outcomes, process PID/create-time, checkpoints, event hash chain, immutable commits và report rebuild. Hash chứng minh nhất quán thông thường, không attestation hoặc scientific preregistration độc lập. Nghiên cứu đa GPU và final blind heldouts vẫn cần tài nguyên/thực nghiệm mới.
