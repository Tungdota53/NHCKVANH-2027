# Phương pháp thực thi 0.9.1

## 1. Bài toán và phạm vi kết luận

Đầu vào khóa trước search: dataset/splits, phần cứng/runtime, tập hữu hạn C, horizon E, tập seed S, chất lượng tối thiểu, trần thời gian/bộ nhớ, thang chuẩn hóa, trọng số và selection rule. Test không tham gia chọn config/checkpoint/objective.

Với cấu hình c được train trên toàn bộ train/val tới E cho mọi seed s:

- Q(c) = trung bình validation quality; Qmin(c) = min theo seed.
- T(c) = trung bình cold-worker wall time; Tmax(c) = max theo seed. Bao gồm khởi tạo/validation/checkpoint của worker, không phải tổng search.
- M(c) = max primary memory counter theo seed. CUDA là allocator reserved MiB, **không phải** toàn bộ VRAM; CPU là process-tree RSS, có thể đếm trùng trang chia sẻ.
- Feasible khi đủ mọi seed/horizon/evidence, Qmin≥qmin, Tmax≤time_limit (nếu đặt), M≤memory_limit. Timeout/OOM không được giả làm phép đo full chính xác.

Weighted objective thực thi:

`U(c) = wA*Q(c) - wT*T(c)/Tref - wM*M(c)/Mref`, với `w_i = weight_i/sum(weights)`.

Weights/scales là **ưu tiên do người dùng khóa trước**, không phải hằng số tối ưu khoa học suy ra cho mọi bài toán. Trong tập feasible, Pareto loại điểm kém hơn hoặc bằng trên cả3trục và kém hơn nghiêm ngặt ít nhất1trục. Presets chỉ lấy từ Pareto front, không chọn điểm lãng phí tài nguyên do hash tie-break.

Tchebycheff/TOPSIS dùng anchors/scales cố định; knee đo khoảng cách có dấu từ mặt phẳng qua3điểm cực trị của observed Pareto front. Knee không dùng weights trực tiếp; mặt phẳng suy biến fallback Tchebycheff đã khai báo. Không thay đổi rule sau khi xem test.

## 2. Lập lịch có ngân sách

Reserve ưu tiên model nhỏ theo descriptor tham số, rồi resolution/batch, không tra bảng tên GPU để giả số đo. Probe đo subset để ước lượng chi phí; selected vẫn cần full confirmation. Predictor card hợp phạm vi có thể cung cấp point prior; khi OOD phải fallback số đo thực hoặc công khai chưa dự báo được.

Cross-config cost policy2 tách `T_hat = H_ref + L_ref * area_ratio * width_ratio^beta * model_factor`. H là max observed non-loop overhead (không chỉ import; còn checkpoint/teardown), L là median observed loop. Beta fit log-OLS từ median loop của≥2width cùng configsetting, bỏ qua LR; clamp0..2, thiếu contrast fallback2. Chỉ dùng số đo online đã có, không nhìn independent reference/test để order. Giả định H ổn định ở width mới và fallback model_factor vẫn heuristic; metadata công bố số width/beta/raw slope. Không nhân toàn bộ cold-worker wall theo width² rồi coi đó là dự báo chính xác.

SH policy2:

1. Nếu chưa có bất kỳ forecast full-cost: bootstrap một cấu hình đủ seeds trước, mỗi worker unknown-cost có cap chia đều phần budget còn lại. Đây là fallback incumbent, không nói đã hoàn thành SH.
2. Với mỗi rung, chỉ so quality giữa các ứng viên cùng subset/seed/horizon. Mỗi rung restart độc lập, không lấy subset epoch làm full epoch.
3. Chỉ nhận rung nếu tổng probe forecast nằm trong quota `B*probe_fraction - probe_spent` và vẫn còn đủ dự phòng finalist.
4. Dự phòng finalist là `max_i(planning_time_i * missing_seed_count_i)`, không chỉ giá của config rẻ nhất. Trừ thêm cleanup/finalization reserve.
5. Nếu rung không đủ chi phí/forecast: ghi `rung_deferred`; rung dở không được dùng mixed-fidelity để loại ứng viên. Full-confirm finalists với seed/horizon đã khóa.

Probe charge gồm worker và controller/preflight quan sát được, giữ qua resume. Nếu phục hồi probe đã journal nhưng mất scheduler charge, khóa hết quota probe một cách bảo thủ, không coi lần chạy đó miễn phí. Forecast margin, cross-model scaling và bootstrap cap đều **heuristic**, không là upper-bound/hard-deadline guarantee.

Kết luận chỉ là `fully_trained_best_observed` khi chưa đo đủ C; `empirical_optimal_in_declared_space` chỉ khi mọi cấu hình đã đo full đủ S/E và có nghiệm feasible. Đây là optimum của phép đo hữu hạn, không là optimum kỳ vọng ở seed tương lai hay lúc hội tụ.

## 3. Hiệu năng không đánh đổi bằng chứng

- SHA streaming giảm cấp phát cả checkpoint. Mỗi execution full hash đầu/cuối; giữa trials kiểm stat signatures và băm lại file thay đổi. Cache metadata không chống adversarial tampering. Audit nguồn/checkpoint vẫn đọc bằng chứng gốc, không tin số tổng hợp.
- Cache decoded CPU images có tổng trần `min(requested_cache,10%available_RAM)`, chia giữa train/val/worker copies. Encoder trainable vẫn forward/backward từng mẫu. Trần không bao gồm model, optimizer, queues và pinned batches.
- Auto workers bảo thủ; người dùng có thể benchmark workers0..8. CUDA pin/nonblocking chỉ là phương án truyền dữ liệu, không thay precision/quality definition. Detection giữ RAM cache YOLO tắt vì cache toàn dataset có thể vượt RAM.
- Finalization reserve dựa trên thời gian full verification quan sát, số attempt và checkpoint bytes; bulk hash rate đo trên checkpoint thật, không dùng bytes/sec của hàng nghìn ảnh nhỏ để suy băng thông file lớn. Calibration read/checksum này cũng bị tính vào execution budget. Báo soft end-to-end đến primary report/CLI export; lần refresh timing receipt cuối không thể nằm trong clock của chính nó. Không im lặng bỏ cost audit hoặc gán mọi overhead cho hashing.
- Effectivebatch=`batch*accumulation` là số mẫu danh nghĩa mỗi update, không bảo đảm BatchNorm/stochastic training giống một microbatch lớn. Vì vậy quality phải đo thật theo từng cấu hình.

## 4. Minh chứng bắt buộc

Benchmark khóa quyết định online trước reference độc lập, dùng cùng C/E/S/objective/data/device/runtime/execution policy. So feasible rate, thời gian tới full incumbent, actual online cost, conditional regret trên **cùng reference**, failures riêng (null regret, không0giả). Reference phải đủ toàn bộ C×S; test vẫn đóng.

Để chứng minh cải thiện phải lặp search seeds, nhiều dataset và nhiều GPU vật lý, công bố failures/variance/chi phí calibration và kiểm định phù hợp. Một pilot local không đủ statistical superiority. Predictor chưa có interval coverage/khả năng chuyển GPU mới được xác nhận. GP/qNEHVI, worker pool và exact checkpoint continuation chưa được triển khai/nghiệm thu trong0.9.1.

SH có thể loại nhầm cấu hình học chậm nhưng tốt hơn ở full horizon. Full-confirmation chỉ xác nhận finalist đã đo, không chứng minh các ứng viên bị loại kém hơn. Thực nghiệm hai rung 0.9.1 đã quan sát trường hợp này; phải giữ nhãn best-observed khi thiếu full coverage. Predictor fit-only từ reference đã biết chỉ dùng kiểm thử nhánh kỹ thuật, không được coi là search độc lập hoặc calibration safety interval.

## Nguồn nền tảng

- [Li et al., Hyperband, JMLR2018](https://arxiv.org/abs/1603.06560): phân bổ tài nguyên thích nghi/early-stopping trong HPO. Policy bootstrap/reserve ở đây là biến thể kỹ thuật phải tự benchmark; không kế thừa tự động guarantee/speedup của bài báo.
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): workers/pinned memory cần điều chỉnh theo workload/phần cứng. Không suy mức tăng tốc cụ thể từ hướng dẫn này.

Các số đo phiên bản mới và đường dẫn raw artifacts ở `UPGRADE_091_VALIDATION.md`; unit fixtures không được tính là thực nghiệm ML.
