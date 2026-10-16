# Kế hoạch và tiêu chí nghiệm thu 0.9.0

Phạm vi: nâng cấp pipeline nghiên cứu cấu hình huấn luyện vision, ưu tiên YOLO detection trong đề tài. Giữ nguyên dataset, checkpoint, source snapshots và kết quả lịch sử. Không đoán target của bảng CSV; không tuyên bố tối ưu mọi GPU/dataset hoặc giả minh chứng thực nghiệm còn thiếu.

## 1. Hợp đồng thực thi

- Sửa benchmark gọi doctor bằng JSON và kiểm tra runtime trước training.
- Training không đủ bằng chứng trả mã thất bại; lỗi subprocess/timeout có thông báo xử lý.
- Từ chối epochs=0 thay vì ngầm đổi sang horizon mặc định.
- Regression tests cho các lỗi đã tái hiện.

## 2. Không gian ứng viên và xác minh thực thi

- Hỗ trợ Freeze cho YOLO checkpoint local; ghi số tham số trainable/frozen và kiểm tra số lớp freeze thực tế.
- Giữ core/compact để tương thích; thêm research space có các tổ hợp model/image/batch/LR/freeze rõ ràng.
- Không tự tải pretrained weights; không đóng băng backbone chưa được học trong preset tự sinh.
- Model descriptors lấy từ kiến trúc local; mọi chi phí chuẩn bị nằm trong budget được ghi.

## 3. Chọn cấu hình và báo cáo

- Khóa selection rule trước search: weighted, Tchebycheff, TOPSIS hoặc knee có fallback cho mặt phẳng suy biến.
- Chỉ dùng full-data/full-horizon/all-required-seeds thỏa constraints; không dùng test/proxy để cấp chứng nhận.
- Xuất Pareto có đầy đủ số đo và Accuracy/Time/Memory/Balanced presets; phân biệt predicted với measured, best observed với full-space.
- Benchmark, deployment và audit dùng cùng selection rule. Các protocol cũ thiếu trường mới vẫn được rebuild theo đúng semantics cũ.

## 4. Predictor offline

- Đặc trưng có kiến trúc/cỡ model, pretrained/freeze, tham số/FLOPs nếu đọc được, dataset và cấu hình.
- Card version mới; kiểm scope model/runtime/hardware/feature. Card không tương thích fallback đo thật.
- Fit T/M và quality point priors từ audited full outcomes, không từ probe/test. Chất lượng dự đoán không thay full confirmation.
- Sai số và quantile theo nhóm domain; công bố thiếu calibration groups và giả thiết exchangeability. Không biến p90/margin heuristic thành safety guarantee.

## 5. Search và minh chứng

- Bổ sung successive halving nhiều rung, horizon scheduler cố định, subsets cố định, restart mỗi rung và tính đủ mọi chi phí.
- Full-confirm finalists nhiều seed; nếu budget thiếu thì không tạo nghiệm giả.
- Benchmark các policy dưới cùng grid/budget/objective/seeds và reference độc lập; regret tính cùng reference metric/rule.
- Báo feasibility rate, failures, cost và proxy-ranking diagnostics; không kết luận superiority từ một pilot.

## 6. Nghiệm thu

- Toàn bộ tests ở CUDA runtime và CPU runtime; contract tests CLI/benchmark, predictor kiến trúc và freeze/rules/rungs.
- Training thật ngắn trên dữ liệu thật đã có local (CPU classification và GPU detection), không mở test mới.
- Benchmark nhỏ qua CLI, audit raw evidence và đối chiếu source lịch sử.
- Hướng dẫn CLI/method/limitations và release validation chỉ ghi kết quả thực sự đã chạy.

## Không thể thay bằng code

Kiểm chứng nhiều GPU vật lý, dữ liệu mới độc lập, training đến hội tụ, coverage trên domain mới và superiority có statistical power cần thêm tài nguyên/thực nghiệm. Gaussian-process/qNEHVI và recovery đổi batch không được gắn nhãn đã hoàn thành nếu chưa triển khai và kiểm chứng. Recovery sửa cấu hình phải là provenance mới, không tiếp tục dùng chứng nhận cấu hình cũ.
