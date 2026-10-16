# Kế hoạch hoàn thiện bộ tối ưu cấu hình 0.10.0

Ngày khóa thiết kế: 2026-10-16. Sản phẩm chính là thuật toán tìm cấu hình cân bằng chất lượng, thời gian và bộ nhớ; CLI chỉ là phương tiện thực thi.

## Giả thuyết và phạm vi

Giả thuyết cần kiểm chứng: lựa chọn thích nghi từ các phép đo full đã hoàn tất, có tính chi phí và độ mới/độ bất định, sử dụng ngân sách hữu ích hơn thứ tự cố định mà không loại ứng viên theo chất lượng epoch sớm. Không giả định giả thuyết đúng trước thực nghiệm. Không tuyên bố thuật toán hoàn toàn mới so với tài liệu hoặc có bảo đảm tối ưu toàn cục.

Đầu vào khóa: dữ liệu/splits, máy/runtime, grid hữu hạn, full horizon E, training seeds S, objective Q/T/M, budget và siêu tham số acquisition. Đầu ra: cấu hình feasible đã full-confirm đủ S/E, Pareto, coverage, chi phí và lý do chọn từng hành động. Chưa đo đủ grid chỉ là best-observed.

## Các bước triển khai

1. Thêm acquisition module: kernel ridge từ **quality full-data/full-horizon/đủ training seeds**, mixed config features, heuristic uncertainty/novelty, optimistic utility improvement trên chi phí xác nhận đủ seeds. Không đọc probe quality, test hoặc reference.
2. Thêm `adaptive_full` coordinator: bootstrap, ưu tiên hoàn tất cấu hình dang dở, tiếp nhận hành động theo all-seed cost forecast, không permanent early-quality pruning; journal mọi acquisition/admission và fallback. Forecast không là hard bound.
3. Khóa thuật toán/siêu tham số trong protocol, giữ compatibility với 0.9.1. Thêm ablation bỏ cost, uncertainty và diversity; CLI cho chọn rõ, không gắn kết quả mới cho source cũ.
4. Thêm phân tích study: đối chứng cùng budget nhỏ hơn measured exhaustive cost, reference độc lập sau quyết định; paired repeats, no-selection riêng, CI chỉ khi đủ repeats và không giả nhiều training seeds là nhiều dataset.
5. Unit/regression tests: không leakage, không chọn probe, slow starter, deterministic resume/order, partial confirmation, OOM/timeouts, finite acquisition, budget và historical raw audit.
6. Training thật theo source đã khóa: search có budget hữu hạn với reference full sau search; kiểm tra CPU và GPU theo nguồn lực laptop, không concurrent GPU workers. Báo rõ horizon, số repeats/domain và không thay đổi thuật toán dựa trên kết quả rồi gọi cùng study là độc lập.
7. Tài liệu hóa phương trình, pseudocode, độ phức tạp, lựa chọn tham số, phần kế thừa nghiên cứu trước, thực nghiệm âm/dương và giới hạn. Đóng gói, kiểm thử hai runtime và nghiệm thu.

## Tiêu chí nghiệm thu

- Mã thực sự thực thi acquisition thích nghi và lưu lý do quyết định, không chỉ đổi tên một thứ tự tĩnh.
- Chỉ full confirmations được chọn; early learning curve không thể trực tiếp loại một config trong adaptive_full.
- So sánh không dùng test hoặc reference để ra quyết định; chi phí tìm kiếm và chi phí reference tách biệt.
- Không coi unit fixture/offline replay là training thật; không coi một local study là proof thắng mọi dataset/GPU.
- Nếu chưa thắng Random hoặc ablation không cho lợi ích, công bố đúng kết quả và không gắn nhãn superiority.

## Trạng thái

Đã hoàn tất triển khai và nghiệm thu kỹ thuật bản **0.10.0** theo các bước trên. Source được khóa trước các lượt training cuối; cả 17 protocol mới khớp đủ 60 source hashes. Hai môi trường chạy 329 tests thành công (CPU bỏ qua 1 test chỉ dành cho CUDA); wheel đã kiểm tra cài đặt và đối chiếu source.

Thực nghiệm đã hoàn tất: hai matched studies MNIST/CPU và CIFAR-10/CUDA, reference độc lập, integration learned acquisition có 2 fitted steps với 8/8 full outcomes, và YOLO detection có 1 fitted step với 6/6 full outcomes. Tổng 93 attempts mới được audit: 85 full thành công, 7 full timeout và 1 probe timeout. Pilot B60 dở dang được giữ riêng, không tính là completed benchmark.

**Đây là hoàn thành phạm vi phần mềm và kiểm chứng thực thi, không phải hoàn thành chứng minh ưu thế khoa học.** Hai matched studies chỉ có R1 và không đi đến learned-fit action; các integration chứng minh nhánh học thực thi thật nhưng không phải đối chứng hiệu suất. Chưa đủ bằng chứng để kết luận tốt hơn Random về mặt thống kê, hội tụ/chất lượng ứng dụng, hoặc tối ưu trên GPU/dataset chưa quan sát. Nghiên cứu tiếp theo cần nhiều search repeats độc lập, nhiều phần cứng/dataset, kiểm tra độ nhạy objective và hiệu chuẩn uncertainty/resource forecasts; không được trình bày các việc này như đã hoàn tất.

Chi tiết phương pháp: `ALGORITHM_010_METHOD.md`; thiết kế và điều chỉnh nguồn lực: `ALGORITHM_010_EXPERIMENTS.md`; số đo, kiểm tra và giới hạn cuối: `ALGORITHM_010_VALIDATION.md`.
