# Nâng cấp 0.9.1 — sửa lỗi và giảm chi phí thực thi

Phạm vi: pipeline nghiên cứu cấu hình cho detection/classification/captioning được hỗ trợ. Không hứa tối ưu toàn cục trên mọi máy/dataset. Không sửa dataset, checkpoint, snapshot hoặc báo cáo lịch sử.

## Kế hoạch thực hiện

1. Sửa quyết định: fallback predictor OOD, preset không bị dominated khi hòa điểm, thứ tự model theo descriptor; khóa policy mới trong protocol.
2. Bảo vệ ngân sách: SH giới hạn probe và dự phòng full-confirmation đủ seed; dự phòng final audit theo chi phí quan sát; phân biệt thời gian đến commit và tổng xuất báo cáo. Không biến forecast heuristic thành hard-deadline guarantee.
3. Giảm I/O điều khiển: SHA streaming; cache xác minh trong một execution bằng metadata với full hash lúc đầu/cuối; kiểm chứng thay đổi nội dung và công khai giới hạn metadata cache.
4. Nâng cấp dữ liệu trong module v2 riêng: cache ảnh decoded có trần RAM, Dataset picklable trên Windows, workers/pinned memory/nonblocking theo policy khóa trước; giữ nguyên backend legacy để audit lịch sử.
5. Đóng gói/CLI: module entrypoint, benchmark không phụ thuộc root script; xuất lại benchmark từ bằng chứng đã audit, từ chối che giấu hard crash có chi phí không ghi nhận; launcher chọn đúng interpreter.
6. Nghiệm thu: regression tests cả CPU/CUDA runtime; kiểm tra wheel ngoài source checkout; training thật ngắn CPU, GPU detection và captioning; benchmark có reference hoàn tất trên dữ liệu local, không mở test mới.
7. Đồng bộ README/hướng dẫn/phương pháp; ghi kết quả đo thật, phần chưa làm và giới hạn so sánh.

## Nguyên tắc an toàn / khoa học

- Chỉ full-data/full-horizon/all-required-seeds mới đủ điều kiện selection; không nâng probe thành chứng nhận.
- Metadata cache phát hiện thay đổi thông thường, không là attestation chống sửa độc hại. Full hash cuối execution vẫn bắt buộc trước commit.
- Không tái gán số đo 0.9.0 cho mã mới; benchmark mới có source/protocol riêng. Không chạy nhiều ML workers cạnh tranh GPU khi đo.
- Không đổi batch/precision giữa chừng rồi giữ chứng nhận cũ. Resume epoch/worker pool chưa được kiểm chứng không được giả gắn nhãn hoàn tất.
- Độ chính xác tương lai, unseen GPU, mọi dataset, convergence và statistical superiority cần thực nghiệm thêm.

## Trạng thái

**Hoàn tất phạm vi nâng cấp kỹ thuật 0.9.1** ngày 2026-10-15. Đã thực hiện các mục 1–7 và ghi số đo/bằng chứng trong [UPGRADE_091_VALIDATION.md](UPGRADE_091_VALIDATION.md); thuật toán, công thức và giới hạn ở [METHOD_091.md](METHOD_091.md).

- 283 tests qua CUDA runtime; CPU 282 qua + 1 CUDA-only skip.
- 17 lượt chạy thật / 121 attempts qua raw audit; có CPU, GPU detection, GPU captioning và deployment source cuối. Training nghiệm thu ngắn E2, không giả thành convergence/production quality.
- Benchmark local cuối đủ reference: cả reserve/random/SH đo 12/12 full outcomes trong budget 90s/phương pháp. Một repeat sau hiệu chỉnh trên cùng domain không chứng minh superiority.
- Nhánh SH hai rung thực tế chạy được nhưng proxy đã loại cấu hình full-quality tốt hơn; giữ nhãn best-observed, công bố kết quả âm thay vì tuyên bố tối ưu.
- Release wheel cài/import ngoài checkout, 57 packaged source hashes trùng bản cuối; metadata dependency cập nhật ở cả hai môi trường. Không chứng nhận mọi OS/GPU hoặc mọi dependency version.
- Legacy 19/19 file không đổi; dataset/checkpoint/snapshot/report lịch sử được giữ nguyên.

Các hạng mục nghiên cứu tiếp theo **chưa thực hiện**: nhiều GPU vật lý và dataset độc lập, nhiều search repeats có kiểm định, full horizons để đánh giá hội tụ, calibration/evaluation domains độc lập cho predictor, worker autotuning/pool, exact epoch continuation, GP/qNEHVI và tabular adapter. Đây là giới hạn được công khai, không phải tính năng đã hoàn thành trong 0.9.1.
