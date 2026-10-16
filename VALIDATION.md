# Kết quả xác minh ngày 2026-10-12

Đã hoàn thành thuật toán và mã nguồn bản thử nghiệm. Đây là kết quả kiểm tra chức năng, chưa phải thực nghiệm chứng minh hiệu quả tối ưu trên dataset thực.

## Môi trường

- Python 3.12.14, PyTorch 2.14.1+cpu, Ultralytics 8.4.175.
- Đã cài project dạng editable; cả `python -m yolo_tuner` và `yolo-tune.exe` hoạt động.
- `uv pip check`: 42 package tương thích.
- `compileall`: thành công.
- NVML đọc được RTX 3050 Laptop 6 GB, compute capability 8.6, driver 616.92. Python hiện dùng bản CPU; chưa kiểm thử huấn luyện CUDA, precision GPU, VRAM peak trên GPU hoặc recovery OOM thực tế.

## Các kiểm tra đã đạt

| Kiểm tra | Kết quả |
|---|---|
| Unit và CLI tests | 13/13 đạt |
| Pareto, preset và ràng buộc time | Đạt; không tạo lựa chọn khi tập khả thi rỗng |
| Budget/deadline và rung bị gián đoạn | Giữ cohort cùng fidelity; thử subprocess timeout thực tế cũng đạt |
| Dataset | Lấy mẫu nhiều nhãn, lớp hiếm/vật thể nhỏ/ảnh âm tính; từ chối train/val trùng đường dẫn và nhãn sai |
| Offline model | Fit/calibration/test tách theo dataset group; từ chối GPU/runtime/đặc trưng ngoài miền và schema cũ |
| Ngưỡng final accuracy thiếu predictor | Không dùng early mAP để chứng nhận; không xuất selected.json |
| YOLO profile thật trên CPU | Batch 1/2 ở imgsz 128/160 đều hoàn tất, tạo optimizer state |
| Tìm kiếm + LR refinement | Hoàn tất 0.5×/1×/2× LR; khoảng 51.969 giây / ngân sách 120 giây |
| Full training từ cấu hình đã xuất | 3/3 epoch hoàn tất, lưu best.pt và last.pt; khoảng 13.078 giây |
| Thu thập offline tự động | Một full run hoàn tất và ghi CSV |
| Export offline từ full training | Ghi thêm số đo đúng schema, kèm runtime protocol |
| Fit bằng một dataset tổng hợp | Từ chối vì thiếu nhóm calibration/train/test độc lập; không tạo predictor giả |

## Artifacts để kiểm tra lại

- Tìm kiếm hai kích thước ảnh: `runs/smoke_search/20261012_144108_514809/report.json`.
- Tìm kiếm sau khi bổ sung thời gian khởi động và kiểm tra LR: `runs/smoke_search_final/20261012_144821_113403/report.json`.
- Cấu hình lựa chọn: `runs/smoke_search_final/20261012_144821_113403/selected.json` và `train.yaml`.
- Full training: `runs/smoke_full_final/20261012_145851_078103/training_report.json`.
- Thu thập offline: `runs/smoke_collection/20261012_145542_013098/collection.json`.
- Số đo: `runs/smoke_measurements_final.csv`.

Dataset thử gồm 24 ảnh train, 8 ảnh validation, một lớp hình chữ nhật; model `yolo11n.yaml` khởi tạo ngẫu nhiên. mAP của thử nghiệm ngắn là 0, nên các kết quả trên chỉ chứng minh luồng chương trình chạy được. Không dùng cấu hình hoặc CSV này làm khuyến nghị cho dataset thực hay bằng chứng về độ chính xác.

Chưa có dataset thực của người dùng và bộ predictor đã fit từ thực nghiệm đầy đủ. Vì vậy chưa thể đưa ra cấu hình tối ưu riêng cho bài toán thực hoặc khoảng dự đoán đã kiểm định cho máy này. Các công cụ thu thập/fit cần thiết đã có trong mã nguồn.

Không còn tiến trình thử nghiệm/cài đặt thuộc tác vụ này đang chạy khi xác minh hoàn tất.
