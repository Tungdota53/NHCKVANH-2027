# Kiểm chứng CLI 0.8.0 — 2026-10-14

## Bản sửa 0.8.1 — chẩn đoán dataset đầu vào

Đã tái hiện lỗi với `C:\Users\<USER>\Downloads\archive` (ẩn tên tài khoản trong bản công khai): thư mục chỉ có `marketing_campaign.csv`, header 29 cột phân cách TAB. File được nhận dạng là bảng, chưa được training bằng vision adapter; không tự chọn `Response`/`Income` làm target.

207 tests đạt trên `.venv`; `.venv-cpu` chạy207, đạt206 và bỏ qua1 CUDA-only. 13 tests mới kiểm tra CSV/TAB/semicolon, header-only/bounded reads, table không fallback thành ảnh gần đó, caption CSV thiếu ảnh, thư mục trống/file nén, không tự giải nén, hint cụ thể, retry/back tại bước dataset, dữ liệu không hợp lệ không gọi GPU runtime/training, và lệnh activate chỉ được xử lý như input chứ không thực thi.

Thay đổi là nhận dạng/lỗi/hướng dẫn nhập liệu, **không bổ sung tabular training**. Source guards/checkpoints/runs cũ giữ nguyên; source mới không tương thích live resume của protocol khóa bản0.8.0. Các kết quả full training dưới đây vẫn là số đo của source0.8.0 đã snapshot, không gán lại thành thực nghiệm0.8.1.

## Kiểm chứng ban đầu 0.8.0

Nâng cấp giao diện và điều khiển thao tác; không thay objective, scheduler, định nghĩa quality hay điều kiện chứng nhận của 0.7.1. Không thêm dependency UI. Package 0.8.0 đã cài editable vào `.venv` và `.venv-cpu` bằng `--no-deps`; không nâng PyTorch/Ultralytics.

## Kiểm thử tự động

- `.venv`: **194 tests passed**.
- `.venv-cpu`: **194 tests chạy, 193 passed, 1 CUDA-only skipped**.
- Trong đó 32 tests UX mới: TTY/non-TTY, no-argument menu/help, typo, tương thích argv lệnh cũ, JSON, default-no confirmation, back/EOF/Ctrl+C, số không hợp lệ/NaN/Inf, đường dẫn có nháy/khoảng trắng, phút sang giây, đủ seeds, dataset không nhãn không được launch, thời gian nhập menu không tính budget, Unicode/ASCII, câu lệnh PowerShell literal, deployment không bị gán thành optimization, heartbeat/partial JSON/deadline.
- `compileall` và `uv pip check` ở cả hai environment: đạt.
- Launcher `cli.ps1`: PowerShell parser không báo lỗi.

Các fixture/mock trong tests **không phải số liệu benchmark nghiên cứu**.

## Chạy thật qua terminal tương tác

Đã mở `python train_config.py` bằng PTY Windows, đi qua menu và wizard với đường dẫn dataset có nháy/khoảng trắng. Dataset MNIST thật đã chuẩn bị ở `runs/datasets/mnist_v2`: 1000 train / 300 val / 500 test; CPU, compact 3 cấu hình, 3 seeds (17/29/43), 2 epoch, budget 180 giây. Không mở test.

Run: `runs/v2/20261014_103057_145628`.

- Hoàn tất **9/9 full-data/full-horizon config × seed**, cộng 2 probe: **11 raw attempts audit passed**.
- Thời gian session ghi trước commit: **118.843 giây**; worker attempts **92.125 giây**, trong đó probe **12.171 giây**. Các spans worker nằm trong session, không cộng lại thành tổng chi phí. Budget overrun **0**.
- Selected: CNN classifier width16, image32, batch16, accumulation2, FP32, AdamW, LR0.003, weight decay0.01.
- Validation accuracy **0.552222 ± SD0.012620**; mean full-worker time **8.886 giây**; max process-tree RSS **483.887 MiB**, không gọi là VRAM.
- `empirical_optimal_in_declared_space` chỉ trong tập 3 cấu hình, 3 seeds và horizon **2 epoch** này; không là bằng chứng đã hội tụ, chất lượng ứng dụng tốt hoặc optimum trên GPU/dataset khác.
- Đã quan sát epoch/heartbeat trong CLI; `status` ở cửa sổ khác xác nhận worker PID đang sống; sau hoàn tất menu quản lý đọc kết quả và audit mà không train lại.
- Báo cáo `REPORT.md`, raw request/result/checkpoint/ledger được giữ trong run; không thay đổi dataset hoặc test history cũ.

## Kiểm tra hủy worker thật

Run riêng: `runs/cli_cancel_check/20261014_103515_859258`, CPU, 50 epoch, budget90s. Một probe hoàn tất, sau đó gửi Ctrl+C khi full worker đang khởi tạo:

- Cha ghi `phase=interrupted`; parent result ghi `status=interrupted`, `KeyboardInterrupt`.
- Process record có return code không thành công; không còn process ML của phép thử.
- **Không có selected.json**; không gán full training hoàn tất hay quality bằng 0 giả.
- Raw logs/request/result và event `trial_completed`/`session_interrupted` được giữ. Không có checkpoint mới thì không tuyên bố đã tạo checkpoint.

Audit trực tiếp của run bị hủy báo `Completed worker omitted from state; reconcile before resume`: đây là trạng thái recovery đã thiết kế của backend, vì interrupt xảy ra sau journal outcome và trước scheduler state-save. Không bỏ guard hoặc sửa state trong kiểm tra này. Resume chính thức sẽ reconcile các outcome đã journal trước audit và cần budget bổ sung/source tương thích. Đây không phải audit passed của một full-training certificate.

## Bảo toàn và giới hạn

- `show-training` vẫn đọc được full run ViECAP cũ `runs/viecap_production/20261012_173534_642290` và test/checkpoint đã lưu.
- Audit run detection 0.7.1 `runs/verified_detection_071/20261013_111148_405825`: **24 attempts passed** sau nâng cấp.
- So SHA256 19 file `yolo_tuner/*.py` với protocol detection 0.7.1: **0 file thay đổi**.
- Source mới làm protocol cũ không còn tương thích live resume/evaluation; guard được giữ, không đổi hash/protocol cũ để ép chạy.
- Các lệnh train lại/evaluate/resume không được tự chạy trong menu. Luồng backend đã có kiểm chứng trước; lần kiểm tra UX này không train lại toàn ViECAP, không mở test mới, không kiểm chứng GPU vật lý khác hay môi trường cài mới.
