# Thiết kế thực nghiệm khóa trước kết quả 0.10.0

Ngày 2026-10-16. Đây là development pilot trên laptop, không là blind-domain hoặc unseen-GPU evaluation. MNIST/CIFAR/Penn-Fudan đã xuất hiện trong lịch sử development, dù chưa có reference cho grid/horizon/source mới này.

## Study A — CPU algorithm và ablation

- Dataset: MNIST local `runs/datasets/mnist_v2`, nguyên split train 1000/val 300/test 500. Test đóng.
- Grid `configs/algorithm_010_grid.json`: 8 ứng viên width {16,32} × LR {0.0001,0.0003,0.001,0.003}; index0 giữ cấu hình mặc định thực của adapter width16/LR0.001, không cố tình chọn LR kém làm default baseline. Các tham số còn lại theo classification adapter (32px, batch16, accumulation2, AdamW, FP32, weight_decay0.01).
- Horizon full **6 epochs**, training seeds **17,29**, search seeds **17,29,43**. Đây là full fixed-horizon đã khóa, không là chứng minh hội tụ.
- Methods: adaptive_full, adaptive_no_cost, adaptive_no_uncertainty, adaptive_no_diversity, random, reserve, successive_halving, default.
- Budget mỗi online run **60s**, reference tối đa **600s** chạy sau khi mọi quyết định đã khóa. Reference phải đủ 8×2=16 full outcomes; nếu không đủ, không có regret/optimum giả.
- Objective weighted [0.6,0.25,0.15], **time_scale15s, memory_scale512MiB**, max_memory2048MiB, min_quality0. Đây là thang ưu tiên tài nguyên khóa trước kết quả, chọn để không làm time term gần bằng0 ở worker CPU ngắn; không phải scale tối ưu suy ra từ reference. Cache64MiB, workers0, pin off. Rungs subset1/2/4, train_subset128/val_subset64, probe_fraction0.1, cost_margin1.25, cleanup5s.
- Acquisition ridge0.05/length1/exploration0.25/novelty0.05 giữ nguyên; không tune từ reference. Rotate method order theo repeat như benchmark runner.

## Study B — second dataset check

- CIFAR-10 local `runs/datasets/cifar10_research`, cùng split sizes, grid, E/S/objective/loader/budget như Study A.
- Methods adaptive_full và random, search seeds17/29/43; independent full reference sau cả 6 online runs. Không dùng Study A reference/card để fit Study B.
- R3 không đủ bootstrap CI đáng tin hoặc tuyên bố superiority. Ghi missing baselines của study B; không đánh đồng hai dataset với nhiều máy/GPU.

## GPU integration

Chạy adaptive_full trên Penn-Fudan/grid detection đã có, full horizon3, seeds17/29, budget hữu hạn. Đây là kiểm tra adapter/coordinator thực thi trên RTX3050, không gộp thành study đối chứng độc lập hoặc chứng nhận transfer sang GPU vật lý khác.

Tham số launch khóa trước lượt detection: `configs/detection_090_verification.json` (YOLO11n/freeze0, YOLO11n/freeze5, YOLO11s/freeze0; local pretrained weights), E3/S17,29/search17, B600s/trial_timeout180s, FP16, workers0/cache64/pin auto, Tref15/Mref512/max_memory2048. Không import prior/quality/reference từ studies classification; task/metric detection riêng.

## Báo cáo bắt buộc

Raw audit; coverage config×seed; feasible/no-selection; actual end-to-end seconds và overrun; time-to-first full incumbent; conditional regret trên cùng reference; paired regret differences và số paired repeats. Kiểm tra **B < measured reference cost** sau khi reference xong: đây là điều kiện kiểm tra thực nghiệm, không lấy reference để quyết định online budget. Không dùng chi phí reference để sửa bất cứ quyết định search nào.

Giữ cả kết quả âm. Không hứa adaptive thắng Random hoặc mọi ablation ở một pilot. Ghi riêng chi phí reference, không gọi nó là chi phí miễn phí của thuật toán online.

## Revision 2 — resource feasibility, trước mọi kết quả quality/reference

Pilot gốc `runs/algorithm_010_mnist/20261016_150413_175270` được dừng sau hai child hoàn tất không có full outcomes và child thứ ba timeout/interrupted. Không có validation quality hoặc reference để tune thuật toán. Giữ nguyên toàn bộ protocol/raw của pilot; nó **không phải completed benchmark** và không nhập vào matched analysis.

Quan sát chi phí: prep23–28s ở hai child đầu; boundary/reserve chiếm thêm ~7–12s/~13–24s; worker cap3–14s không đủ khởi tạo và E6. Diagnostic import TorchCPU8.5s. Budget60 không thích hợp ở host hiện tại, không là kết luận phương pháp tốt/xấu.

Khóa revision này trước lượt operational đầu:

- **B180s**, reference budget **900s**, search seed **17** (R1 pilot), vẫn E6 và training seeds17/29. Giảm repeats để giới hạn chi phí laptop; không báo R3 đã hoàn tất hoặc CI/superiority.
- Grid riêng `configs/algorithm_010_operational_grid.json` gồm 12 configs: C8 cũ + 4 resolution48 contrasts (width16/32 × LR0.001/0.003). Mở rộng theo prior worker-cost để còn là bài toán budget-constrained; không dựa quality/reference chưa tồn tại. Default index0 không đổi.
- MNIST: đủ8 methods như Study A; CIFAR: adaptive_full/random. Mọi search của mỗi study đều khóa trước reference riêng. Các tham số acquisition/objective/loader khác giữ nguyên thiết kế ban đầu.
- Không sửa source hoặc acquisition defaults từ pilot. Nếu reference không đủ hoặc B không nhỏ hơn reference đo thật, analysis phải từ chối kết luận budget-limited; không đổi nhãn để che thiếu bằng chứng.

### Study B backend clarification, trước khi chạy Study B

Study B dùng **CIFAR-10 trên CUDA RTX3050**, interpreter `.venv`, FP32, cùng C12/E6/S2/B180/R1 và objective như operational Study A; `pin_memory=auto`, workers0/cache64. Lý do: mở rộng kiểm tra sang backend GPU đang có và tránh nhân đôi chi phí CPU worker startup lớn. Không chuyển measured priors/quality/reference từ MNIST sang CIFAR. Đây là dataset khác + backend khác trên cùng host, không phải GPU vật lý mới và không là phép đối chiếu CPU-vs-GPU có kiểm soát. Study B chỉ adaptive_full/random với reference riêng; analysis phải tách hai studies, không gộp paired counts thành independent multi-GPU proof.

### Learned-acquisition integration receipt

Một lượt riêng CIFAR/CUDA FP32, grid `configs/algorithm_010_fit_integration.json`: 4 LR {0.001,0.0003,0.002,0.003}, còn lại width16/img32/batch16/accum2/AdamW. Full E6/S17,29, B360s, searchseed17, Tref15/Mref512/objective như operational study. Mục đích là thực thi model-fit acquisition thật sau >=2 full configs, không chứng minh efficiency qua việc đo hết grid nhỏ. Chỉ gọi nhánh fit đã kiểm chứng nếu có >=2 history receipts `surrogate.fitted=true` và 8/8 full workers; nếu thiếu phải ghi coverage thực. Không chia sẻ mô hình/quality từ lượt này vào Study B hoặc bất kỳ online run nào.

Trước khi launch lượt integration, budget được tăng từ360 lên **600s** vì worker CIFAR/CUDA thực tế vẫn có startup/epoch cost lớn trên host hiện tại; đây là kiểm tra branch execution, không budget-efficiency benchmark. Grid/E/S/acquisition không thay đổi. Ghi budget600thật, không báo như đã chạy trong360s.

### Reference completion grant

MNIST reference initial900s kết thúc ở21/24full outcomes, có1timeout. Cấp thêm **300s** qua resume cho reference, giữ nguyên source/grid/E/S/objective và boundary các online decisions. Không cấp thêm budget online hoặc thay quyết định search. Tổng reference budget1200s và mọi timeout/audit/resume cost phải được công bố; không gọi900sbanđầu là đủ hoặc coi phần bổ sung miễn phí.

CIFAR/CUDA reference initial900s kết thúc ở19/24full outcomes; cấp thêm **600s** riêng cho reference để hoàn tất phần thiếu. Tổng grants1500s; source/grid/E/S/objective và online budget180s/decisions giữ nguyên. Chi phí reference cuối lấy từ cumulative clock sau resume, không dùng900s như chi phí giả định.
