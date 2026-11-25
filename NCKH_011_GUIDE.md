# Hướng dẫn Thực nghiệm Nghiên cứu Khoa học (NCKH) — Phiên bản 0.11.0 (Chuẩn mAP50)

> **Lưu ý phiên bản:** Tài liệu này lưu giữ hồ sơ lịch sử và số đo kiểm chứng của phiên bản **0.11.0**. Đối với phiên bản **1.0.0** hiện hành (GP/Adaptive Kernel Ridge LOOCV, `adaptive_ehvi`, `adaptive_staged_seeds`, `observed_cross_memory` + OOM Guard, `dedup-dataset`, `evaluate-v2` latency/PR/F1/Confusion Matrix, Anytime `AUARC`, Wilcoxon/Sign-Flip/Holm-Bonferroni, `paper-export --lang vi|en`, Rich CLI & Offline Web Dashboard Studio), vui lòng xem **[NCKH_100_GUIDE.md](NCKH_100_GUIDE.md)**.

Tài liệu này mô tả đầy đủ các nâng cấp trong bản **0.11.0** để phục vụ một bài báo / đề tài **Nghiên cứu Khoa học (NCKH)** về tối ưu hóa cấu hình huấn luyện mô hình thị giác máy tính (đặc biệt là Object Detection với YOLO) có ràng buộc ngân sách tài nguyên.

---

## 1. Các điểm đã sửa và nâng cấp cốt lõi trong 0.11.0

1. **Chuyển chỉ số chất lượng chính của Object Detection sang `mAP50` (`IoU = 0.50`):**
   - Trước đây (`<= 0.10.0`), worker và schema của pipeline dùng `mAP50-95` làm $Q(c)$, đồng thời cơ chế lưu `best.pt` mặc định của Ultralytics 8.4 (`Metric.fitness`) cũng dùng trọng số `[0, 0, 0, 1]` (tức `100% mAP50-95`).
   - Từ **0.11.0**, mặc định trên toàn bộ CLI, Menu/Wizard, `optimize`, `benchmark`, `plan-search`, `research`, `verify-config` là **`--detection-metric map50`**:
     - $Q(c, s)$ là **best validation `mAP50`** trong chân trời $E$ epoch đã khóa.
     - Hàm `Metric.fitness` trong tiến trình worker được đồng bộ trực tiếp với `mAP50`, bảo đảm tệp trọng số **`best.pt` trên đĩa được lưu tại đúng epoch đạt cực đại `mAP50`**.
     - Ép buộc **`rect=False` (letterbox vuông bất biến theo batch)** trên cả tập `val` trong quá trình huấn luyện (`worker.py`) và tập `test` độc lập (`evaluation.py`), triệt tiêu hoàn toàn độ lệch `mAP50` phụ thuộc kích thước batch (lên tới `1.24% – 3.57%` khi Ultralytics bật `rect=True` mặc định).
     - Bộ kiểm toán `audit` xác minh bắt buộc các bất biến: `primary_metric == protocol["detection_metric"]`, `validation_rect is False` và `best_checkpoint_epoch == best_epoch`.
   - Các chỉ số phụ gồm **`mAP50-95`, `Precision`, `Recall`, `F1-score` và `Best Epoch`** vẫn được ghi nhận đầy đủ tại đúng `best_epoch` của từng seed trong `report.json`, `REPORT.md` và bảng LaTeX để bài báo có bảng phân tích đa chiều trọn vẹn.
   - Tùy chọn `--detection-metric map50_95` vẫn được giữ tường minh nếu muốn chạy thực nghiệm đối chứng giữa hai thước đo. Mọi lượt chạy lịch sử (`0.7` – `0.10.0`) giữ nguyên khả năng `audit` tương thích ngược 100%.

2. **Đánh giá tập Test độc lập trên toàn bộ Training Seeds (`evaluate-v2 --seed-scope all`):**
   - Thay vì chỉ đánh giá 1 seed đầu tiên trên tập Test, `evaluate-v2` nay mặc định đánh giá **toàn bộ các seeds đã khóa** (ví dụ `[17, 29, 43]`) của cấu hình duy nhất đã chọn trong một lần mở test có kiểm soát.
   - Xuất đầy đủ `mean ± SD`, `min`, `max` cho **`mAP50`**, `mAP75`, `mAP50-95`, `Precision`, `Recall`, `F1-score`, bảng **Per-Class AP@50 / AP@50:95** và **Độ lệch tổng quát hóa (`Test - Val`)**.

3. **Lệnh kết xuất học thuật `paper-export` (LaTeX + CSV + Vector SVG + NCKH Summary):**
   - Tự động thực hiện `audit` chứng cứ gốc trước khi xuất sang một thư mục mới (tuyệt đối không chạm vào dữ liệu gốc của lượt chạy).
   - Sinh tự động:
     - `tables.tex`: Các bảng LaTeX chuẩn `\usepackage{booktabs}` (bảng kết quả ứng viên, bảng đa chỉ số Object Detection, bảng độ nhạy trọng số hoặc bảng đối chứng thuật toán / Ablation).
     - `candidates.csv`, `learning_curves.csv`, `acquisition_history.csv`, `weight_sensitivity.csv`, `test_evaluation.csv` (hoặc `benchmark_outcomes.csv`, `method_summary.csv`).
     - `pareto_front.svg`, `learning_curves.svg` (hoặc `benchmark_regret.svg`): Biểu đồ vector sắc nét chèn trực tiếp vào báo cáo.
     - Tính chính xác **3D Dominated Hypervolume** và chẩn đoán nội bộ **Leave-One-Out Cross-Validation (LOOCV)** cho mô hình đại diện Kernel Ridge.

4. **Cải tiến mô hình chi phí liên kiến trúc & Trải nghiệm CLI/Wizard:**
   - `CostModel` sử dụng tỷ số GFLOPs / số tham số thực đo giữa các họ mô hình YOLO (ví dụ suy luận thời gian `yolo11n` từ `yolo11s` sẽ giảm tương ứng thay vì nhân đôi cứng nhắc).
   - `wizard` tự động phát hiện các tệp trọng số `.pt` cục bộ trong thư mục `weights/` cho bài toán Detection.
   - `benchmark` và `algorithm-report` bổ sung chỉ số **Quality (`mAP50`) Regret**, **Exact Optimum Hit Rate** và **Pareto Hit Rate**, đồng thời chặn mọi phép so sánh lệch thước đo.

---

## 2. Quy trình chuẩn 4 bước để chạy thực nghiệm cho bài NCKH

### Bước 1: Kiểm tra môi trường và tập dữ liệu (không huấn luyện)

```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 doctor
.\.venv\Scripts\python.exe -m research_pipeline.v2 inspect --data "D:\datasets\ten_dataset\data.yaml"
```

### Bước 2: Chạy tối ưu hóa cấu hình theo `mAP50` (hoặc chạy Benchmark đối chứng nhiều thuật toán)

**Trường hợp A — Tối ưu hóa cấu hình trên tập dữ liệu thực tế:**
```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 optimize `
  --data "D:\datasets\ten_dataset\data.yaml" `
  --models weights/yolo11n.pt weights/yolo11s.pt `
  --space core `
  --method adaptive_full `
  --full-epochs 20 `
  --seeds 17 29 43 `
  --budget-seconds 3600 `
  --output runs/nckh_main
```

**Trường hợp B — Chạy Benchmark đối chứng thuật toán & Ablation (chuẩn bài báo NCKH):**
```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 benchmark `
  --data "D:\datasets\ten_dataset\data.yaml" `
  --grid configs/detection_v2_controlled.json `
  --methods adaptive_full adaptive_no_cost adaptive_no_uncertainty adaptive_no_diversity random reserve successive_halving default `
  --search-seeds 17 29 43 59 71 `
  --seeds 17 29 43 `
  --full-epochs 20 `
  --budget-seconds 1200 `
  --reference-budget-seconds 10800 `
  --output runs/nckh_benchmark_map50
```
*(Với 5 `search-seeds`, `algorithm-report` sẽ tính thêm khoảng bootstrap phân vị có điều kiện trên từng study).*

### Bước 3: Kiểm toán chứng cứ và Đánh giá tập Test độc lập (sau khi đã khóa lựa chọn)

```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 audit --run "runs/nckh_main/<ten_luot>"
.\.venv\Scripts\python.exe -m research_pipeline.v2 evaluate-v2 --run "runs/nckh_main/<ten_luot>" --seed-scope all
```

### Bước 4: Xuất trọn bộ bảng LaTeX, CSV, biểu đồ SVG và báo cáo tổng hợp NCKH

```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 paper-export `
  --run "runs/nckh_main/<ten_luot>" `
  --output "runs/paper_exports/nckh_main"
```

---

## 3. Kết quả kiểm chứng thực tế trên GPU (0.11.0)

- **Kiểm thử phần mềm & hồi quy:**
  - Runtime GPU (`.venv`, PyTorch 2.6.0+cu124, Ultralytics 8.4.175, RTX 3050 6GB): **341/341 tests đạt**.
  - Runtime CPU (`.venv-cpu`): **340 tests đạt + 1 CUDA-only skip**.
  - Toàn bộ **19/19 tệp `yolo_tuner`** giữ nguyên SHA-256 lịch sử; các lượt chạy `0.9.1` và `0.10.0` đều vượt qua `audit` và giữ nguyên digest tổng hợp (`d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5`).
- **Kiểm chứng thực tế #1 — Bộ dữ liệu đa lớp thực tế African Wildlife (4 lớp: `buffalo`, `elephant`, `rhino`, `zebra`):**
  - Chuẩn bị dữ liệu (`scripts/prepare_african_wildlife_research.py`): Tải trực tiếp kho lưu trữ chính thức `african-wildlife.zip` (`SHA-256: 6b97a768...0040061`), tự động phát hiện và loại bỏ **41 ảnh trùng lặp byte chính xác** (trong đó có 21 ảnh rò rỉ xuyên tập `train`/`val`/`test`), tạo `runs/datasets/african_wildlife_full` (`1036/216/211`) và tập đối chứng phân tầng `runs/datasets/african_wildlife_benchmark` (`160 train / 40 val / 40 test`).
  - Benchmark đối chứng GPU (`runs/african_wildlife_nckh_benchmark/20261118_173050_553132`): 5 cấu hình (`yolo11n`, `yolo11s`, `yolo26n`, `batch=4` vs `8`, `freeze=0` vs `5`) × 2 seeds `[17, 29]` × 4 epochs, đối chứng `adaptive_full`, `random`, `default` với `offline_reference` độc lập (`425.47s`).
    - `adaptive_full` đạt **U Regret thấp nhất (`0.0740`)** và **Quality Regret thấp nhất (`0.2385`)** với thời gian tìm kiếm ngắn nhất (`81.25s / 150.0s`, `0s` overrun), vượt qua cả `default` (`0.0934`, `114.69s`) và `random` (`0.1107`, `146.31s`).
    - **Chứng cứ thực nghiệm cứu checkpoint `best.pt`:** Trên `yolo26n.pt` (`seed=17`), `mAP50` đạt đỉnh tại **Epoch 3 (`0.57819`)** rồi giảm ở Epoch 4 (`0.55823`), trong khi `mAP50-95` lại tăng ở Epoch 4 (`0.39959 > 0.39212`). Nhờ bản vá `Metric.fitness`, cả 3 lượt chạy độc lập đều lưu chính xác `best.pt` ở Epoch 3 (bảo toàn **`+2.00% mAP50`**).
    - **Đối chiếu sai số tuyệt đối:** Kiểm tra độc lập bằng `YOLO(best.pt).val(split="test", rect=False)` khớp với `evaluation.json` tới **sai số `0.000000000000` (`< 10^{-12}`)**; trung bình cộng `AP@50` của 4 lớp bằng chính xác `mAP50` toàn cục tới `< 10^{-12}`.
    - Kết quả Held-out Test đa lớp (`evaluate-v2 --seed-scope all`): Cấu hình Balanced tham chiếu (`18bb4aea7cdf`) đạt **Val `mAP50 = 0.6102 ± 0.0652`**, **Test `mAP50 = 0.6060 ± 0.0304`** (độ lệch tổng quát hóa chỉ **`-0.0042`**); cấu hình `adaptive_full` (`8f073124b1e2`) đạt **Test `mAP50 = 0.6777 ± 0.0059`** (`buffalo: 0.7627`, `elephant: 0.6232`, `rhino: 0.5439`, `zebra: 0.7810`). Cấu hình preset Accuracy (`bc02e5049297`, `yolo11s.pt`) đạt **Val `mAP50 = 0.8403 ± 0.0126`** và tái lập chính xác `Q = 0.8492`, `M = 966.0 MiB` khi chạy `train-v2`.
    - Xuất trọn bộ tài liệu NCKH tại `runs/paper_exports/african_wildlife_reference_v2`, `african_wildlife_adaptive_v2`, `african_wildlife_benchmark_v2` và `african_wildlife_algorithm_v2`.
- **Kiểm chứng thực tế #2 — Penn-Fudan Pedestrian Detection:**
  - Lượt chạy: `runs/nckh_011_map50_verification/20261118_170356_035958` (130 train / 20 val / 20 test, 3 cấu hình × 2 seeds `[17, 29]` × 3 epochs, FP16, `adaptive_full`).
  - Hoàn tất **6/6 full workers** trong `208.06s / 480s` (0 overrun), **Val `mAP50` = `0.9027 ± 0.1036`**, **Test `mAP50` = `0.8508 ± 0.0469`**.
