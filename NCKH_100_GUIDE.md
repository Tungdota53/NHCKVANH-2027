# Hướng dẫn Thực nghiệm Nghiên cứu Khoa học (NCKH) — Phiên bản 1.0.0

> **Tài liệu liên quan:** Tổng quan dự án tại [README.md](README.md) | Tham chiếu CLI đầy đủ tại [CLI_GUIDE.md](CLI_GUIDE.md) | Hồ sơ lịch sử 0.11.0 tại [NCKH_011_GUIDE.md](NCKH_011_GUIDE.md).

Phiên bản **1.0.0** hoàn thiện chuẩn mực nghiên cứu khoa học cho `yolo-budget-tuner / research_pipeline.v2` trên cả hai khía cạnh: **độ tin cậy thuật toán – thống kê** (Giai đoạn 1–3) và **trải nghiệm vận hành – trực quan hóa hai tầng 100% offline** (Giai đoạn 4), đồng thời giữ nguyên 100% tính bất biến của lớp `yolo_tuner/*.py` (19 tệp) và khả năng `audit` các lượt chạy lịch sử `0.7` – `0.11.0`.

---

## 1. Nguyên tắc khoa học, chống rò rỉ dữ liệu và quy tắc kiểm toán (R1)

1. **Chống rò rỉ tập `test` tuyệt đối:**
   - Tập `test` không bao giờ được đọc hoặc sử dụng trong quá trình lựa chọn cấu hình, chọn checkpoint `best.pt`, khớp mô hình đại diện (surrogate), tính trọng số hay tinh chỉnh siêu tham số.
   - Mọi so sánh và chọn cấu hình trong `optimize` / `benchmark` chỉ sử dụng tập `train` và `val`. Tập `test` chỉ được mở sau khi quyết định đã khóa (`decision.locked.json`) qua `evaluate-v2`.
2. **Xác nhận đầy đủ (Full Confirmation Only):**
   - Cấu hình được chọn cuối cùng bắt buộc phải hoàn tất `full-data`, đủ `full-epochs` và đủ tất cả `training_seeds` đã khóa.
   - Kết quả `probe` (1 epoch / subset) hoặc kết quả một phần hạt giống (`adaptive_staged_seeds` seed 1) **tuyệt đối không** đi vào `full_observations()`, `selection.select()`, `report.json` hay bảng xếp hạng cuối.
3. **Trung thực với lượt thất bại / không khả thi:**
   - Các lượt chạy lỗi, OOM, timeout hoặc không chọn được cấu hình khả thi giữ nguyên `regret = null` (không bao giờ gán `0` giả).
4. **Chuẩn mực báo cáo thống kê:**
   - Độ lệch chuẩn (`SD`) qua các `training_seeds` phản ánh biến thiên huấn luyện trên cùng một miền, **không phải** khoảng tin cậy xuyên miền (cross-domain CI).
   - Mọi kiểm định cặp trên `search_seeds` (`n >= 5`) luôn ghi rõ phạm vi có điều kiện trên miền quan sát (`observed-domain conditionality`).
5. **Tái lập tất định (Deterministic Reproducibility) & Tách chữ ký giao diện:**
   - Khớp GP LOOCV, chọn siêu tham số kernel, xếp hạng acquisition và xuất SVG đều tất định 100% (không khởi tạo ngẫu nhiên không kiểm soát), có dấu vết đầy đủ trong `state.json` và `events.jsonl`.
   - Các mô-đun hiển thị thuần túy (`research_pipeline/v2/web_ui/`, `terminal.py`, `wizard.py`, `report.py`, `diagnostics.py`, `paper_export.py`, `cli.py`) được loại trừ khỏi `predictors.training_signature()`, giúp cải tiến giao diện không làm mất hiệu lực thẻ dự báo hay chữ ký huấn luyện.
   - Không có thiết lập cứng (hardcode) cấu hình hay thuật toán thắng cuộc theo tên dataset: mọi mô hình đại diện, dự báo chi phí/VRAM và xếp hạng được suy luận động từ mô tả dữ liệu (`data_features`), kiến trúc (`model_features`) và các phép đo thực nghiệm trực tiếp của chính lượt chạy hiện tại.

---

## 2. Kiến trúc thuật toán và phương pháp luận 1.0.0

### 2.1 Mô hình vật lý VRAM (`observed_cross_memory`) & Chi phí thời gian (`observed_cross_cost`)
- **Dự báo VRAM liên cấu hình (`predictors.py`):**
  - `observed_cross_memory(protocol, fulls, index)` tách peak VRAM thành ba thành phần vật lý:
    1. `base_allocator_mb` (chi phí nền của CUDA/framework),
    2. `static_model_state_mb` (tỷ lệ theo `parameters`, `trainable_parameters` tính đến `freeze`, optimizer `AdamW`/`SGD` và `precision` `fp16`/`bf16`/`fp32`),
    3. `activation_mb` (tỷ lệ theo `batch * imgsz^2 * model_scale / layers`, **bất biến với gradient `accumulation`**).
  - `CostModel.estimate()` ghi nhận `predicted_memory_mb`, `memory_basis`, `memory_components` và `upper_bound_guaranteed=False`.
  - **OOM / Max-Memory Risk Guard (`scheduler.py`):** trước khi cấp phát worker, bộ lập lịch kiểm tra `predicted_memory_mb` so với `objective["max_memory"]` (hoặc `0.98 * vram_free_mb`) và chủ động hoãn (`candidate_memory_risk_deferred`) các cấu hình có nguy cơ vượt trần bộ nhớ.
- **Dự báo thời gian thích nghi (`observed_cross_cost`):**
  - Tách riêng `non_loop_overhead` (khởi tạo worker, nạp dữ liệu, lưu checkpoint) khỏi `loop_seconds` (thời gian vòng lặp huấn luyện/validation theo `imgsz^2`, `GFLOPs`/`params`, `batch`, `freeze`, `precision`).
  - Theo dõi sai số tương đối trực tuyến qua các thử nghiệm hoàn tất để làm mượt biên an toàn cho phần `loop_seconds` (không nhân hệ số biên cố định `1.25` lên phần overhead cố định), giúp cải thiện tỷ lệ tận dụng ngân sách ở cuối phiên mà vẫn giữ vượt ngân sách gần `0`.

### 2.2 Surrogate Heteroscedastic GP LOOCV & Acquisition v2
- **Khớp GP / Adaptive Kernel Ridge tất định (`acquisition.py`, `--acquisition-version 2`):**
  - Tự động chọn kernel (`Matérn-5/2` và `RBF`) cùng `(length_scale, ridge)` trên lưới hữu hạn khóa trước bằng Leave-One-Out Cross-Validation giải tích chính xác qua ma trận mũ (hat matrix $O(n^3)$, phân rã Cholesky, không random restarts).
  - Khi `|training_seeds| >= 2`, phương sai thực nghiệm giữa các seed được đưa vào thành phần nhiễu đường chéo dị phương sai (heteroscedastic diagonal noise).
  - Toàn bộ chẩn đoán (`kernel`, `length_scale`, `ridge`, `loocv_mae`, `loocv_rmse`, `heteroscedastic`, `predictions`) được lưu trong `surrogate_diagnostics` của mỗi bước `acquisition_history`.
- **Hỗ trợ cả 4 quy tắc lựa chọn (`--selection-rule`) & các thuật toán mới:**
  - Gỡ bỏ giới hạn chỉ dùng `weighted` ở `0.11.0`: `adaptive_full`, `adaptive_ehvi` và `adaptive_staged_seeds` hoạt động nhất quán trên cả 4 quy tắc **`weighted`**, **`tchebycheff`**, **`topsis`** và **`knee`** (tính toán kỳ vọng qua cầu phương Gauss-Hermite 21 điểm cho các quy tắc phi tuyến).
  - **`adaptive_ehvi`**: Tính Expected Hypervolume Improvement 3D (`Quality, Time, VRAM`) trên từng giây ngân sách còn lại (`EHVI / cost`).
  - **`adaptive_staged_seeds`**: Sau khi có ứng viên đối chứng hoàn chỉnh, đánh giá **seed 1** trước để cập nhật nhanh chi phí thời gian/VRAM thực tế của ứng viên mới, ưu tiên hoàn tất đủ tất cả các seed cho những ứng viên dẫn đầu và chỉ giải ngân các cấu hình yếu hơn (`partial_deferred`) nếu còn ngân sách.
  - Giữ nguyên toàn bộ các thuật toán đối chứng (`random`, `reserve`, `successive_halving`, `default`, `adaptive_no_*`) và tương thích ngược `--acquisition-version 1`.

### 2.3 Kiểm toán dữ liệu tiền kiểm (`inspect` & `dedup-dataset`), Không gian tìm kiếm & Đánh giá sâu (`evaluate-v2`)
- **Kiểm toán dữ liệu đọc-chỉ (`inspect --data ... --audit-json ...`):**
  - Phát hiện ảnh trùng lặp nội-tập theo SHA-256 (`intra_split_duplicates`) và rò rỉ xuyên tập `train / val / test` theo SHA-256 & `group_id` (`cross_split_leakage`).
  - Với Object Detection, kiểm tra nhãn thiếu (`missing_labels`), nhãn rỗng (`empty_labels`), định dạng lỗi (`malformed_labels`), tọa độ ngoài `[0, 1]` (`out_of_bounds_boxes`), hộp suy biến `w <= 0` hoặc `h <= 0` (`degenerate_boxes`), mã lớp không hợp lệ (`invalid_class_ids >= nc`), phân bố số lượng theo từng lớp (`class_counts`, `imbalance_ratio`) và phân bố kích thước vật thể (`object_size_distribution`: `coco_pixel_area` và `normalized_area` `small/medium/large`).
- **`dedup-dataset --data ... --output ...`:** Tạo bản sao sạch không trùng lặp, không rò rỉ (ưu tiên giữ `train > val > test`), từ chối chạy đè tại chỗ (in-place) hoặc lồng vào thư mục nguồn, kèm báo cáo `dedup_audit.json` và `data_audit.json`.
- **Không gian tìm kiếm có kiểm soát (`--space research` / `core`):**
  - Hỗ trợ đầy đủ siêu tham số `cos_lr`, `warmup_epochs`, `mosaic`, `close_mosaic` (giữ nguyên `config_id` lịch sử khi vắng các khóa này).
  - Preset `core` tự động khám phá `freeze=5` khi cung cấp trọng số `.pt` cục bộ (`YOLO11`, `YOLO26`, `YOLOv8`), trích xuất chính xác `parameters`, `trainable_parameters`, `layers`, `block_parameters` và `gflops_reference_32` mà không tải mạng.
- **Đánh giá Held-out Test chuyên sâu (`evaluate-v2 --seed-scope all`):**
  - Chạy `rect=False` (khớp `< 1e-12` với `YOLO(best.pt).val(split="test", rect=False)`), có warm-up (`--warmup-batches 2`), tùy chọn độ chính xác suy luận (`--inference-precision fp32|fp16`).
  - Xuất đầy đủ `latency` (`preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms_per_image`, `fps`), `pr_curve` (từng lớp & trung bình), `f1_curve` (`best_confidence`, `max_mean_f1`), `confusion_matrix` (`sum_raw`, `mean_raw`, `normalized`), `per_class_summary` và `validation_vs_test` (`generalization_gap_test_minus_val`).

### 2.4 Hội tụ tức thời (Anytime Convergence), Thống kê cặp có điều kiện & `paper-export`
- **Anytime Trajectory (`diagnostics.py`, `benchmark.py`):** Tính quỹ đạo `Best Confirmed Utility` và `Best Confirmed mAP50` theo thời gian thực `t` từ `events.jsonl` + `state.json` (chỉ tính các xác nhận đủ dữ liệu/đủ seed), cùng **`AUARC`** (Area Under Anytime Regret Curve) và **`Time-to-Target`** (`95%` utility tham chiếu).
- **Thống kê phi tham số có điều kiện (`algorithm_study.py`):**
  - Giữ nguyên đầu ra schema-1 byte-for-byte cho `configs/algorithm_010_studies.json` (`n < 5`, `sha256 == d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5`).
  - Khi `paired_seed_count >= 5`, bổ sung **Paired Wilcoxon signed-rank**, **Exact Sign-Flip Permutation Test** và hiệu chỉnh đa kiểm định **Holm-Bonferroni**, kèm tuyên bố phạm vi thống kê minh bạch.
- **Xuất bộ công bố khoa học (`paper-export --lang vi|en`):**
  - Xuất bảng LaTeX `\usepackage{booktabs}` (`tables.tex`), tóm tắt `PAPER_SUMMARY.md`, các tệp CSV UTF-8-BOM, chứng cứ `PAPER_EVIDENCE.json` + `PROVENANCE.json` (SHA-256 toàn bộ đầu vào/đầu ra), và đồ thị vector SVG bảng màu an toàn cho người mù màu Okabe-Ito: `pareto_front.svg`, `learning_curves.svg`, `anytime_convergence.svg`, `per_class_ap50.svg`, `pr_curve.svg`, `confusion_matrix.svg`.

---

## 3. Giao diện hai tầng 1.0.0 (Rich Terminal & Offline Web Dashboard Studio)

- **Tầng 1 — Rich Terminal & Wizard (`terminal.py`, `wizard.py`):**
  - Huy hiệu trạng thái ngữ nghĩa (`EMPIRICAL OPTIMAL`, `BEST OBSERVED`, `AUDIT PASS`, `OOM/TIMEOUT`), bảng ứng viên căn cột làm nổi bật `★` và điểm Pareto, cùng thanh tiến trình `TrialProgress` trực tiếp (`[Config cXX | Seed S | Epoch e/E | Progress Bar | Budget Remaining | Val mAP50 | VRAM MiB | GPU Temp]`).
  - Tự động rơi về UTF-8/ASCII thuần (0 mã ANSI escape) khi `!isatty()`, chuyển hướng log, đặt `NO_COLOR` hoặc truyền `--no-color`.
  - `wizard` tự động quét `runs/datasets/` và `weights/*.pt` (kèm tham số & GFLOPs@640), gợi ý `time_scale`/`memory_scale` theo phần cứng và bổ sung luồng `[5] benchmark` và `[6] ui`.
- **Tầng 2 — Local Web Dashboard Studio (`train_config.py ui`):**
  - Chạy 100% offline trên loopback `127.0.0.1:<port>` (không cần Node.js/npm, hỗ trợ chế độ Sáng/Tối, bảo vệ path-traversal, bắt buộc CSRF token trên mọi `POST`, chỉ đọc chứng cứ).
  - Gồm 6 màn hình chức năng:
    1. **Overview & Hardware Monitor (`/`)**: theo dõi GPU/VRAM/RAM/CPU trực tiếp và lọc các lượt optimize/benchmark/study trong `runs/`.
    2. **Live Experiment Monitor (`/monitor`)**: cập nhật tiến trình trực tiếp, đếm ngược ngân sách, bảng ứng viên, đường cong học, `worker.log` và Safe Stop.
    3. **Visual Run & Benchmark Launcher (`/launcher`)**: quét dataset/weights/grid, chạy `inspect` chống rò rỉ trước khi chạy, xem trước lệnh CLI tương đương và yêu cầu xác nhận tường minh.
    4. **Scientific Analysis & Pareto Explorer (`/analysis`)**: Pareto 3D, Learning Curves, Surrogate Predicted vs Observed, Anytime Convergence, thanh trượt *Phân tích độ nhạy hậu nghiệm — không làm thay đổi quyết định nghiên cứu đã khóa*, và Held-out Test Explorer (Val vs Test gap, Per-Class AP@50, PR Curve, F1-Confidence, Confusion Matrix, Latency/FPS).
    5. **Benchmark & Ablation Studio (`/benchmark`)**: so sánh `U Regret`, `Quality Regret`, `Pareto Hit`, `AUARC`, và kiểm định cặp Wilcoxon/Sign-Flip/Holm-Bonferroni.
    6. **Audit & Paper Export Center (`/export`)**: kiểm toán SHA-256 1 chạm và xuất/tải `tables.tex`, `PAPER_SUMMARY.md`, CSV và SVG.

---

## 4. Quy trình thực nghiệm GPU trên hai tập dữ liệu độc lập (African Wildlife & Penn-Fudan)

> **Ghi chú:** Các lệnh chuẩn bên dưới áp dụng cho thực nghiệm `1.0.0` trên cả hai tập dữ liệu Object Detection sạch trong workspace (`runs/datasets/african_wildlife_benchmark/data.yaml` — 4 lớp động vật hoang dã, và `runs/datasets/penn_fudan_research/data.yaml` — 1 lớp người đi bộ với mật độ hộp/ảnh và phân bố kích thước khác biệt) để kiểm chứng sự thích nghi theo từng dataset và xuất trọn bộ kết quả NCKH.

### Bước 1: Kiểm tra môi trường & tiền kiểm dữ liệu (không rò rỉ, không sửa dữ liệu gốc)
```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 doctor
.\.venv\Scripts\python.exe -m research_pipeline.v2 inspect --data "runs/datasets/african_wildlife_benchmark/data.yaml" --audit-json "runs/audits/african_wildlife_audit.json"
.\.venv\Scripts\python.exe -m research_pipeline.v2 inspect --data "runs/datasets/penn_fudan_research/data.yaml" --audit-json "runs/audits/penn_fudan_audit.json"
```
*(Nếu bộ dữ liệu thô có trùng lặp/rò rỉ, chạy `dedup-dataset --data <nguon> --output <ban_sach>` để tạo bản sao sạch trước khi tối ưu).*

### Bước 2: Chạy Benchmark đa thuật toán trên African Wildlife (đối chứng với `offline_reference`)
```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 benchmark `
  --data "runs/datasets/african_wildlife_benchmark/data.yaml" `
  --grid configs/african_wildlife_nckh_grid.json `
  --methods adaptive_full adaptive_ehvi adaptive_staged_seeds random default `
  --selection-rule weighted `
  --acquisition-version 2 `
  --search-seeds 17 29 `
  --seeds 17 29 `
  --full-epochs 4 `
  --budget-seconds 150 `
  --reference-budget-seconds 600 `
  --output runs/african_wildlife_100_benchmark
```

### Bước 3: Chạy tối ưu hóa thực nghiệm trên tập dữ liệu thứ hai (Penn-Fudan Pedestrian)
```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 optimize `
  --data "runs/datasets/penn_fudan_research/data.yaml" `
  --models weights/yolo11n.pt weights/yolo11s.pt weights/yolo26n.pt `
  --space core `
  --method adaptive_full `
  --selection-rule weighted `
  --acquisition-version 2 `
  --full-epochs 4 `
  --seeds 17 29 `
  --budget-seconds 240 `
  --output runs/penn_fudan_100_optimize
```

### Bước 4: Kiểm toán SHA-256, Đánh giá Held-out Test (`evaluate-v2`), Xuất bài báo (`paper-export`) & Khám phá trên Web UI
```powershell
# Kiểm toán chứng cứ trước khi mở test
.\.venv\Scripts\python.exe -m research_pipeline.v2 audit --run "runs/penn_fudan_100_optimize/<ten_luot>"

# Đánh giá tập Test độc lập trên toàn bộ seeds đã khóa (đo latency, PR/F1 curves, confusion matrix)
.\.venv\Scripts\python.exe -m research_pipeline.v2 evaluate-v2 --run "runs/penn_fudan_100_optimize/<ten_luot>" --seed-scope all

# Xuất bảng booktabs LaTeX, CSV, vector SVGs Okabe-Ito và PROVENANCE.json
.\.venv\Scripts\python.exe -m research_pipeline.v2 paper-export `
  --run "runs/penn_fudan_100_optimize/<ten_luot>" `
  --output "runs/paper_exports/penn_fudan_100" `
  --lang vi

# Mở Web Dashboard Studio 100% offline để trực quan hóa và đối chiếu số liệu
.\.venv\Scripts\python.exe train_config.py ui --host 127.0.0.1 --port 8765 --runs-dir runs
```

---

## 5. Kết quả thực nghiệm GPU trên hai tập dữ liệu độc lập & Kiểm chứng trình duyệt / đóng gói 1.0.0

Toàn bộ thực nghiệm bên dưới được chạy tuần tự trên **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (`device 0`, `$env:YOLO_AUTOINSTALL="false"`, trọng số cục bộ `weights/yolo11n.pt`, `weights/yolo11s.pt`, `weights/yolo26n.pt`, lưới `configs/african_wildlife_nckh_grid.json`, `full_epochs=4`, `seeds=[17, 29]`, `weights=[0.6, 0.25, 0.15]`, `time_scale=60`, `memory_scale=512`, `max_memory=2500`, `acquisition_version=2`).

### 5.1 Tập dữ liệu 1: African Wildlife (`runs/datasets/african_wildlife_benchmark/data.yaml`)
- **Kiểm toán tiền kiểm (`runs/audits/african_wildlife_100_audit.json`):** `240` ảnh (`160/40/40`), `nc=4` (`buffalo`, `elephant`, `rhino`, `zebra`), `445` bounding boxes (`imbalance_ratio=1.9875`), **`0` trùng lặp nội-tập, `0` rò rỉ xuyên tập, `0` lỗi nhãn/bbox**.
- **Benchmark `150s` (`runs/african_wildlife_nckh_benchmark_100/20261125_121712_003198`) so với `0.11.0` (`81.25s/150s`):**
  - `offline_reference` (`319.24s`): cấu hình tối ưu tham chiếu là **`c02` (`18bb4aea7cdf`, `yolo11n.pt`, `320/8`, `freeze=5`)** với `Val mAP50 = 0.61018 ± SD 0.06522`, `Time = 25.76s`, `VRAM = 376.0 MiB`, `U = 0.148632`; mặt trận Pareto `['696caa73e7e1', '18bb4aea7cdf', 'bc02e5049297']`.

| Thuật toán (`150s` budget) | Chọn | Thời gian (`%` ngân sách) | Overrun | `U Regret` | `Quality Regret` | Exact Hit | Pareto Hit | `AUARC` | `TTT-95` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adaptive_full` (v2) | `c02` (`18bb4aea7cdf`) | `123.45s` (**`82.30%`**) | `0.00s` | **`0.00000`** | `0.23011` | `1` | `1` | `0.07466` | `122.45s` |
| `adaptive_staged_seeds` | `c02` (`18bb4aea7cdf`) | `118.84s` (`79.23%`) | `0.00s` | **`0.00000`** | `0.23011` | `1` | `1` | **`0.07184`** | **`117.89s`** |
| `adaptive_ehvi` | `c04` (`8f073124b1e2`) | `74.83s` (`49.89%`) | `0.00s` | `0.07857` | `0.23845` | `0` | `0` | `0.08914` | `null` |
| `random` | `c04` (`8f073124b1e2`) | `146.05s` (`97.36%`) | `0.00s` | `0.07857` | `0.23845` | `0` | `0` | `0.09505` | `null` |
| `default` | `c00` (`696caa73e7e1`) | `85.64s` (`57.09%`) | `0.00s` | `0.09439` | `0.33844` | `0` | `1` | `0.09760` | `null` |

  - So với `0.11.0` (`81.25s/150s = 54.17%`, `regret = 0.07401`), `adaptive_full` v2 tăng mức tận dụng ngân sách lên **`82.30%` (`123.45s/150s`)**, tìm đúng cấu hình tối ưu **`c02` (`regret = 0.00000`)**, **`0s` overrun**, **`0` OOM**.
- **Tối ưu đa cấu hình GP LOOCV (`runs/african_wildlife_optimize_100/20261125_123213_543341`, `240s`):** hoàn tất `3/5` cấu hình (`c04, c02, c00`), chọn **`c02`** (`U = 0.144723`), surrogate **`rbf`** (`length_scale=0.25, ridge=0.5`, online LOOCV RMSE=`0.0044`); `audit` **PASS 100%**.
- **Đánh giá Held-out Test (`evaluate-v2 --seed-scope all`, `rect=False`, `fp32`, `batch=8`):**
  - `Test mAP50 = 0.6060 ± SD 0.0304` (seed 17: `0.584488502773545`, seed 29: `0.627464839514165`), `mAP50-95 = 0.4253 ± SD 0.0294`, `mAP75 = 0.4838`, `F1 = 0.5676`, `generalization_gap = -0.0042`, độ trễ **`4.44 ms/ảnh` (`225.59 FPS`)**, `confusion_matrix sum_raw = 18858.0` (`diag = 45.0`).
  - Đối chiếu trực tiếp `YOLO(best.pt).val(split="test", rect=False)` cho sai khác **`0.000e+00` (`< 1e-12`)** trên cả `map50` và `map50_95` ở cả 2 seeds.
  - Xuất `paper-export` (`en` & `vi`): `runs/paper_exports/african_wildlife_100_benchmark_{en,vi}` và `runs/paper_exports/african_wildlife_100_optimize_{en,vi}`.

### 5.2 Tập dữ liệu 2: Penn-Fudan Pedestrian (`runs/datasets/penn_fudan_research/data.yaml`)
- **Kiểm toán tiền kiểm (`runs/audits/penn_fudan_100_audit.json`):** `170` ảnh (`130/20/20`), `nc=1` (`pedestrian`), `423` bounding boxes (`imbalance_ratio=1.00`), **`0` trùng lặp nội-tập, `0` rò rỉ xuyên tập, `0` lỗi nhãn/bbox**.
- **Benchmark `150s` (`runs/penn_fudan_nckh_benchmark_100/20261125_192630_694472`):**
  - `offline_reference` (`382.53s`): cấu hình tối ưu tham chiếu dịch chuyển sang **`c00` (`696caa73e7e1`, `yolo11n.pt`, `320/4`, `accum=4`, `freeze=0`)** với `Val mAP50 = 0.97103 ± SD 0.00581`, `Time = 39.55s`, `VRAM = 318.0 MiB`, `U = 0.324641`; mặt trận Pareto `['696caa73e7e1', '74930d591668', '18bb4aea7cdf', 'bc02e5049297', '8f073124b1e2']`.

| Thuật toán (`150s` budget) | Chọn | Thời gian (`%` ngân sách) | Overrun | `U Regret` | `Quality Regret` | Exact Hit | Pareto Hit | `AUARC` | `TTT-95` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adaptive_ehvi` | `c02` (`18bb4aea7cdf`) | `119.19s` (`79.46%`) | `0.00s` | `0.01988` | `0.06120` | `0` | `1` | `0.10865` | `null` |
| `adaptive_staged_seeds` | `c02` (`18bb4aea7cdf`) | `141.36s` (`94.24%`) | `0.00s` | `0.01988` | `0.06120` | `0` | `1` | **`0.10637`** | `null` |
| `adaptive_full` (v2) | `c04` (`8f073124b1e2`) | `132.09s` (`88.06%`) | `0.00s` | `0.04737` | `0.01819` | `0` | `1` | `0.17120` | `null` |
| `random` | `c04` (`8f073124b1e2`) | `146.09s` (`97.39%`) | `0.00s` | `0.04737` | `0.01819` | `0` | `1` | `0.10799` | `null` |
| `default` | `c00` (`696caa73e7e1`) | `88.23s` (`58.82%`) | `0.00s` | **`0.00000`** | `0.01731` | `1` | `1` | `0.10973` | **`86.95s`** |

- **Tối ưu độc lập (`runs/penn_fudan_optimize_100/20261125_194326_637373`, `220s`):** hoàn tất `c04` (`8f073124b1e2`, `yolo26n.pt`, `freeze=0`), đạt `Val mAP50 = 0.9702 ± SD 0.0129`, `U = 0.235344` (`109.23s`, `0s` overrun); toàn bộ child & reference `audit` **PASS 100%**.
- **Đánh giá Held-out Test (`evaluate-v2 --seed-scope all`, `rect=False`, `fp32`, `batch=8`):**
  - `Test mAP50 = 0.8906 ± SD 0.0069` (seed 17: `0.8955040954927918`, seed 29: `0.8856882705932964`), `mAP50-95 = 0.6613 ± SD 0.0028`, `mAP75 = 0.8068`, `F1 = 0.8745`, `generalization_gap = -0.0796`, độ trễ **`19.15 ms/ảnh` (`52.74 FPS`)**, `confusion_matrix sum_raw = 9420.0` (`diag = 84.0`).
  - Đối chiếu trực tiếp `YOLO(best.pt).val(split="test", rect=False)` đạt sai khác **`0.000e+00` (`< 1e-12`)** trên cả 2 seeds.
  - Xuất `paper-export` (`en` & `vi`): `runs/paper_exports/penn_fudan_100_benchmark_{en,vi}` và `runs/paper_exports/penn_fudan_100_optimize_{en,vi}`.

### 5.3 Bằng chứng thích nghi xuyên tập dữ liệu (Chống hardcode / thiên vị mẫu)
- **African Wildlife (`nc=4`, `445` boxes, mất cân bằng `1.99x`):** `freeze=5` (`c02`) bảo toàn đặc trưng tổng quát cho 4 lớp và đạt `Val mAP50 = 0.6102` (vượt `c00` `0.5019`), trở thành cấu hình tối ưu tham chiếu **`c02` (`18bb4aea7cdf`)**; trong benchmark `150s`, cả `adaptive_full` và `adaptive_staged_seeds` đạt `regret = 0.00000`.
- **Penn-Fudan (`nc=1`, `423` boxes, cân bằng `1.00x`):** đóng băng backbone (`c02`, `freeze=5`) kìm hãm thích nghi đơn lớp (`Val mAP50 = 0.92715`), trong khi huấn luyện toàn mạng (`c00`, `freeze=0`, `batch=4, accum=4`) đạt `Val mAP50 = 0.97103` ở mức `318.0 MiB` VRAM, làm dịch chuyển cấu hình tối ưu tham chiếu sang **`c00` (`696caa73e7e1`)** (và lượt `optimize` chọn **`c04` `8f073124b1e2`**, khác hẳn `c02` trên African Wildlife).
- Khớp surrogate GP LOOCV, đường cong VRAM/Cost và tập Pareto thích nghi hoàn toàn theo phép đo trực tiếp của từng dataset (khác nhau về mặt trận Pareto, phân hạng cấu hình, độ trễ `225.59 FPS` vs `52.74 FPS`, và cấu hình thắng cuộc).

### 5.4 Kiểm chứng Web UI trên trình duyệt thực, Đóng gói & Bất biến `1.0.0`
- **Trình duyệt thực (`http://127.0.0.1:8765`):** kiểm tra đủ 6 trang (`Overview`, `Live Monitor`, `Launcher`, `Analysis` trên cả African Wildlife `20261125_123213_543341` & Penn-Fudan `20261125_194326_637373` với đầy đủ đồ thị SVG kể cả Confusion Matrix khác 0 và thanh trượt độ nhạy hậu nghiệm không đổi quyết định khóa, `Benchmark` trên cả 2 study, `Export` với `POST /api/audit` PASS), kiểm tra chuyển đổi Dark/Light `#theme-toggle`, **`0` lỗi console/exception/network**, bố cục không tràn ngang.
- **Kiểm thử tự động & Bất biến:** `420/420` unit/integration tests trên `.venv` (CUDA), `420` (`419 + 1 CUDA skip`) trên `.venv-cpu`, `75/75` runs v2 hoàn chỉnh vượt qua `audit_run(check_published=True)` + `make_report()`, `pyproject.toml` `1.0.0` đã làm mới offline trên cả `.venv` và `.venv-cpu`; toàn bộ `19` tệp `yolo_tuner/*.py` (`git status` sạch) và `configs/algorithm_010_studies.json` (`d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5`) giữ nguyên SHA-256 lịch sử 100%.

