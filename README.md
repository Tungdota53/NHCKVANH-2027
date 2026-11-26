# Pipeline tự động chọn cấu hình huấn luyện ảnh (`yolo-budget-tuner` / `research_pipeline.v2`)

## Bản hiện tại: 1.0.0 — Chuẩn mực Nghiên cứu Khoa học, Surrogate GP LOOCV & Giao diện Hai tầng 100% Offline

Phiên bản **1.0.0** nâng cấp toàn diện `research_pipeline.v2` thành nền tảng tối ưu hóa cấu hình huấn luyện đa mục tiêu (`Quality / Time / VRAM`) có kiểm toán mật mã, thống kê phi tham số có điều kiện và giao diện hai tầng **100% offline**, trong khi **giữ bất biến tuyệt đối toàn bộ 19 tệp `yolo_tuner/*.py` và khả năng `audit` các lượt chạy `0.7` – `0.11.0`**.

### Các năng lực cốt lõi trong `1.0.0`

1. **Mô hình vật lý VRAM, Chi phí thời gian thích nghi & Chống OOM (Phase 1):**
   - **Dự báo VRAM liên cấu hình (`observed_cross_memory`):** tách thành `base_allocator_mb`, `static_model_state_mb` (tính `parameters`, `trainable_parameters` theo `freeze`, `AdamW`/`SGD`, `fp16/bf16/fp32`) và `activation_mb` (tỷ lệ theo `batch * imgsz^2 * model_scale / layers`, bất biến với `accumulation`), tích hợp vào `CostModel.estimate()` kèm `OOM / Max-Memory Risk Guard` trong `scheduler.py`.
   - **Dự báo chi phí `observed_cross_cost`:** tách `non_loop_overhead` khỏi `loop_seconds` và làm mượt biên an toàn theo sai số thực nghiệm trực tuyến (không nhân `1.25` lên phần overhead cố định), giảm lãng phí cuối ngân sách mà vẫn giữ vượt ngân sách gần `0`.
   - **Deterministic Heteroscedastic GP / Adaptive Kernel Ridge (`--acquisition-version 2`):** tự động chọn `Matérn-5/2` hoặc `RBF` cùng `(length_scale, ridge)` qua exact analytic hat-matrix LOOCV ($O(n^3)$, không random restarts) và phương sai dị phương sai từ đa seeds (`|S| >= 2`), lưu đầy đủ `surrogate_diagnostics`.
   - **Mở khóa đủ 4 quy tắc lựa chọn & Thuật toán mới:** `adaptive_full`, **`adaptive_ehvi`** (3D Pareto `EHVI / cost`) và **`adaptive_staged_seeds`** (đo seed 1 cập nhật nhanh chi phí/VRAM rồi hoàn tất đủ seeds cho ứng viên dẫn đầu; kết quả 1 seed tuyệt đối không thay full confirmation) hỗ trợ cả **`weighted`**, **`tchebycheff`**, **`topsis`** và **`knee`**.
2. **Kiểm toán dữ liệu tiền kiểm, Không gian kiểm soát & Đánh giá sâu (Phase 2):**
   - **`inspect --data` (`--audit-json`) & `dedup-dataset`:** đọc-chỉ phát hiện trùng lặp SHA-256 nội-tập, rò rỉ xuyên tập `train/val/test`, thiếu/rỗng/lỗi nhãn, bbox ngoài `[0, 1]` hoặc suy biến, mã lớp `>= nc`, mất cân bằng lớp và phân bố kích thước (`coco_pixel_area` & `normalized_area`); `dedup-dataset` xuất bản sao sạch mà không chạm vào dữ liệu gốc.
   - **Không gian tìm kiếm có kiểm soát (`--space research` / `core`):** hỗ trợ `cos_lr`, `warmup_epochs`, `mosaic`, `close_mosaic`, `core` tự động khám phá `freeze=5` khi dùng local `.pt` (`YOLO11`, `YOLO26`, `YOLOv8`), giữ nguyên `config_id()` lịch sử.
   - **Đánh giá Held-out Test `evaluate-v2 --seed-scope all`:** mặc định `rect=False` (khớp `< 1e-12` với `YOLO(best.pt).val(rect=False)`), đo Inference Latency (`preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms_per_image`, `fps`), và xuất dữ liệu JSON cho `pr_curve`, `f1_curve`, `confusion_matrix` (`sum_raw`, `mean_raw`, `normalized`) và `per_class_summary`.
3. **Anytime Convergence, Thống kê cặp & Bộ xuất bài báo `paper-export` (Phase 3):**
   - Tính quỹ đạo hội tụ tức thời **`AUARC`** và **`Time-to-Target` (95%)** từ `events.jsonl` + `state.json` (chỉ tính các xác nhận đủ dữ liệu/đủ seed).
   - `algorithm-report`: giữ nguyên 100% đầu ra schema-1 `0.10.0` (`sha256 == d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5`) khi `n < 5`; khi `n >= 5` bổ sung **Paired Wilcoxon signed-rank**, **Exact Sign-Flip Permutation Test** và hiệu chỉnh **Holm-Bonferroni** kèm phạm vi có điều kiện.
   - **`paper-export --lang vi|en`:** xuất bảng `\usepackage{booktabs}` (`tables.tex`), `PAPER_SUMMARY.md`, CSVs UTF-8-BOM, `PAPER_EVIDENCE.json`, `PROVENANCE.json` và đồ thị vector SVG bảng màu Okabe-Ito (`pareto_front.svg`, `learning_curves.svg`, `anytime_convergence.svg`, `per_class_ap50.svg`, `pr_curve.svg`, `confusion_matrix.svg`).
4. **Giao diện hai tầng 100% Offline (Phase 4):**
   - **Tier 1 — Rich Terminal & Wizard `[1..7]`:** huy hiệu trạng thái (`EMPIRICAL OPTIMAL`, `BEST OBSERVED`, `AUDIT PASS`, `OOM/TIMEOUT`), bảng ứng viên đánh dấu `★`/Pareto, `TrialProgress` trực tiếp, tự động xuất văn bản thuần 0 mã ANSI khi `!isatty()` hoặc `--no-color` / `NO_COLOR`; `wizard` tự quét `runs/datasets/` & `weights/*.pt` và gợi ý `time_scale`/`memory_scale`.
   - **Tier 2 — Local Web Dashboard Studio (`train_config.py ui`):** bind loopback `127.0.0.1:<port>`, 100% offline (không Node/CDN), Dark/Light mode, bảo vệ path-traversal & CSRF `POST`, tách khỏi `training_signature()`, với 6 màn hình: Overview & Hardware Monitor, Live Experiment Monitor (+ Safe Stop), Visual Launcher (+ `inspect`), Scientific Analysis & Pareto Explorer (+ Post-hoc Weight Sensitivity & Held-out Test Explorer), Benchmark & Ablation Studio, và Audit & Paper Export Center.

Tài liệu chi tiết:
- Phương pháp luận & thực nghiệm NCKH `1.0.0`: **[NCKH_100_GUIDE.md](NCKH_100_GUIDE.md)**
- Tham chiếu toàn bộ 20 lệnh CLI & cờ: **[CLI_GUIDE.md](CLI_GUIDE.md)**
- Hồ sơ lịch sử `0.11.0` & `0.10.0`: [NCKH_011_GUIDE.md](NCKH_011_GUIDE.md), [ALGORITHM_010_METHOD.md](ALGORITHM_010_METHOD.md)

---

### Bắt đầu nhanh & Kiểm thử

```powershell
# Kiểm tra môi trường & phiên bản (1.0.0)
.\.venv\Scripts\python.exe train_config.py --version
.\.venv\Scripts\python.exe -m research_pipeline.v2 doctor

# Mở menu tương tác [1..7] hoặc Web Dashboard Studio 100% offline
.\.venv\Scripts\python.exe train_config.py
.\.venv\Scripts\python.exe train_config.py ui --host 127.0.0.1 --port 8765 --runs-dir runs

# Kiểm tra dữ liệu chống rò rỉ (không training), tối ưu, đánh giá test & xuất bài báo
.\.venv\Scripts\python.exe -m research_pipeline.v2 inspect --data "runs/datasets/african_wildlife_benchmark/data.yaml" --audit-json "runs/audit.json"
.\.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\yolo_data\data.yaml" --models weights/yolo11n.pt weights/yolo11s.pt --method adaptive_full --selection-rule weighted --budget-seconds 1200
.\.venv\Scripts\python.exe -m research_pipeline.v2 evaluate-v2 --run "runs/v2/<luot>" --seed-scope all
.\.venv\Scripts\python.exe -m research_pipeline.v2 paper-export --run "runs/v2/<luot>" --output "runs/paper_exports/<luot>" --lang vi
```

Chạy bộ kiểm thử tự động (trên backend CUDA `.venv` và CPU `.venv-cpu`):
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -q
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests_e2e -q
```

### Kết quả kiểm chứng `1.0.0` (Dual-Dataset GPU, Web UI & Đóng gói)

- **Kiểm thử tự động & Đóng gói offline (`1.0.0`):** `420/420` tests (`tests/`) đạt trên `.venv` (CUDA), `419 pass + 1 CUDA skip` trên `.venv-cpu`, và `184/184` opaque-box E2E (`tests_e2e/`) đạt trên cả hai môi trường; toàn bộ `19` tệp `yolo_tuner/*.py` (`85a858aa...`) và `configs/algorithm_010_studies.json` (`d246015b...`) giữ nguyên SHA-256 lịch sử 100%.
- **Thực nghiệm GPU tuần tự trên 2 dataset độc lập (RTX 3050 6GB, `YOLO_AUTOINSTALL=false`):**
  - **African Wildlife (`240` ảnh, `nc=4`):** tại ngân sách `150s` (`runs/african_wildlife_nckh_benchmark_100/20261125_121712_003198`), `adaptive_full` v2 tận dụng **`82.30%`** (`123.45s/150s`, vượt `54.17%` của `0.11.0`) và đạt **`U Regret = 0.00000`** (`c02`, `18bb4aea7cdf`, `freeze=5`), `adaptive_staged_seeds` đạt **`U Regret = 0.00000`** (`118.84s/150s`), `0s` overrun, `0` OOM; ở `240s` GP LOOCV (`rbf`, `loocv_mae=0.00436` / `0.0474` nội bộ) chọn `c02`, `evaluate-v2` `Test mAP50 = 0.6060` (`225.59 FPS`, `confusion_matrix sum_raw = 18858.0`).
  - **Penn-Fudan Pedestrian (`170` ảnh, `nc=1`):** cấu hình tối ưu tham chiếu dịch chuyển sang **`c00` (`696caa73e7e1`, `freeze=0`, `batch=4, accum=4`)** trong `runs/penn_fudan_nckh_benchmark_100/20261125_192630_694472` (`382.53s`), ở `150s` `adaptive_ehvi` và `adaptive_staged_seeds` đạt `U Regret = 0.01988` (`79.46%` và `94.24%` ngân sách), `default` (`c00`) đạt `0.00000`; ở `optimize` `220s` (`20261125_194326_637373`) hệ thống chọn **`c04` (`8f073124b1e2`, `yolo26n.pt`, `freeze=0`)** với `Val mAP50 = 0.9702` và `evaluate-v2` `Test mAP50 = 0.8906` (`52.74 FPS`, `confusion_matrix sum_raw = 9420.0`).
- **Đối chiếu chuẩn & Trình duyệt thực:** `YOLO(best.pt).val(split="test", rect=False)` khớp **`0.000e+00` (`< 1e-12`)** trên toàn bộ seeds ở cả 2 dataset; trình duyệt kiểm tra trực tiếp Web UI `1.0.0` đi qua đủ **6 màn hình**, Dark/Light toggle, CSRF `POST`, `8` đồ thị SVG không `NaN`, ma trận nhầm lẫn hiển thị đầy đủ trên cả 2 dataset, và **`0` lỗi console/network**. Chi tiết xem [NCKH_100_GUIDE.md § 5](NCKH_100_GUIDE.md#5-kết-quả-thực-nghiệm-gpu-trên-hai-tập-dữ-liệu-độc-lập--kiểm-chứng-trình-duyệt--đóng-gói-100).

  - [NV] **Đột phá `1.0.0` (Conformal-MF + Risk-Gated Handoff + `epsilon_constraint`):**
    - Thuật toán lai **`conformal_mf`** loại bỏ trước các cấu hình vi phạm VRAM qua thẻ hiệu chuẩn **`ConformalVRAMGuard`** ([conformal_vram_card_rtx3050.json](configs/conformal_vram_card_rtx3050.json), ngăn **`100%`** WDDM spill trên tập hiệu chuẩn), thăm dò Giai đoạn 1 tối đa `2` cấu hình rồi dùng **GP LOOCV** ở Giai đoạn 2, và tự động kích hoạt **`risk_gated_tpe_handoff`** sang `tpe_bo` khi bài toán không có card/ràng buộc VRAM.
    - Đối chứng ghép cặp **`n = 8`** hạt giống tìm kiếm trên **3 miền độc lập** (`144` lượt online): `conformal_mf` đứng **hạng #1 trên cả 3/3 miền** — **`U Regret = 0.00000` (`100%` trúng tối ưu, `0.00` vi phạm VRAM / `0.00s` lãng phí, lọc `5/12` cấu hình `fp32` nặng)** trên cả `african_wildlife_benchmark` (`p=0.0078`, Holm `0.0312` vs `default`) và `penn_fudan_research` (`p=0.0078`, Holm `0.0312`), đồng thời đạt **`U Regret = 0.00781`** trên CPU `cifar10_research` (`p=0.0156` vs `default`).
    - Quy tắc **`--selection-rule epsilon_constraint`** bảo vệ chất lượng trên `african_wildlife_full` (`1,036` train / `216` val / `211` test), đạt **`Test mAP@50 = 0.9528 ± 0.0004`** ở **`159.05 FPS`**.
    - Giao diện Web UI đồng bộ đầy đủ `conformal_mf`, `mf_no_conformal`, `tpe_bo`, `epsilon_constraint`, chọn card Conformal, `--max-memory`, và 3 cột an toàn VRAM trên Benchmark; bộ xuất bài báo tạo sẵn LaTeX ([tables.tex](runs/paper_exports/nckh_2027_final_3domain_vi/tables.tex)), CSV và đồ thị hiệu chuẩn SVG ([conformal_vram_calibration.svg](runs/nckh_conformal_vram_study/conformal_vram_calibration.svg)).
    - Báo cáo chi tiết: [NCKH_BREAKTHROUGH_REPORT.md](NCKH_BREAKTHROUGH_REPORT.md) & [BAO_CAO_NCKH_2027.md](BAO_CAO_NCKH_2027.md).

### Lấy mã nguồn từ GitHub & Cài đặt

Dùng Python 3.12 trên Windows cho cấu hình cài đặt bên dưới. Sau khi clone, chọn **một** backend:

```powershell
git clone https://github.com/Tungdota53/NHCKVANH-2027.git
cd NHCKVANH-2027
.\setup-v2.ps1 -Backend cpu
# Hoặc GPU NVIDIA tương thích CUDA 12.4: .\setup-v2.ps1 -Backend cu124
.\cli.ps1
```

Repository gồm mã nguồn, tests, cấu hình và tài liệu; **không kèm dataset, weights, môi trường ảo hoặc thư mục `runs/`**. Các đường dẫn `runs/...` trong tài liệu nghiệm thu/manifest là provenance của thực nghiệm cục bộ, không phải artifact có sẵn sau clone. Chuẩn bị dataset riêng trước khi optimize; các cấu hình dùng local weights cần có file `.pt` cục bộ tương ứng (`worker` không tự tải weights qua mạng).

## Nền thuật toán `0.10.0` – `0.11.0` & Kiến trúc tương thích

- **Chuẩn `mAP50` (`0.11.0`+):** nhánh Object Detection mặc định dùng `--detection-metric map50` (`Metric.fitness` đồng bộ với `mAP50`, ép `rect=False` trên `val` và `test`, kiểm toán `best_checkpoint_epoch == best_epoch`), vẫn ghi nhận `mAP50-95`, `mAP75`, `Precision`, `Recall`, `F1` tại đúng `best_epoch` và hỗ trợ `--detection-metric map50_95` khi cần đối chứng.
- **Chứng nhận & Ranh giới khoa học (`0.10.0` / `0.9.1`):** chỉ sử dụng phép đo **full-data / full-horizon / đủ training seeds** để xếp hạng và chọn cấu hình cuối; thất bại giữ `regret = null`; `SD` không phải khoảng tin cậy xuyên miền; budget là soft end-to-end; uncertainty là heuristic phục vụ tìm kiếm; mọi thay đổi source code nằm ngoài mô-đun UI sẽ làm thay đổi `training_signature`.
- **Tài liệu nền tảng & Lịch sử:**
  - `0.11.0`: [NCKH_011_GUIDE.md](NCKH_011_GUIDE.md)
  - `0.10.0`: [ALGORITHM_010_METHOD.md](ALGORITHM_010_METHOD.md), [ALGORITHM_010_PLAN.md](ALGORITHM_010_PLAN.md), [ALGORITHM_010_EXPERIMENTS.md](ALGORITHM_010_EXPERIMENTS.md), [ALGORITHM_010_VALIDATION.md](ALGORITHM_010_VALIDATION.md)
  - `0.9.1` & `0.8.x` / `0.7.x`: [UPGRADE_091_PLAN.md](UPGRADE_091_PLAN.md), [METHOD_091.md](METHOD_091.md), [UPGRADE_091_VALIDATION.md](UPGRADE_091_VALIDATION.md), [PIPELINE_V2.md](PIPELINE_V2.md), [PROTOCOL_V2.md](PROTOCOL_V2.md), [CLI_VALIDATION.md](CLI_VALIDATION.md)
  - Các lệnh legacy (`--data`, `yolo_tuner`, `research`, `show-training`, `legacy-help`) vẫn được định tuyến nguyên vẹn qua `research_pipeline/cli.py` và gói bất biến `yolo_tuner/` (19 tệp giữ nguyên SHA-256).
