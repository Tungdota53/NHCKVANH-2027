# CLI 1.0.0 — Hướng dẫn dòng lệnh, Wizard `[1..7]` và Web Dashboard Studio

Phiên bản **1.0.0** cung cấp **20 lệnh v2 chính thức** (`menu`, `wizard`, `runs`, `inspect`, `dedup-dataset`, `doctor`, `plan-search`, `optimize`, `status`, `audit`, `resume`, `benchmark`, `benchmark-report`, `algorithm-report`, `paper-export`, `calibrate`, `train-v2`, `evaluate-v2`, `ui`, `legacy-help`) và vẫn giữ nguyên routing tương thích cho các lệnh `LEGACY_COMMANDS`.

Mặc định cho Object Detection là **`--detection-metric map50`** (lưu checkpoint `best.pt`, khớp surrogate, tính hàm mục tiêu $U(c)$ và chọn cấu hình theo **best validation mAP50**; đồng thời ghi nhận `mAP50-95`, `mAP75`, `Precision`, `Recall`, `F1` tại đúng `best_epoch`). Xem [NCKH_100_GUIDE.md](NCKH_100_GUIDE.md) cho phương pháp luận khoa học và [README.md](README.md) cho tổng quan dự án.

```powershell
.\.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\yolo_data\data.yaml" --models weights/yolo11n.pt weights/yolo11s.pt --method adaptive_full --selection-rule weighted
.\.venv\Scripts\python.exe -m research_pipeline.v2 evaluate-v2 --run "runs/v2/<luot>" --seed-scope all --inference-precision fp32 --warmup-batches 2
.\.venv\Scripts\python.exe -m research_pipeline.v2 paper-export --run "runs/v2/<luot>" --output "runs/paper_exports/<luot>" --lang vi
.\.venv\Scripts\python.exe -m research_pipeline.v2 ui --host 127.0.0.1 --port 8765 --runs-dir runs
```

---

## Cờ toàn cục (Global Flags) & Bảng 20 lệnh v2

### Cờ toàn cục (`train_config.py` / `python -m research_pipeline.v2`)
- `--version`: in `train-config 1.0.0` và thoát.
- `--debug`: giữ nguyên traceback đầy đủ khi chẩn đoán lỗi.
- `--no-color`: tắt mã màu ANSI trên terminal (tự động kích hoạt khi `NO_COLOR` được đặt, `TERM=dumb`, hoặc stdout/stderr không phải TTY).

### Toàn bộ 20 lệnh v2 đã đăng ký (`COMMANDS`)
| Lệnh | Mục đích | Cờ riêng nổi bật |
|---|---|---|
| `menu` | Menu tương tác Rich Terminal `[1..7]` + `[0]` | `--root` (mặc định `runs`) |
| `wizard` | Hướng dẫn từng bước, tự quét `runs/datasets/` & `weights/*.pt`, gợi ý scale | — |
| `runs` | Liệt kê lượt chạy đã lưu (không training) | `--root`, `--limit` (20), `--json` |
| `inspect` | Tiền kiểm dataset (trùng lặp SHA-256, rò rỉ xuyên tập, nhãn/bbox) & phần cứng | `--data`, `--task`, `--device`, `--audit-json`, `--json` |
| `dedup-dataset` | Tạo bản sao dataset sạch không trùng lặp/rò rỉ (không sửa dữ liệu gốc) | `--data`, `--output`, `--task`, `--json` |
| `doctor` | Kiểm tra runtime/backend/dependencies (`exit 2` nếu chưa sẵn sàng) | `--device`, `--json` |
| `plan-search` | Lưu `protocol.json` và khóa phạm vi (không training) | `add_common`, `--mode {online,offline_reference}`, `--reference-boundary` |
| `optimize` | Tối ưu cấu hình bằng full training & budget dùng chung | `add_common`, `--protocol`, `--mode`, `--reference-boundary` |
| `benchmark` | Đối chứng nhiều thuật toán + `offline_reference` | `add_common`, `--methods`, `--search-seeds`, `--reference-budget-seconds` (3600) |
| `benchmark-report` | Audit và tổng hợp lại benchmark study (gồm Anytime `AUARC` & `Time-to-Target 95%`) | `--run`, `--reference` |
| `algorithm-report` | Tổng hợp nhiều benchmark studies (`n >= 5` kích hoạt Wilcoxon, Sign-Flip & Holm-Bonferroni) | `--manifest`, `--output` |
| `paper-export` | Audit gốc & xuất `tables.tex`, `PAPER_SUMMARY.md`, CSVs, Okabe-Ito SVGs, `PROVENANCE.json` | `--run`, `--output`, `--lang {vi,en}` (mặc định `vi`) |
| `status` | Xem tiến độ, trạng thái và kết quả đã lưu | `--run`, `--json` |
| `audit` | Kiểm tra toàn vẹn SHA-256 request/result/checkpoint/ledger và tính lại lựa chọn | `--run` |
| `resume` | Tiếp tục search với ngân sách bổ sung, giữ nguyên protocol | `--run`, `--additional-budget-seconds` |
| `calibrate` | Khớp thẻ dự báo T/M từ phân hoạch dữ liệu độc lập | `--manifest`, `--output` |
| `train-v2` | Huấn luyện triển khai từ `selected.json` (`exit 3` nếu chưa `execution_verified`) | `--selection`, `--data`, `--device`, `--epochs`, `--budget-seconds` (7200), `--output`, `--preset {Balanced,Accuracy,Time,Memory}` |
| `evaluate-v2` | Đánh giá tập Test độc lập (`rect=False`, latency/FPS, PR/F1 curves, confusion matrix) | `--run`, `--timeout-seconds` (1800), `--seed-scope {all,first}`, `--inference-precision {fp32,fp16}`, `--warmup-batches` (2), `--allow-retest` |
| `ui` | Khởi chạy Web Dashboard Studio 100% offline trên loopback | `--host` (`127.0.0.1`/`localhost`/`::1`), `--port` (8765), `--runs-dir` (`runs`), `--open`, `--no-browser` |
| `legacy-help` | Trợ giúp toàn bộ `LEGACY_COMMANDS` | — |

---

## Thuật toán, Surrogate GP LOOCV, Acquisition v2 & Không gian tìm kiếm

- **Các thuật toán (`--method`):**
  - Thích nghi v2: `adaptive_full` (mặc định), `adaptive_ehvi` (tối đa hóa 3D Expected Hypervolume Improvement trên giây ngân sách), `adaptive_staged_seeds` (đo seed 1 cập nhật nhanh chi phí/VRAM rồi ưu tiên hoàn tất đủ tất cả seeds cho ứng viên dẫn đầu; kết quả 1 seed tuyệt đối không thay thế full confirmation), cùng 3 ablation `adaptive_no_cost`, `adaptive_no_uncertainty`, `adaptive_no_diversity`.
  - Đối chứng cổ điển: `reserve`, `random`, `successive_halving`, `default`, `profile_all`, `random_proxy`, `quality_sh`, `utility_sh`, `no_reserve`, `prior_full`.
- **Surrogate & Acquisition (`--acquisition-version`):**
  - `--acquisition-version 2` (**mặc định trong 1.0.0**): khớp Deterministic Heteroscedastic GP / Adaptive Kernel Ridge (`Matérn-5/2` & `RBF`) qua Leave-One-Out Cross-Validation giải tích chính xác trên lưới `(length_scale, ridge)` khóa trước + phương sai dị phương sai từ đa hạt giống (`|S| >= 2`); tính `EI / cost` (cho `weighted`), cầu phương Gauss-Hermite 21 điểm nhất quán theo từng rule, và `EHVI / cost`. Mỗi bước ghi đầy đủ `surrogate_diagnostics` trong `acquisition_history`.
  - `--acquisition-version 1`: giữ nguyên hành vi isotropic kernel ridge lịch sử (`--acquisition-ridge 0.05`, `--acquisition-length-scale 1.0`, `--acquisition-exploration 0.25`, `--acquisition-novelty 0.05`).
- **4 quy tắc lựa chọn (`--selection-rule`):**
  - Cả **`weighted`**, **`tchebycheff`**, **`topsis`** và **`knee`** nay được mở khóa đầy đủ cho **`adaptive_full`**, **`adaptive_ehvi`** và **`adaptive_staged_seeds`** (cũng như `reserve`/`random`/`successive_halving`).
  - `knee` tự động fallback có khai báo sang `tchebycheff` khi mặt Pareto ít hơn 3 cực trị hoặc suy biến.
- **Dự báo VRAM/chi phí vật lý & OOM Risk Guard:**
  - `observed_cross_memory` tách `base_allocator_mb`, `static_model_state_mb` (tính `freeze`, `AdamW`/`SGD`, `fp16/bf16/fp32`) và `activation_mb` (tỷ lệ theo `batch * imgsz^2 * model_scale / layers`, bất biến với `accumulation`), kết hợp OOM Risk Guard trong scheduler để hoãn cấu hình vượt `max_memory`.
  - `observed_cross_cost` tách `non_loop_overhead` khỏi `loop_seconds` và làm mượt biên an toàn theo sai số thực nghiệm trực tuyến (không nhân `cost_margin` lên phần overhead cố định).
- **Không gian cấu hình (`--space`):**
  - `core` (mặc định, 12 ứng viên; tự động khám phá `freeze=5` khi dùng local `.pt`), `compact` (3 ứng viên LR), `research` (tổ hợp có kiểm soát gồm `cos_lr`, `warmup_epochs`, `mosaic`, `close_mosaic` và `freeze=0/5` cho `.pt`), hoặc `--grid <file.json>`.
- **Các cờ `add_common` khác trên `plan-search`, `optimize`, `benchmark`:**
  - `--task {auto,detection,classification,captioning}`, `--device`, `--output`, `--full-epochs` (20), `--seeds` (`17 29 43`), `--budget-seconds` (1200), `--models`, `--detection-metric {map50,map50_95}`, `--precision {fp32,fp16,bf16}`, `--probe-epochs` (1), `--split-seed` (101), `--search-seed` (17), `--trial-timeout` (1800), `--probe-fraction` (0.1), `--cost-margin` (1.25), `--cleanup-seconds` (5), `--train-subset` (64), `--val-subset` (32), `--weights A T M` (`0.6 0.25 0.15`), `--time-scale` (600), `--memory-scale` (1024), `--max-memory`, `--max-time`, `--min-quality` (0), `--predictor`, `--rungs`, `--workers {auto,0..8}`, `--image-cache-mb` (64.0), `--pin-memory {auto,on,off}`.

---

## Tiền kiểm dữ liệu, Đánh giá sâu & Xuất công bố NCKH

### Kiểm toán dữ liệu & Tạo bản sao sạch (không sửa dữ liệu gốc)
```powershell
.\.venv\Scripts\python.exe train_config.py inspect --data "D:\datasets\my_dataset\data.yaml" --audit-json "runs/audits/my_audit.json" --json
.\.venv\Scripts\python.exe train_config.py dedup-dataset --data "D:\datasets\my_dataset\data.yaml" --output "D:\datasets\my_dataset_clean"
```
- `inspect` kiểm tra trùng lặp SHA-256 trong từng split, rò rỉ xuyên tập `train/val/test` (theo SHA-256 & `group_id`), nhãn thiếu/rỗng/lỗi, bbox ngoài `[0, 1]`, bbox suy biến (`w <= 0` / `h <= 0`), `class_id >= nc`, tỷ lệ mất cân bằng lớp và phân bố kích thước vật thể (`coco_pixel_area` & `normalized_area`).
- `dedup-dataset` từ chối ghi đè tại chỗ hoặc vào thư mục con của nguồn, giữ ưu tiên `train > val > test` và lưu `dedup_audit.json` + `data_audit.json`.

### Đánh giá Test độc lập (`evaluate-v2`) & Xuất bài báo (`paper-export`)
```powershell
.\.venv\Scripts\python.exe train_config.py evaluate-v2 --run "runs/v2/<luot>" --seed-scope all --inference-precision fp32 --warmup-batches 2
.\.venv\Scripts\python.exe train_config.py paper-export --run "runs/v2/<luot>" --output "runs/paper_exports/<luot>" --lang en
```
- `evaluate-v2` giữ `rect=False` (khớp `< 1e-12` với `YOLO(best.pt).val(rect=False)`), đo latency (`preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms_per_image`, `fps`), và xuất dữ liệu JSON cho `pr_curve`, `f1_curve`, `confusion_matrix` (`sum_raw`, `mean_raw`, `normalized`).
- `paper-export` hỗ trợ `--lang vi|en`, tự động audit chứng cứ gốc trước khi xuất `tables.tex` (`\usepackage{booktabs}`), `PAPER_SUMMARY.md`, CSVs, `PAPER_EVIDENCE.json`, `PROVENANCE.json` và các biểu đồ SVG bảng màu Okabe-Ito (`pareto_front.svg`, `learning_curves.svg`, `anytime_convergence.svg`, `per_class_ap50.svg`, `pr_curve.svg`, `confusion_matrix.svg`).

### Tổng hợp đa study (`algorithm-report`)
`algorithm-report --manifest studies.json --output <dir>` yêu cầu mỗi study có reference full đã audit; giữ nguyên đầu ra schema-1 `0.10.0` khi `n < 5`, và khi `paired_seed_count >= 5` tính thêm **Paired Wilcoxon signed-rank**, **Exact Sign-Flip Permutation Test** và hiệu chỉnh **Holm-Bonferroni** kèm tuyên bố điều kiện miền quan sát (SD không phải cross-domain CI).

---

## Giao diện hai tầng: Menu Wizard `[1..7]` & Web Dashboard Studio (`ui`)

### Tier 1 — Rich Terminal & Menu `[1..7]`
Chạy `.\.venv\Scripts\python.exe train_config.py` (hoặc `.\cli.ps1`) trong terminal tương tác để mở menu:
- **`[1]`** Tìm cấu hình mới — hướng dẫn từng bước (`optimize`, tự quét `runs/datasets/` & `weights/*.pt` kèm params/GFLOPs@640, tự gợi ý `time_scale`/`memory_scale`)
- **`[2]`** Kiểm tra dataset và phần cứng (`inspect` — không training)
- **`[3]`** Kiểm tra môi trường (`doctor`)
- **`[4]`** Xem / quản lý các lượt chạy đã lưu (Xem, Audit, Bắt đầu/Resume, Train lại, Test, `paper-export`)
- **`[5]`** Đối chứng nhiều thuật toán & reference (`benchmark`)
- **`[6]`** Mở Web Dashboard Studio offline (`ui`)
- **`[7]`** Trợ giúp lệnh nâng cao (`--help`)
- **`[0]`** Thoát

Trong terminal tương tác, bảng ứng viên nổi bật `★` / Pareto, các huy hiệu ngữ nghĩa (`EMPIRICAL OPTIMAL`, `BEST OBSERVED`, `AUDIT PASS`, `OOM/TIMEOUT`) và `TrialProgress` theo epoch/VRAM/GPU Temp được hiển thị rõ ràng; khi chuyển hướng log, không có TTY hoặc bật `--no-color` / `NO_COLOR`, hệ thống xuất văn bản UTF-8/ASCII thuần không chứa bất kỳ mã ANSI nào.

### Tier 2 — Offline Web Dashboard Studio (`train_config.py ui`)
```powershell
.\.venv\Scripts\python.exe train_config.py ui --host 127.0.0.1 --port 8765 --runs-dir runs --open
```
- Chỉ bind địa chỉ loopback (`127.0.0.1`, `localhost`, `::1`), 100% offline (không CDN/Node.js), có bảo vệ path-traversal, kiểm tra CSRF token trên mọi route `POST`, chế độ Dark/Light và không làm thay đổi tệp chứng cứ.
- Cung cấp 6 màn hình: Overview & Hardware Monitor (`/`), Live Experiment Monitor (`/monitor` với Safe Stop), Visual Launcher (`/launcher` tích hợp `inspect`), Scientific Analysis & Pareto Explorer (`/analysis` gồm Post-hoc Weight Sensitivity & Held-out Test Explorer), Benchmark & Ablation Studio (`/benchmark`), và Audit & Paper Export Center (`/export`).

---

## Quản lý khi đang chạy, Exit codes & Tương thích lịch sử

- **Theo dõi chạy nền:** `train_config.py status --run "runs/v2/<luot>" [--json]` kiểm tra PID và thời điểm tạo process; dùng `audit --run ...` để xác minh toàn vẹn SHA-256.
- **DataLoader & Bộ nhớ:** `--workers` (`auto` hoặc `0..8`), `--image-cache-mb` (`64.0` MiB mặc định, chặn ở 10% RAM khả dụng; `0` để tắt; detection không dùng RAM cache YOLO không giới hạn), `--pin-memory {auto,on,off}`.
- **Exit codes:** `0` thành công / thoát menu / hủy trước khi chạy; `2` tham số, dữ liệu, môi trường hoặc chứng cứ không hợp lệ (kể cả `doctor` khi chưa sẵn sàng); `3` `train-v2` chưa đạt `execution_verified`; `130` `Ctrl+C`.
- **Tương thích protocol & legacy:** Không sửa bất kỳ tệp nào trong `yolo_tuner/*.py` hay dữ liệu lịch sử `runs/`. Protocol cũ bị khóa bởi chữ ký nguồn (`training_signature`) sẽ không bị ép resume bằng mã mới; các mô-đun giao diện thuần túy (`web_ui/`, `terminal.py`, `wizard.py`, `report.py`, `diagnostics.py`, `paper_export.py`, `cli.py`) không làm thay đổi `training_signature`.
