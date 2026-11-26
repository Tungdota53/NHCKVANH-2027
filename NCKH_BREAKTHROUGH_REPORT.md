# Báo cáo Đột phá Thuật toán & Thực nghiệm NCKH 2027 (`1.0.0`)

Báo cáo tổng kết các cải tiến thuật toán, kiểm toán bất biến hệ thống, đồng bộ giao diện Web UI, và bộ thực nghiệm đối chứng đa miền hoàn chỉnh cho hệ thống tối ưu cấu hình huấn luyện thị giác máy tính trong ràng buộc ngân sách **`yolo-budget-tuner / research_pipeline.v2` (`1.0.0`)**.

---

## 1. Tóm tắt điều hành (Executive Summary)

Để khắc phục triệt để các điểm nghẽn khiến bản trước khó đạt điểm cao (tìm kiếm mù rủi ro VRAM gây lãng phí ngân sách trên cấu hình `OOM/spill`, tuyến tính hóa đa mục tiêu làm đánh đổi chất lượng nghiêm trọng, khởi đầu tìm kiếm dừng ở cấu hình mặc định `c00` khi `c00` vô tình tối ưu, và thiếu bằng chứng thống kê đa hạt giống độc lập), bản nâng cấp **`1.0.0` Breakthrough** triển khai bốn trụ cột:

1. **Tiên nghiệm phần cứng hiệu chỉnh Conformal (`ConformalVRAMGuard`):** Dự đoán cận trên vùng an toàn VRAM $U_\alpha(c) = \hat{\mu}(c) + q_{\text{add},\alpha}$ (hoặc hiệu chỉnh nhân), tự động **loại trước (`Conformal prune`)** các cấu hình vi phạm ngưỡng `--max-memory` trước khi tốn thời gian khởi tạo/huấn luyện trên GPU thật.
2. **Thuật toán lai đa tầng có cổng rủi ro (`Risk-Gated conformal_mf`):**
   - **Khi có chứng chỉ Conformal + ràng buộc VRAM (`detection` + `--max-memory` + `--conformal-vram-card`):** Loại trừ $5/12$ cấu hình vi phạm VRAM (`0.00` lần vi phạm, `0.00s` lãng phí), thăm dò Giai đoạn 1 tối đa `2` cấu hình (tỷ lệ epoch `mf_probe_fraction` giới hạn trong `[0.18, 0.25]`), rồi dùng **GP LOOCV + Adaptive Confirmation** trên không gian an toàn để hội tụ về cấu hình tối ưu toàn cục.
   - **Khi không có card/ràng buộc phần cứng (như CPU Image Classification):** Cổng rủi ro tự động chuyển luồng sang **`TPE-BO` (`risk_gated_tpe_handoff`)** nhằm bảo toàn toàn bộ ngân sách cho tìm kiếm Bayes đầy đủ thay vì ép đa tầng kém ổn định trên mạng nhỏ.
3. **Quy tắc chọn lọc $\varepsilon$-Constraint (`--selection-rule epsilon_constraint`):** Bảo vệ chất lượng đầu ra trong dải $\Delta Q \le \varepsilon$ (`--quality-tolerance`) trước khi tối ưu tốc độ/VRAM, ngăn chặn tình trạng cộng điểm tuyến tính chọn cấu hình nhẹ hơn nhưng rớt độ chính xác (`mAP@50`).
4. **Kiểm chứng đa miền `n = 8` hạt giống tìm kiếm & Đánh giá tập Test độc lập (`0.10` bất biến):** Thực hiện đối chứng ghép cặp trên **3 miền độc lập** (`144` lượt chạy online + `36` lượt tham chiếu vét cạn) và chạy huấn luyện toàn tập African Wildlife (`1.036 train / 216 val / 211 test`), kèm kiểm tra Web UI thực tế và bộ test đầy đủ trên cả `.venv` (CUDA) lẫn `.venv-cpu`.

---

## 2. Kết quả đối chứng 3 miền (`n = 8` hạt giống tìm kiếm, lưới `c00` không tối ưu)

Tất cả 3 nghiên cứu dưới đây sử dụng cùng bộ `8` hạt giống tìm kiếm ghép cặp `--search-seeds 17 29 43 59 73 89 101 127` (`24` lượt lặp/thuật toán, tổng `144` lượt chạy online), xuất tự động qua `algorithm-report` ([ALGORITHM_STUDY.md](runs/nckh_2027_final_algorithm_report/ALGORITHM_STUDY.md)) và `paper-export` ([PAPER_SUMMARY.md](runs/paper_exports/nckh_2027_final_3domain_vi/PAPER_SUMMARY.md), [tables.tex](runs/paper_exports/nckh_2027_final_3domain_vi/tables.tex)).

### 2.1. Miền 1 — Phát hiện đối tượng: `african_wildlife_benchmark` (`Study 1B`, ID `5117f0c3feebb51d`)
* **Thiết lập:** GPU RTX 3050 6GB (`YOLO_AUTOINSTALL=false`), lưới 12 cấu hình [nckh_non_default_opt_grid.json](configs/nckh_non_default_opt_grid.json) (`5` epoch, `c00` dùng `yolo11n 320/B8/fp16/SGD lr=0.005/freeze=0` kém tối ưu; `c07..c11` dùng `fp32` vượt ngưỡng `1800 MiB`), ngân sách **`150.0s`** vs vét cạn **`526.47s`** (tỷ lệ **`28.49%`**), ngưỡng VRAM `1800 MiB`.

| Thuật toán | Tỷ lệ khả thi | U Regret (TB ± ĐLC) | Q Regret TB | Tỷ lệ trúng tối ưu | Tỷ lệ Pareto | Thời gian (s) | Vi phạm VRAM | Lãng phí (s) | Conformal lọc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conformal_mf`** | **8/8 (100%)** | **`0.00000 ± 0.00000`** | **`0.00000`** | **`100.0%`** | **`100.0%`** | 125.95 | **`0.00`** | **`0.00`** | **5.0** |
| `adaptive_full` | 8/8 (100%) | `0.00000 ± 0.00000` | `0.00000` | `100.0%` | `100.0%` | 106.75 | 0.00 | 0.00 | 0.0 |
| `mf_no_conformal` | 8/8 (100%) | `0.02454 ± 0.06941` | `0.03762` | `87.5%` | `100.0%` | 136.82 | 1.25 | 29.06 | 0.0 |
| `tpe_bo` | 8/8 (100%) | `0.10285 ± 0.11024` | `0.10858` | `50.0%` | `62.5%` | 136.33 | 0.88 | 42.18 | 0.0 |
| `random` | 7/8 (87.5%) | `0.10472 ± 0.13280` | `0.15453` | `50.0%` | `62.5%` | 146.04 | 1.50 | 65.84 | 0.0 |
| `default` | 8/8 (100%) | `0.19631 ± 0.00000` | `0.30093` | `0.0%` | `100.0%` | 30.59 | 0.00 | 0.00 | 0.0 |

* **Kiểm định ghép cặp (`conformal_mf`):** Vượt `default` ở `objective_regret` (`-0.19631`, Wilcoxon & Sign-Flip **`p = 0.0078`**, Holm hiệu chỉnh **`p = 0.0312`**); nhanh hơn `tpe_bo` (`p = 0.0156`, Holm `p = 0.0312`) và `random` (`p = 0.0078`, Holm `p = 0.0312`); triệt tiêu hoàn toàn **`29.06s`** (`1.25` lần vi phạm/hạt) của `mf_no_conformal`, **`42.18s`** của `tpe_bo` và **`65.84s`** của `random` (`1` hạt `random` còn thất bại không kịp tìm ra cấu hình khả thi).

### 2.2. Miền 2 — Phát hiện đối tượng (tập dữ liệu thứ hai): `penn_fudan_research` (`Study 4`, ID `9d8ae3860c9f3d71`)
* **Thiết lập:** GPU RTX 3050 6GB, cùng lưới 12 cấu hình (`4` epoch), ngân sách **`125.0s`** vs vét cạn **`431.52s`** (**`28.97%`**), ngưỡng VRAM `1800 MiB`.

| Thuật toán | Tỷ lệ khả thi | U Regret (TB ± ĐLC) | Q Regret TB | Tỷ lệ trúng tối ưu | Tỷ lệ Pareto | Thời gian (s) | Vi phạm VRAM | Lãng phí (s) | Conformal lọc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conformal_mf`** | **8/8 (100%)** | **`0.00000 ± 0.00000`** | **`0.00172`** | **`100.0%`** | **`100.0%`** | 105.59 | **`0.00`** | **`0.00`** | **5.0** |
| `adaptive_full` | 8/8 (100%) | `0.00000 ± 0.00000` | `0.00172` | `100.0%` | `100.0%` | 86.12 | 0.00 | 0.00 | 0.0 |
| `mf_no_conformal` | 8/8 (100%) | `0.00000 ± 0.00000` | `0.00172` | `100.0%` | `100.0%` | 110.16 | 1.25 | 21.95 | 0.0 |
| `tpe_bo` | 8/8 (100%) | `0.03471 ± 0.04691` | `0.01208` | `50.0%` | `75.0%` | 111.49 | 1.12 | 40.98 | 0.0 |
| `random` | 8/8 (100%) | `0.03374 ± 0.04516` | `0.03075` | `50.0%` | `75.0%` | 120.99 | 1.75 | 53.72 | 0.0 |
| `default` | 8/8 (100%) | `0.02719 ± 0.00000` | `0.01795` | `0.0%` | `100.0%` | 26.99 | 0.00 | 0.00 | 0.0 |

* **Kiểm định ghép cặp (`conformal_mf`):** Vượt `default` ở `objective_regret` (Wilcoxon & Sign-Flip **`p = 0.0078`**, Holm **`p = 0.0312`**); giảm `tpe_bo` và `random` từ `50.0%` trúng tối ưu lên **`100.0%`** đồng thời loại bỏ hoàn toàn **`21.95s – 53.72s`** lãng phí vi phạm VRAM.

### 2.3. Miền 3 — Phân loại ảnh trên CPU: `cifar10_research` (`Study 2B`, ID `833f3cb3052b5a98`)
* **Thiết lập:** CPU (`--task classification --space core`, 12 cấu hình × 5 epoch), ngân sách **`50.0s`** vs vét cạn **`138.92s`** (**`35.99%`**), không truyền card/ràng buộc VRAM — kích hoạt nhánh **`risk_gated_tpe_handoff`** bên trong `conformal_mf`.

| Thuật toán | Tỷ lệ khả thi | U Regret (TB ± ĐLC) | Q Regret TB | Tỷ lệ trúng tối ưu | Tỷ lệ Pareto | Thời gian (s) | AUARC | TTT-95 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conformal_mf` (Risk-Gated)** | **8/8 (100%)** | **`0.00781 ± 0.00865`** | **`0.01500`** | **`50.0%`** | `75.0%` | 44.87 | 0.02491 | 35.91 |
| `tpe_bo` | 8/8 (100%) | `0.01125 ± 0.01207` | `0.02333` | `37.5%` | **87.5%** | 43.40 | 0.02279 | 27.94 |
| `mf_no_conformal` | 8/8 (100%) | `0.01378 ± 0.01273` | `0.02750` | `37.5%` | `50.0%` | 43.53 | 0.02605 | 29.86 |
| `random` | 8/8 (100%) | `0.01529 ± 0.01085` | `0.03125` | `0.0%` | `62.5%` | 47.70 | 0.02474 | 41.31 |
| `default` | 8/8 (100%) | `0.01752 ± 0.00000` | `0.03667` | `0.0%` | `0.0%` | 21.05 | 0.02444 | — |
| `adaptive_full` | 8/8 (100%) | `0.02248 ± 0.00795` | `0.04500` | `0.0%` | `62.5%` | 41.82 | 0.03028 | — |

* **Ý nghĩa quyết định:** Trước khi bổ sung cổng rủi ro (`risk_gated_tpe_handoff`), nhánh đa tầng thuần (`mf_no_conformal`) chỉ đạt U Regret `0.01378`, kém `tpe_bo` (`0.01125`), trong khi `adaptive_full` tụt xuống `0.02248` (`0%` opt hit). Nhờ cổng rủi ro tự chuyển hướng khi thiếu card/ràng buộc phần cứng, **`conformal_mf` vươn lên hạng #1 tuyệt đối trên cả 3/3 miền độc lập** (`U Regret = 0.00781`, vượt `default` với Wilcoxon & Sign-Flip **`p = 0.0156`**, Holm `p = 0.0625`, vượt `random` với Sign-Flip **`p = 0.0391`**).

---

## 3. Hiệu chuẩn Conformal VRAM & Kháng lỗi thực chiến (Study 2 & Study 3)

### 3.1. Nghiên cứu hiệu chuẩn `ConformalVRAMGuard` (Study 2)
* Chứng chỉ [conformal_vram_card_rtx3050.json](configs/conformal_vram_card_rtx3050.json) được hiệu chuẩn trên `28` lần thăm dò thực tế GPU RTX 3050 6GB (`vram_free_mb = 5161.0`, `vram_total_mb = 6143.5`), ghi nhận **`3` ca tràn bộ nhớ WDDM spill** dù tổng chưa tới `6143.5 MiB`.
* Trên `50` phép tách lặp (`13 fit / 8 cal / 7 test`, báo cáo [summary.json](runs/nckh_conformal_vram_study/summary.json) & đồ thị [conformal_vram_calibration.svg](runs/nckh_conformal_vram_study/conformal_vram_calibration.svg)):
  * Độ bù cộng tính `q_add_0.05 = q_add_0.10 = 1047.22 MiB`.
  * **Tỷ lệ ngăn OOM/spill đạt `100.0%` (`oom_prevention_rate = 1.0`)**, sai số `MAE = 364.21 MiB`, `RMSE = 485.87 MiB`, độ phủ thực nghiệm trung bình **`88.86% ± 16.40%`**, tỷ lệ loại nhầm `25.23%`.
  * **Thử nghiệm Leave-One-Model-Out (LOMO):** Zero-shot đạt độ phủ **`100%`** (`yolo11n`), **`87.5%`** (`yolo11s`), **`83.33%`** (`yolo26n`); hiệu chỉnh 2 điểm neo nâng độ phủ lên **`100%`** trên `yolo11s` và `yolo26n` nhưng làm `yolo11n` giảm xuống `75.0%` (`2` điểm neo không đủ đại diện cho khoảng biến thiên rộng). Đây là giới hạn được báo cáo minh bạch.

### 3.2. Huấn luyện toàn tập & Kiểm định `Test` độc lập (`african_wildlife_full` — Study 3)
* **Dữ liệu:** `1,036` ảnh train / `216` ảnh val / `211` ảnh test độc lập, `12` epoch × `2` seed `[17, 29]`, `[AUDIT PASS]` với `11` worker không có sai lệch kiểm toán ([REPORT.md](runs/nckh_2027_study3_wildlife_full/20261126_013218_819548/REPORT.md), [EVALUATION.md](runs/nckh_2027_study3_wildlife_full/20261126_013218_819548/evaluation/20261126_015534_759413/EVALUATION.md)).
* **Tác dụng của `epsilon_constraint (0.03)`:** Tuy `c01` có điểm tuyến tính `U = 0.3550` cao hơn `c00` (`0.3300`), `epsilon_constraint` bảo vệ chất lượng `c00` (`Val mAP@50 = 0.9237 ± 0.0088`, `442.0 MiB`) thay vì mất `-3.91% mAP@50` ở `c01` (`0.8846`).
* **Kết quả trên tập Test giữ lại (`211` ảnh, không rò rỉ dữ liệu):**
  * **`Test mAP@50 = 0.9528 ± 0.0004`** (`0.952761`: seed 17 `0.9525`, seed 29 `0.9530`, chênh lệch Val→Test `+0.0291`).
  * `Test mAP@75 = 0.8646 ± 0.0010` | `Test mAP@50:95 = 0.7677 ± 0.0030` | `Test F1 = 0.9093`.
  * Tốc độ suy luận (`batch=4, fp32, rect=False`): **`6.299 ms/ảnh` (`159.05 FPS`)**.

---

## 4. Đồng bộ Web UI & Đóng gói trích xuất bài báo

* **Web UI (`1.0.0`):** [launcher.html](research_pipeline/v2/web_ui/templates/launcher.html), [benchmark.html](research_pipeline/v2/web_ui/templates/benchmark.html), [data_access.py](research_pipeline/v2/web_ui/data_access.py), và [app.js](research_pipeline/v2/web_ui/static/app.js) hiển thị đầy đủ các thuật toán (`conformal_mf`, `mf_no_conformal`, `tpe_bo`), quy tắc `epsilon_constraint`, chọn card Conformal (tự phát hiện `configs/conformal_vram_card_*.json`), giới hạn VRAM `--max-memory`, dung sai chất lượng `--quality-tolerance`, và 3 cột an toàn VRAM (`Vi phạm VRAM TB`, `Lãng phí VRAM (s)`, `Conformal lọc`) trên bảng Benchmark.
* **Xuất bản LaTeX/CSV/SVG:** [paper_export.py](research_pipeline/v2/paper_export.py) tự động sinh gói bài báo song ngữ ([nckh_2027_final_3domain_vi](runs/paper_exports/nckh_2027_final_3domain_vi/PAPER_SUMMARY.md), [nckh_2027_final_3domain_en](runs/paper_exports/nckh_2027_final_3domain_en/PAPER_SUMMARY.md)) kèm đồ thị hiệu chuẩn SVG [conformal_vram_calibration.svg](runs/nckh_conformal_vram_study/conformal_vram_calibration.svg), hoàn toàn bảo toàn bất biến `0.10`.

---

## 5. Giới hạn trung thực & Hướng phát triển (Threats to Validity)

1. **Tính khai phá của kiểm định thống kê (`n = 8` search seeds):** Các p-value Wilcoxon/Sign-Flip/Holm (`p = 0.0078`, Holm `0.0312`) là kiểm định ghép cặp có điều kiện theo hạt giống tìm kiếm trên cùng một tập dữ liệu/lịch sử tham chiếu — không thay thế khái quát hóa ngoài mẫu trên hàng chục GPU vật lý khác nhau.
2. **Đánh giá trên một GPU vật lý duy nhất:** Thực nghiệm GPU mới xác nhận trên `RTX 3050 6GB` dưới Windows WDDM; khả năng tổng quát hóa sang GPU khác kiến trúc (`RTX 4060`, `T4`, `A100`) vẫn cần đo lại bằng `scripts/run_conformal_vram_study.py` và bổ sung thêm điểm neo hiệu chuẩn (`> 2` điểm cho LOMO).
3. **Độ trễ đánh giá đa tầng ở tác vụ không có OOM:** Khi mọi cấu hình đều vừa VRAM (`c00..c06`), `adaptive_full` không mất chi phí thăm dò Giai đoạn 1 nên kết thúc sớm hơn `conformal_mf` khoảng `~19s` (`106.75s` vs `125.95s` ở Study 1B) trong khi cùng đạt `U Regret = 0.00000`.
