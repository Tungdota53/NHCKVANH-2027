# Báo cáo Kết quả Nâng cấp & Thực nghiệm Đột phá — `yolo-budget-tuner / research_pipeline.v2` (`1.0.0`)

Báo cáo tổng hợp trung thực toàn bộ các kết quả thực nghiệm sau khi triển khai 3 trụ cột phương pháp luận:
1. **Physics-Residual Split Conformal VRAM (`conformal_vram.py`):** dự báo điểm vật lý (`static_model_state_mb` + Ridge phi tuyến trên độ phức tạp kích hoạt/kiến trúc) kèm cận trên đảm bảo độ phủ mẫu hữu hạn theo không gian tỷ lệ chuẩn hóa cục bộ $s_i = (y_i - \hat{y}_i)/\max(1024, \hat{y}_i)$ và hiệu chỉnh trực tuyến 2 điểm (`calibrate_online_2point`).
2. **Budget-Constrained Conformal Multi-Fidelity Search (`conformal_mf`):** lọc trước ứng viên vi phạm trần VRAM / tràn bộ nhớ dùng chung ở Stage 0, kết hợp probe ngắn chân trời (`probe_epochs=1`) và xếp hạng theo quy tắc khóa trước ở Stage 1; đối chứng trực tiếp bằng cắt bỏ nhân quả (`mf_no_conformal`) và baseline chuẩn hóa mật độ (`tpe_bo` — Tree-structured Parzen Estimator tất định).
3. **Quality-Guarded Multi-Objective Selection (`selection_rule = "epsilon_constraint"`):** tối ưu đồng thời chi phí thời gian và VRAM trong dải giữ chất lượng $\varepsilon$ (`quality_tolerance = 0.03`) thay cho hàm cộng tuyến tính đơn điệu dễ đánh đổi mAP50 lấy tốc độ.

---

## 1. Phase 2 — Khảo sát Thực nghiệm Split Conformal VRAM trên RTX 3050 6GB

- **Phần cứng thực nghiệm:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`5161.0 MiB` free / `6143.5 MiB` total tại thời điểm đo).
- **Quy mô đo cách ly tiến trình (`research_pipeline.v2.worker`):** `28` probe 1-epoch độc lập trên `african_wildlife_benchmark` qua 3 kiến trúc (`yolo11n.pt`, `yolo11s.pt`, `yolo26n.pt`), `imgsz` $\in \{320, 480, 640, 800\}$, `batch` $\in \{2, 4, 8, 16\}$, `precision` $\in \{\text{fp16}, \text{fp32}\}$, `freeze` $\in \{0, 5\}$, `optimizer` $\in \{\text{AdamW}, \text{SGD}\}$.
- **Dữ liệu gốc & Thẻ xuất bản:**
  - `runs/nckh_conformal_vram_study/measurements.json`, `summary.json`, `summary.md`
  - `configs/conformal_vram_card_rtx3050.json` (`sha256`: `c178b61caa2f3a25a3bd9c27557e985e1425df446ea41a9a753f3df95031f91c`, `fit=14`, `cal=14`)

### Kết quả Split Conformal In-Distribution (50 lần chia ngẫu nhiên `fit=13 / cal=8 / test=7`)
- **Point MAE:** `364.2 MiB` | **RMSE:** `485.9 MiB`
- **Empirical Coverage ($\alpha=0.05$, $\alpha=0.10$):** **`0.889 ± 0.164`** (khớp với cận hữu hạn $m/(m+1) = 8/9 = 0.889$ khi cỡ tập hiệu chuẩn $m=8$).
- **Biên conformal trung bình (@95):** **`548.2 MiB`** (độ thiếu hụt cực đại trung bình `72.9 MiB`).
- **Tỷ lệ chặn chính xác cấu hình tràn ngưỡng (Recall):**
  - Tại ngưỡng biên thiết bị nhúng (`1800.0 MiB`): **Recall = `1.000`** (`100%`), False Rejection = `0.252`.
  - Tại ngưỡng tràn VRAM vật lý sang shared RAM WDDM (`5161.0 MiB`): **Recall = `1.000`** (`100%`), False Rejection = `0.068`.
- **Phát hiện 3/3 ca tràn VRAM vật lý sang WDDM Shared Memory:**
  - `yolo11n.pt` `800/B16/fp32`: đo thực tế **`6996.0 MiB`** (`10.80s`) → cận trên Conformal = `7421.3 MiB` (`> 5161 MiB`, **Flagged: True**)
  - `yolo11s.pt` `640/B16/fp32`: đo thực tế **`8180.0 MiB`** (`15.75s`) → cận trên Conformal = `9489.1 MiB` (`> 5161 MiB`, **Flagged: True**)
  - `yolo11s.pt` `800/B16/fp32`: đo thực tế **`12576.0 MiB`** (`49.53s`, nghẽn thrashing ~5×) → cận trên Conformal = `14043.6 MiB` (`> 5161 MiB`, **Flagged: True**)

### Kết quả Leave-One-Model-Out (LOMO) Cross-Architecture (`evaluate_conformal_vram`)
| Mô hình giữ lại | n unseen | Zero-shot MAE (MiB) | 2-pt Cal MAE (MiB) | Zero-shot Cov | 2-pt Cal Cov |
|---|---:|---:|---:|---:|---:|
| `yolo11n.pt` | 8 | 705.1 | 794.3 | `1.000` | `0.750` |
| `yolo11s.pt` | 8 | 283.4 | 463.4 | `0.875` | **`1.000`** |
| `yolo26n.pt` | 6 | 147.6 | 180.8 | `0.833` | **`1.000`** |

*(Ghi chú minh bạch: khi phân chia nguồn theo xáo trộn ngẫu nhiên `seed=2027` trong `_lomo_evaluation`, 2 anchor VRAM thấp trên `yolo11s.pt` cho coverage `0.375` do chưa chứa điểm phi tuyến WDDM ở 8–12 GiB, trong khi chuyển miền `yolo26n.pt` vẫn đạt `1.000` coverage với biên chỉ `280.2 MiB`.)*

---

## 2. Phase 3 — Kiểm chứng Đa-Seed Độc lập (`n = 5` search seeds) trên Hai Miền

Tổng hợp từ `runs/nckh_2027_algorithm_report/ALGORITHM_STUDY.md` (`configs/nckh_2027_breakthrough_studies.json`): `2` miền độc lập, `10` lần lặp tìm kiếm cho mỗi phương pháp (`60` lượt chạy online + `2` reference vét cạn hoàn tất 12/12).

### Study 1 — Detection VRAM-Stress (`african_wildlife_benchmark`, RTX 3050 6GB, `max_memory = 1800 MiB`, `budget = 150s`, `epochs = 5`, `n = 5`)
- Reference vét cạn: `12/12` hoàn tất (`536.05s`, tỷ lệ `budget/ref = 0.2798`), gồm `7` cấu hình khả thi (`298..954 MiB`) và `5` cấu hình vượt trần VRAM (`2582..4542 MiB`).

| Method | Feasible | Conditional U Regret mean | Quality Regret mean | Exact Opt Hit | Pareto Hit | VRAM-Viol Attempts / lượt | Wasted s / lượt | Pruned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`conformal_mf`** | **5/5** | **`0.00000`** | **`0.00000`** | **100.00%** | **100.00%** | **`0.0`** | **`0.00s`** | **`5.0`** |
| `mf_no_conformal` *(cắt bỏ)* | 5/5 | `0.17315` | `0.18539` | 20.00% | 40.00% | `4.0` | `38.42s` | `0.0` |
| `tpe_bo` | 5/5 | `0.07944` | `0.09609` | 60.00% | 80.00% | `1.0` | `49.73s` | `0.0` |
| `random` | **4/5** *(1 fail)* | `0.12678` (`n=4`) | `0.17602` | 20.00% | 40.00% | `1.8` | `73.36s` | `0.0` |
| `adaptive_full` | 5/5 | `0.00000` | `0.00000` | 100.00% | 100.00% | `0.0` | `0.00s` | `0.0` |
| `default` | 5/5 | `0.00000` | `0.00000` | 100.00% | 100.00% | `0.0` | `0.00s` | `0.0` |

- **Kết luận nhân quả trực tiếp (`conformal_mf` vs `mf_no_conformal`):** Cùng cơ chế probe multi-fidelity nhưng bỏ bộ lọc Split Conformal VRAM làm tỷ lệ đạt cấu hình tối ưu tụt từ **`100% -> 20%`**, quality regret tăng từ **`0.0000 -> 0.1854`**, và lãng phí trung bình **`4.0` lượt (`38.42s`)** cho các cấu hình vượt trần VRAM; `time_to_first_full_config_seconds` giảm `-13.84s` (Wilcoxon/Sign-flip `p = 0.0625`, Holm adj. `p = 0.2500`).
- **So với `random`:** `conformal_mf` loại bỏ hoàn toàn ca thất bại (`5/5` vs `4/5`) và giảm `73.36s/lượt` lãng phí xuống `0.00s/lượt` (`online_seconds` giảm `-18.79s`, `p = 0.0625`).
- *(Ghi chú trung thực: trong Study 1, cấu hình `c00` tối ưu sẵn `freeze=5` vô tình đạt mAP50 cao nhất ở 5 epoch nên `default` và `adaptive_full` — vốn luôn chạy `c00` đầu tiên — cũng đạt regret `0.0000`.)*

### Study 2 — Classification Cross-Domain (`cifar10_research`, CPU, `--space core`, `budget = 50s`, `epochs = 5`, `n = 5`)
- Reference vét cạn: `12/12` hoàn tất (`99.98s`, tỷ lệ `budget/ref = 0.5001`), không có thẻ Conformal VRAM cho CPU (chạy fallback vật lý trực tuyến `observed_cross_memory`). Trong miền này `c00` không còn tối ưu (`default` regret > 0).

| Method | Feasible | Conditional U Regret mean | Quality Regret mean | Exact Opt Hit | Pareto Hit | AUARC mean |
|---|---:|---:|---:|---:|---:|---:|
| **`tpe_bo`** | **5/5** | **`0.00685`** | **`0.01467`** | **60.00%** | **100.00%** | **`0.01985`** |
| `random` | 5/5 | `0.01101` | `0.02400` | 20.00% | 100.00% | `0.02176` |
| `conformal_mf` | 5/5 | `0.01700` | `0.03667` | 0.00% | **100.00%** | `0.02908` |
| `adaptive_full` | 5/5 | `0.01700` | `0.03667` | 0.00% | 100.00% | `0.02879` |
| `default` | 5/5 | `0.01725` | `0.03667` | 0.00% | 100.00% | `0.02304` |
| `mf_no_conformal` | 5/5 | `0.02178` | `0.04267` | 20.00% | 40.00% | `0.03013` |

- **Phát hiện khoa học qua đối chứng 2 chế độ:** Khi phần cứng dư bộ nhớ và probe 1-epoch trên tập nhỏ nhiễu mạnh (CIFAR-10 CPU 64 ảnh cho `val acc ~ 0.06–0.09`), thuật toán single-fidelity `tpe_bo` (chiếm toàn bộ budget cho full run dưới dẫn hướng KDE) vượt trội các phương pháp multi-fidelity (`0.00685` vs `0.01700`, `60%` vs `0%` exact hit), còn `conformal_mf` vẫn giữ `100%` Pareto hit và nhỉnh hơn `default` (`-0.00025`, Wilcoxon `p = 0.0625`). Ngược lại, khi xuất hiện ràng buộc VRAM thực tế (Study 1), `conformal_mf` loại bỏ hoàn toàn vi phạm (`0` vs `1.0` ở `tpe_bo` và `4.0` ở `mf_no_conformal`) để đạt `100%` exact hit.

---

## 3. Study 3 — Thực nghiệm Quy mô Đầy đủ (`african_wildlife_full`) & Held-Out Test Độc lập

- **Bộ dữ liệu:** `runs/datasets/african_wildlife_full` (`1,036` train / `216` val / `211` held-out test, 4 lớp: `buffalo`, `elephant`, `rhino`, `zebra`).
- **Thư mục run (đã pass `[AUDIT PASS]` 11/11 lượt worker):** `runs/nckh_2027_study3_wildlife_full/20261126_013218_819548`
- **Cấu hình thực nghiệm:** `--method conformal_mf --selection-rule epsilon_constraint --quality-tolerance 0.03 --max-memory 1800 --full-epochs 12 --seeds 17 29 --budget-seconds 1800`.
- *(Ghi nhận thực nghiệm về Affordability Guard: lần thử đầu với `budget=720s` dừng sớm sau `115.9s` không có cấu hình full vì ngoại suy 1-epoch probe 128 ảnh lên 12 epoch × 1,036 ảnh × 2 seeds dự phóng vượt `607.6s` còn lại; tăng ngân sách lên `1800s` khắc phục triệt để và xác nhận đủ 2/2 ứng viên hàng đầu ở cả 2 seeds trong `1372.80s`.)*

### Kết quả Lựa chọn & Hiệu quả của `epsilon_constraint`
| Ứng viên | Cấu hình | Seeds | Val `mAP50` (mean±SD) | Val `mAP50-95` | Time (s) | VRAM max (MiB) | Utility `U` | Chọn theo `epsilon_constraint` |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`c00` ★** | `yolo11n.pt`, 320, B4, `fp16`, AdamW `3e-4`, `freeze=5` | `2/2` | **`0.9237 ± 0.0088`** | `0.7306 ± 0.0044` | `382.77` | `442.0` | `0.3300` | **ĐƯỢC CHỌN** (`score=0.6409`) |
| `c01` | `yolo11n.pt`, 320, B8, `fp16`, SGD `5e-3`, `freeze=0` | `2/2` | `0.8846 ± 0.0017` | `0.6960 ± 0.0012` | `245.32` | `502.0` | **`0.3550`** | Bị loại (`gap=0.0391 > 0.03`) |

- **Bằng chứng giá trị của `epsilon_constraint`:** Nếu dùng trọng số tuyến tính `U` mặc định (`0.6/0.25/0.15`), hệ thống sẽ chọn `c01` (`U=0.3550 > 0.3300`) và **mất `-3.91%` Val `mAP50`**. Quy tắc `epsilon_constraint` (`eps=0.03`) giữ vững chất lượng cao nhất (`0.9237`) trong giới hạn VRAM `442.0 MiB` (`<< 1800 MiB`).

### Đánh giá Tập Test Độc lập (`evaluate-v2 --seed-scope all`, `211` ảnh, 2 seeds `[17, 29]`, `test_used_for_selection = False`)
- **Held-out Test `mAP@50`:** **`0.9528 ± 0.0004`** (Seed 17: `0.9525`, Seed 29: `0.9530` | Generalization gap `Test - Val = +0.0291`)
- **Held-out Test `mAP@75`:** **`0.8646 ± 0.0010`**
- **Held-out Test `mAP@50:95`:** **`0.7677 ± 0.0030`**
- **Precision / Recall / F1:** `0.9426 ± 0.0097` / `0.8782 ± 0.0118` / `0.9093 ± 0.0108` (`F1-Confidence Peak = 0.9091` tại `conf = 0.482`)
- **Per-class Test `AP@50`:** `buffalo: 0.9464 ± 0.0142`, `elephant: 0.9563 ± 0.0013`, `rhino: 0.9727 ± 0.0036`, `zebra: 0.9357 ± 0.0107`
- **Inference Latency (`rect=False, fp32, batch=4` trên RTX 3050):** **`6.30 ms/ảnh`** (`159.05 ± 9.64 FPS`)

### Bộ Tài liệu & Biểu đồ Xuất bản (`paper-export --lang vi`)
- Tổng hợp đa-nghiên cứu (Study 1 + 2): `runs/paper_exports/nckh_2027_breakthrough_algorithm_vi/PAPER_SUMMARY.md` (`anytime_convergence.svg`, `benchmark_regret.svg`, `tables.tex`, `PROVENANCE.json`)
- Thực nghiệm Full Wildlife (Study 3): `runs/paper_exports/nckh_2027_study3_wildlife_full_vi/PAPER_SUMMARY.md` (`pr_curve.svg`, `confusion_matrix.svg`, `pareto_front.svg`, `per_class_ap50.svg`, `learning_curves.svg`, `anytime_convergence.svg`, `tables.tex`, `PROVENANCE.json`)

---

## 4. Giới hạn Khoa học & Hướng Phát triển Tiếp theo
1. Kết quả `n = 5` search seeds có điều kiện trên từng bộ dữ liệu, lưới cấu hình và phần cứng đã khóa (`RTX 3050 6GB`, `CPU AMD64`); giá trị p-value nhỏ nhất có thể đạt của Wilcoxon/Sign-flip 2 phía ở `n = 5` là `0.0625`, chưa đủ công suất sau hiệu chỉnh Holm-Bonferroni cho tuyên bố vượt trội phổ quát xuyên miền (cần mở rộng lên $n \ge 10$ search seeds và $\ge 2$ GPU vật lý khác nhau).
2. Độ phủ thực nghiệm trên `cal=8` điểm hiệu chuẩn cho cận hữu hạn `0.889`; trên các điểm ngoại suy WDDM Shared Memory lớn (`8–12 GiB`), 2 anchor VRAM nhỏ chưa đủ nắm bắt độ cong paging phi tuyến khi chuyển sang mô hình lớn hơn.
3. Trên hệ thống có ngân sách rất ngắn và không rủi ro OOM (`Study 2`), probe 1-epoch ở tập nhỏ nhiễu cao kém hiệu quả hơn `tpe_bo`; một bộ điều phối chọn lọc thông minh theo mức rủi ro VRAM (`conformal_mf` khi có nguy cơ OOM/tràn trần, `tpe_bo` khi dư VRAM) là hướng nâng cấp trực tiếp tiếp theo.
