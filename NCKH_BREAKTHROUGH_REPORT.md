# Báo cáo Nâng cấp Đột phá `v2.0` (`CC-MFBO`) & Thực nghiệm Kiểm chứng NCKH 2027

Báo cáo tổng kết các nâng cấp thuật toán **v2.0 (`CC-MFBO`)**, nghiên cứu hiệu chuẩn `ConformalVRAMGuard` trên **60** probes GPU thực tế, đối chứng **`8` phương pháp × `n = 8` hạt giống trên `3` miền dữ liệu độc lập (`18` cấu hình/miền)**, phân tích độ nhạy quy tắc chọn lọc Pareto, kiểm định `Test` độc lập ở cả 3 miền, kiểm tra Web UI và bộ kiểm thử tự động cho hệ thống **`yolo-budget-tuner / research_pipeline.v2`**.

---

## 1. Tóm tắt Nâng cấp v2.0 (`CC-MFBO`)

Khắc phục triệt để các điểm hạn chế của bản `1.0` (tập hiệu chuẩn nhỏ `n_cal=8`, tách nhánh ad-hoc sang `tpe_bo`, không gian `12` cấu hình mà `adaptive_full` bám theo `c01`, thiếu đường cơ sở quốc tế `constrained_bo` và `successive_halving`, thiếu độ nhạy $\varepsilon$-constraint):

1. **Nâng cấp `ConformalVRAMGuard` ([conformal_vram.py](research_pipeline/v2/conformal_vram.py), [run_conformal_vram_study.py](scripts/run_conformal_vram_study.py)):**
   - Thu thập **`60` probes cô lập trên GPU RTX 3050 6GB** (`20` probes mỗi họ `yolo11n.pt`, `yolo11s.pt`, `yolo26n.pt`): `56/60` OK, **`3` lần tràn WDDM shared memory (`5,174.0`, `6,216.0`, `6,452.0 MiB > 5,161.0 MiB`)** đều được phát hiện `100%`.
   - Nâng cấp đặc trưng phần dư vật lý phi tuyến (`(flops / 0.01653) ** 0.55`, loại bỏ đa cộng tuyến gây méo LOMO), tự chọn Ridge `PRESS/LOOCV`, chuẩn hóa phần dư dị phương sai cục bộ (`_fit_local_scale`), hàm xác suất khả thi trơn hóa `conformal_feasibility_probability()`, và co ngót độ dốc Bayesian 2-điểm (`prior_w = 15,000.0`).
   - Kết quả `50×` tách ngẫu nhiên (`fit=24 / cal=21 / test=11`, thẻ [conformal_vram_card_rtx3050.json](configs/conformal_vram_card_rtx3050.json)):
     - **`α = 0.05` (`95%`):** độ phủ **`0.964 ± 0.066`**, `MAE = 135.1 MiB`, `RMSE = 187.9 MiB`, tỷ lệ loại nhầm **`6.07%`**, recall ngăn tràn `100%`.
     - **`α = 0.10` (`90%`):** độ phủ **`0.913 ± 0.109`**, tỷ lệ loại nhầm **`3.90%`**, recall ngăn tràn `100%`.
     - **LOMO 2-point transfer (`Cov@95`):** `yolo11n = 1.000`, `yolo11s = 0.944` (`MAE 170.9 MiB`), **`yolo26n = 1.000`** (`MAE 112.8 MiB`, tăng từ `0.643` zero-shot).

2. **Thuật toán Thống nhất `CC-MFBO` ([scheduler.py](research_pipeline/v2/scheduler.py), [acquisition.py](research_pipeline/v2/acquisition.py)):**
   - Giai đoạn 0: Tiền kiểm Conformal loại trước bẫy VRAM (kể cả bẫy `fp16` `c07` `~2,720 MiB > 1,850 MiB` trong khi giữ `fp32` an toàn `c09` `~760 MiB`) và xuất trọng số $\hat{p}_{\text{feas}}(c)$.
   - Giai đoạn 1: Sàng lọc đa tầng **Facility-Location (K-Medoid) phân tầng theo `search_seed`** kết hợp hiệu chỉnh trực tuyến 2-điểm + giãn biên độ tự hồi quy **Kennedy–O'Hagan** ($\hat{\rho}$).
   - Giai đoạn 2: **Multi-Fidelity Residual GP-LOOCV** + hàm mua sắm hậu kiểm $\varepsilon$-Constraint **`cEUI`** (21-pt Gauss–Hermite + phần thưởng bứt phá chất lượng $(q - q_{\text{obs,max}})_+$) với dự báo chi phí điều hòa đa tầng.

3. **Đường cơ sở Quốc tế & Không gian `18` Cấu hình Đa miền:**
   - Bổ sung **`constrained_bo` (`cEI` — Gelbart et al., 2014)** trong [scheduler.py](research_pipeline/v2/scheduler.py) và [algorithm_study.py](research_pipeline/v2/algorithm_study.py).
   - Thiết kế lưới `18` cấu hình Detection ([nckh_2027_v2_detection_18grid.json](configs/nckh_2027_v2_detection_18grid.json)) và Classification CNN ([nckh_2027_v2_cifar10_18grid.json](configs/nckh_2027_v2_cifar10_18grid.json)).

---

## 2. Kết quả Đối chứng `8` Phương pháp × `n = 8` Hạt giống × `3` Miền

Tất cả đối chiếu dùng `--search-seeds 17 29 43 59 73 89 101 127`, `--weights 0.90 0.07 0.03`, `--memory-scale 2048`, `--selection-rule epsilon_constraint` tại [ALGORITHM_STUDY.md](runs/nckh_2027_v2_algorithm_report/ALGORITHM_STUDY.md):

| Miền (Tham chiếu tối ưu ĐỘC LẬP) | Ngân sách / Vét cạn | **`conformal_mf` (CC-MFBO)** | `adaptive_full` | `mf_no_conformal` | `constrained_bo` (`cEI`) | `tpe_bo` | `successive_halving` | `random` | `default` (`c00`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Wildlife (`mAP@50`)** $\rightarrow$ **`c14`** (`0.9379`) | `135s` / `1029.7s` | **`0.0748 ± 0.1143`**<br>`8/8` \| `50% opt` \| **`0.00 viol`** | `0.2868 ± 0.0495`<br>(`p=0.0078`) | `0.1540 ± 0.1333`<br>(`7/8`, `1.38 viol`) | `0.2529 ± 0.1903`<br>(`p=0.0312`, `0.88 viol`) | `0.3054 ± 0.2262`<br>(`7/8`, `0.88 viol`) | `0.2868 ± 0.0495`<br>(`p=0.0078`) | `0.2532 ± 0.1545`<br>(`SF p=0.0391`, `0.62 viol`) | `0.2509 ± 0.0000`<br>(`p=0.0156`) |
| **2. Penn-Fudan (`mAP@50:95`)** $\rightarrow$ **`c17`** (`0.8772`) | `108s` / `503.9s` | **`0.0009 ± 0.0026`**<br>`8/8` \| **`87.5% opt`** \| **`0.00 viol`** | `0.1293 ± 0.0000`<br>(`p=0.0078`) | `0.0908 ± 0.0566`<br>(`7/8`, **`p=0.0156`**, `1.62 viol`) | `0.1450 ± 0.1591`<br>(`7/8`, **`p=0.0312`**, `0.88 viol`) | `0.1081 ± 0.1255`<br>(**`p=0.0156`**, `1.00 viol`) | `0.1542 ± 0.0705`<br>(**`p=0.0078`**) | `0.1398 ± 0.1118`<br>(**`p=0.0156`**, `0.75 viol`) | `0.1293 ± 0.0000`<br>(**`p=0.0078`**) |
| **3. CIFAR-10 GPU (`Top-1 Acc`)** $\rightarrow$ **`c12`** (`0.4200`) | `68s` / `295.6s` | `0.0619 ± 0.0487`<br>`8/8` \| `12.5% opt` | `0.0715 ± 0.0000`<br>(`0% opt`) | `0.0619 ± 0.0487`<br>(`12.5% opt`) | `0.0407 ± 0.0384`<br>(`25% opt`) | **`0.0372 ± 0.0331`**<br>(`25% opt`) | `0.0715 ± 0.0000`<br>(`0% opt`) | `0.0833 ± 0.0314`<br>(`0% opt`) | `0.0715 ± 0.0000`<br>(`0% opt`) |

*Ghi chú trung thực:* Trên miền `CIFAR-10 GPU` không có rủi ro tràn VRAM và thời gian mỗi thử nghiệm nhỏ, `tpe_bo` và `constrained_bo` vượt `conformal_mf` nhờ không tốn ~15s thăm dò đầu khi rank ở epoch 4 trên 256 ảnh bị nhiễu; dù vậy `conformal_mf` vẫn vượt `adaptive_full`, `successive_halving`, `default` và `random`.

---

## 3. Độ nhạy Quy tắc Chọn lọc & Đánh giá `Test` Độc lập trên 3 Miền

- **Độ nhạy Quy tắc Chọn lọc ([summary.md](runs/nckh_2027_v2_selection_ablation/summary.md), [pareto_epsilon_tradeoff.svg](runs/nckh_2027_v2_selection_ablation/pareto_epsilon_tradeoff.svg)):** `epsilon_constraint` với $\varepsilon \in [0.005, 0.015]$ giữ vững tối ưu toàn cục ở cả 3 miền (`c14`, `c17`, `c12`); tăng $\varepsilon \in [0.02, 0.05]$ trên Penn-Fudan trượt êm dọc đường Pareto sang `c14` (`0.8617 mAP@50:95`, tiết kiệm `-5.4s` và `-408 MiB` VRAM); trái lại `Knee` rớt mạnh `-28.6%` trên Wildlife và `-8.0%` trên CIFAR-10.
- **Đánh giá `Test` Giữ lại Độc lập ([heldout_test_summary.json](runs/nckh_2027_v2_selection_ablation/heldout_test_summary.json)):**
  - **Wildlife (`african_wildlife_full`, `12` ep, `2` seeds):** **`Test mAP@50 = 0.9528 ± 0.0004`**, `mAP@50:95 = 0.7677`, `F1 = 0.9093`, **`159.05 FPS`**.
  - **Penn-Fudan (`12` ep, `3` seeds `[17,29,43]`):** **`Test mAP@50 = 0.9785 ± 0.0107`**, **`Test mAP@50:95 = 0.8667 ± 0.0186`**, `F1 = 0.9517`, **`73.16 FPS`**.
  - **CIFAR-10 (`20` ep, `3` seeds `[17,29,43]`):** **`Test Acc = 0.3480 ± 0.0231`** (`Val 0.3678 ± 0.0369`).

---

## 4. Báo cáo Toàn văn & Xuất bản LaTeX / Web UI

- **Báo cáo Khoa học NCKH đầy đủ:** [BAO_CAO_NCKH_2027.md](BAO_CAO_NCKH_2027.md)
- **Gói `paper-export` song ngữ 3 miền (LaTeX `tables.tex`, CSV, SVG):**
  - Tiếng Việt: [nckh_2027_v2_3domain_vi/PAPER_SUMMARY.md](runs/paper_exports/nckh_2027_v2_3domain_vi/PAPER_SUMMARY.md), [tables.tex](runs/paper_exports/nckh_2027_v2_3domain_vi/tables.tex)
  - English: [nckh_2027_v2_3domain_en/PAPER_SUMMARY.md](runs/paper_exports/nckh_2027_v2_3domain_en/PAPER_SUMMARY.md), [tables.tex](runs/paper_exports/nckh_2027_v2_3domain_en/tables.tex)
- **Web UI Studio:** Hỗ trợ đủ 8 phương pháp (bao gồm `CC-MFBO`, `constrained_bo`, `successive_halving`), tự xoay nhãn thanh SVG khi `rows.length > 8` ([charts.js](research_pipeline/v2/web_ui/static/charts.js)) và bảng kiểm toán an toàn VRAM.
