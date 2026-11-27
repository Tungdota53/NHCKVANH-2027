# NCKH 2027 BREAKTHROUGH REPORT — CC-MFBO (v2.0 Rigor Pack)

**Cập nhật:** 27/11/2026 | **Nhánh:** `main` (Commit nền `2c62a19` + Gói củng cố học thuật hoàn chỉnh)

---

## 1. Tóm tắt bước nhảy vọt từ v1 lên v2.0 Rigor Pack

Dự án đã hoàn tất chuyển hóa từ một công cụ dò siêu tham số nội bộ (`v1`) thành một **công trình nghiên cứu thuật toán có kiểm chứng thống kê chặt chẽ và sản phẩm phần mềm hoàn chỉnh (`CC-MFBO`)**:

1. **Đóng góp thuật toán mới:** **CC-MFBO (Conformal-Guarded Cost-Aware Multi-Fidelity BO)** tích hợp:
   - **Split Conformal VRAM Guard** từ vector đặc trưng vật lý $d=5$ trên 60 phép thử RTX 3050 (`56 ok`, `50× Split Eval` đạt **96.36% coverage** tại $\alpha=0.05$, OOM prevention **100%**).
   - **2-Point Online Shrinkage Calibration**: giải quyết chuyển giao kiến trúc mới (`LOMO` trên `yolo26n.pt` phục hồi coverage từ **64.3% $\rightarrow$ 100.0%**, giảm MAE từ `182.1 MiB` xuống `112.8 MiB`).
   - **Cost-Aware Heteroscedastic GP + Conformal Pre-Pruning**: cắt trước `5.0` cấu hình vượt ngưỡng VRAM mỗi lượt chạy trên YOLO Detection, đưa lãng phí VRAM từ `20.2–38.3s` (`15.4–31.9%` tổng thời gian chạy của các baseline) xuống còn **`0.0s (0.0%)`**.

2. **Chứng minh thống kê Two-Family Holm–Bonferroni trên 3 Domain ($n = 8$ paired search seeds):**
   - **Study 2 (Penn-Fudan Pedestrian, `108s`):** `CC-MFBO` đạt simple regret **`0.0009 ± 0.0026`** (**87.5%** trúng tối ưu, `0%` vi phạm), vượt trội **7/7** phương pháp đối chứng với độ tin cậy thống kê $p_{\text{Holm}} < 0.05$ ở cả **Family A (`m=5` external baselines: `0.0391–0.0469`)** và **Family B (`m=2` ablations: `0.0156`)**.
   - **Study 1 (African Wildlife, `135s`):** `CC-MFBO` đứng #1 với regret **`0.0748 ± 0.1143`** (`50.0%` trúng tối ưu), vượt trội có ý nghĩa trước `Adaptive Full` ($p_{\text{Holm}}=0.0391$) và `Successive Halving` ($p_{\text{Holm}}=0.0156$).
   - **Study 3 (CIFAR-10 CNN, `68s`):** Xác lập chính xác **ranh giới áp dụng định lượng** thông qua **Kendall's $\tau$**: khi không gian không thắt VRAM (`0/18` vi phạm) và thứ hạng epoch-1 bất ổn ($\tau = 0.1946, p = 0.269$ so với $\tau = 0.7082\text{--}0.7778$ ở Study 1–2), thuật toán thoái lui an toàn về parity ($p_{\text{Holm}} = 1.00$).

3. **Kiểm chứng Paired 3-Seed Held-Out `Test` (`[17, 29, 43]`, `rect=False`):**
   - **African Wildlife (`12ep` Test `mAP@50`):** `c14` **`0.9587 ± 0.0220`** vs `Default c00` `0.9179 ± 0.0211` (**`+4.44%`**, `115.96 FPS`).
   - **Penn-Fudan (`12ep` Test `mAP@50:95`):** `c17` **`0.8667 ± 0.0186`** vs `Default c00` `0.7302 ± 0.0266` (**`+18.70%`**, thắng trên **3/3** seed, `73.16 FPS`).
   - **CIFAR-10 CNN (`20ep` Test `Top-1 Acc`):** `c12` **`0.3987 ± 0.0110`** vs `Default c00` `0.3973 ± 0.0083` (`+0.34%`).

---

## 2. Tài sản học thuật và công cụ tái lập mới bổ sung

- **Kịch bản kiểm chứng & phân tích:**
  - [scripts/run_paired_test_validation.py](file:///d:/scr%20code/NCHK%202027/scripts/run_paired_test_validation.py) — Chạy & tổng hợp đối chứng Paired 3-Seed Held-Out Test cho cả 3 domain (`runs/nckh_2027_v2_rigor_pack/paired_test_comparison.{json,md}`).
  - [scripts/build_rigor_diagnostics_and_figures.py](file:///d:/scr%20code/NCHK%202027/scripts/build_rigor_diagnostics_and_figures.py) — Tính Kendall $\tau$, Spearman $\rho$, Two-Family Holm, Budget Waste Breakdown và xuất **5 biểu đồ 300-DPI Okabe–Ito (`PNG` + `PDF`)** tại `runs/nckh_2027_v2_rigor_pack/figures/fig1..fig5.*`.
  - [scripts/colab_cross_gpu_conformal_transfer.py](file:///d:/scr%20code/NCHK%202027/scripts/colab_cross_gpu_conformal_transfer.py) — Kịch bản 1-lệnh chạy trên Google Colab T4 / Kaggle (hỗ trợ `--dry-run-measurements`, đã verify 20/20 coverage = `100.0%`, MAE `44.0 MiB`).
- **Tài liệu NCKH hoàn chỉnh:**
  - [BAO_CAO_NCKH_2027.md](file:///d:/scr%20code/NCHK%202027/BAO_CAO_NCKH_2027.md) (Báo cáo toàn văn có Algorithm 1, Flowchart Mermaid, Two-Family Holm, Paired Held-Out Test, Applicability Diagnostics và Phụ lục Colab).
  - [SLIDE_BAO_VE_NCKH_2027.md](file:///d:/scr%20code/NCHK%202027/SLIDE_BAO_VE_NCKH_2027.md) (Bộ khung 14 slide thuyết trình + kịch bản demo 2 phút + 6 câu Q&A phản biện).
