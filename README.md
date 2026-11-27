# NCKH 2027 — CC-MFBO: Conformal-Guarded Multi-Fidelity Bayesian Optimization

Hệ thống nghiên cứu và tối ưu cấu hình huấn luyện đa mục tiêu / đa độ tin cậy có lá chắn **Split Conformal VRAM Safety Guard** dành cho bài toán thị giác máy tính (YOLO Object Detection & CNN Classification) trên GPU giới hạn bộ nhớ, tuân thủ giao thức chống rò rỉ dữ liệu kiểm thử (`Global Test Blindness`) và kiểm toán tái lập SHA-256.

---

## Điểm nhấn khoa học (Verified on RTX 3050 6 GB)

- **Split Conformal VRAM Safety Guard + 2-Point Online Calibration ([conformal_vram.py](file:///d:/scr%20code/NCHK%202027/research_pipeline/v2/conformal_vram.py)):**
  - 60 phép đo worker cô lập trên `yolo11n`, `yolo11s`, `yolo26n` (`56 ok`); kiểm định 50× split tại mức 95% ($\alpha=0.05$) đạt coverage thực tế **`96.36% ± 6.62%`**, **OOM prevention `100.0%`**.
  - Kiểm định chuyển giao kiến trúc mới (**LOMO** trên `yolo26n.pt`): **2-Point Shrinkage Calibration** phục hồi coverage từ `64.3%` lên **`100.0%`** và giảm MAE 38% (`182.1 -> 112.8 MiB`).
- **Hiệu quả loại bỏ lãng phí & Hội tụ vượt trội (3 Domain $\times$ 8 Thuật toán $\times$ 8 Search Seeds):**
  - **Study 2 (Penn-Fudan Detection, `108s` budget):** `CC-MFBO` đạt simple regret **`0.0009 ± 0.0026`** (**`87.5%`** trúng cấu hình tối ưu toàn cục, `0.0s` lãng phí VRAM), thắng có ý nghĩa thống kê toàn bộ **7/7** đối thủ theo **Two-Family Holm–Bonferroni** ($p_{\text{Holm}} \in [0.0156, 0.0469] < 0.05$).
  - **Study 1 (African Wildlife Detection, `135s` budget):** `CC-MFBO` đứng #1 với regret **`0.0748 ± 0.1143`** (`50.0%` trúng tối ưu, giảm 70.2% regret so với `Default c00`, `0.0s` lãng phí VRAM trong khi các BO/Random tiêu chuẩn lãng phí **`20.2–38.3s` (`15.4–31.9%` budget)**).
  - **Study 3 (CIFAR-10 CNN, `68s` budget):** Xác định rõ **ranh giới áp dụng** — khi không gian không thắt VRAM (`0/18` vi phạm) và thứ hạng epoch-1 bất ổn (**Kendall $\tau = 0.1946$** so với $\tau = 0.7082\text{--}0.7778$ ở YOLO), hiệu năng hội tụ an toàn về parity ($p_{\text{Holm}} = 1.00$).
- **Đối chứng Paired 3-Seed Held-Out Test (`[17, 29, 43]`, `rect=False`):**
  - **African Wildlife (12 ep, mAP@50):** `c14` **`0.9587 ± 0.0220`** vs `Default c00` `0.9179 ± 0.0211` (**`+4.44%`**, `115.96 FPS`).
  - **Penn-Fudan (12 ep, mAP@50:95):** `c17` **`0.8667 ± 0.0186`** vs `Default c00` `0.7302 ± 0.0266` (**`+18.70%`**, thắng **3/3 seeds**, `73.16 FPS`).

---

## Cấu trúc thư mục chính

- `research_pipeline/v2/`: Lõi hệ thống tối ưu (`conformal_vram.py`, `acquisition.py`, `search.py`, `algorithm_study.py`, `deployment.py`, `paper_export.py`, `cli.py`, `ui_app.py`).
- `configs/`: Các không gian cấu hình 18 điểm (`nckh_2027_v2_detection_18grid.json`, `nckh_2027_v2_cifar10_18grid.json`), manifest 3 domain (`nckh_2027_v2_3domain_studies.json`), và Conformal VRAM Card (`conformal_vram_card_rtx3050.json`).
- `scripts/`: Bộ kịch bản tái lập thực nghiệm:
  - `run_conformal_vram_study.py`: Đo 60 probes & xuất Conformal VRAM Card.
  - `run_paired_test_validation.py`: Chạy Paired 3-seed Held-Out Test (`CC-MFBO` vs `Default c00`) qua `evaluate-v2 --seed-scope all`.
  - `build_rigor_diagnostics_and_figures.py`: Tính Kendall $\tau$, Two-Family Holm, Budget Waste & xuất 5 biểu đồ Okabe–Ito 300-DPI (`PNG` + `PDF`).
  - `colab_cross_gpu_conformal_transfer.py`: Kiểm chứng chuyển giao đa GPU trên Google Colab T4 / Kaggle (hoặc `--dry-run-measurements`).
- `tests/`: Bộ kiểm thử đơn vị tự động (unit tests).
- Tài liệu báo cáo & bảo vệ:
  - [BAO_CAO_NCKH_2027.md](file:///d:/scr%20code/NCHK%202027/BAO_CAO_NCKH_2027.md)
  - [SLIDE_BAO_VE_NCKH_2027.md](file:///d:/scr%20code/NCHK%202027/SLIDE_BAO_VE_NCKH_2027.md)
  - [NCKH_BREAKTHROUGH_REPORT.md](file:///d:/scr%20code/NCHK%202027/NCKH_BREAKTHROUGH_REPORT.md)

---

## Hướng dẫn chạy nhanh & Tái lập

```powershell
# 1. Chạy toàn bộ unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -q

# 2. Mở Web UI tương tác (Dashboard / Search / Study / Export)
.\.venv\Scripts\python.exe -m research_pipeline.v2 ui

# 3. Tái tạo chẩn đoán học thuật & 5 hình 300 DPI từ kết quả đã chạy
.\.venv\Scripts\python.exe scripts/build_rigor_diagnostics_and_figures.py

# 4. Chạy thử kiểm chứng chuyển giao Conformal VRAM không cần GPU (Dry-Run)
.\.venv\Scripts\python.exe scripts/colab_cross_gpu_conformal_transfer.py --dry-run-measurements runs/nckh_conformal_vram_study/measurements.json

# 5. Chạy thật trên Google Colab T4 / Kaggle GPU (tự probe 24 cấu hình + 2-pt calibration)
python scripts/colab_cross_gpu_conformal_transfer.py --device 0 --output runs/colab_t4_transfer
```
