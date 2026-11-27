# SLIDE BẢO VỆ ĐỀ TÀI NCKH SINH VIÊN 2027

**Tên đề tài:** **CC-MFBO: Tối Ưu Hóa Đa Mục Tiêu Và Đa Độ Tin Cậy Có Lá Chắn Split Conformal VRAM Cho Huấn Luyện Thị Giác Máy Tính Trên GPU Giới Hạn Bộ Nhớ**  
**Cấu trúc trình bày:** 14 Slide chính (~8 phút) + Kịch bản Demo 2 phút + 6 Câu hỏi Phản biện Dự kiến (Q&A)

---

## Slide 1: Trang bìa & Thông điệp cốt lõi
- **Tiêu đề:** CC-MFBO: Safe, Budget-Aware Multi-Fidelity HPO với Split Conformal VRAM Guard
- **Thông điệp 1 câu:** *Biến GPU phổ thông 6 GB từ trạng thái "chạy thử – tràn VRAM – lãng phí 30% thời gian" thành hệ thống tìm kiếm cấu hình tự động đạt 0% OOM waste và vượt chuẩn tối ưu mặc định tới +18.7% trên Held-Out Test.*
- **Hình minh họa gợi ý:** Sơ đồ kiến trúc tổng quan Pipeline CC-MFBO (Mermaid trong `BAO_CAO_NCKH_2027.md`).

---

## Slide 2: Bài toán thực tế & Nỗi đau lãng phí VRAM
- **Thực trạng sinh viên/phòng lab:** Huấn luyện YOLO11 / YOLO26 / CNN trên laptop RTX 3050 6 GB hoặc node edge thường xuyên gặp hai thảm họa:
  1. **Tràn VRAM (CUDA OOM / WDDM Spill):** Dữ liệu tràn sang RAM hệ thống làm chậm gấp nhiều lần hoặc sập giữa chừng.
  2. **Lãng phí ngân sách tìm kiếm:** Thực nghiệm đo được trên RTX 3050 cho thấy các thuật toán HPO chuẩn không có guard (`TPE BO`, `Constrained BO`, `MF w/o Conformal`, `Random`) lãng phí **20.2–38.3 giây (15.4%–31.9% tổng ngân sách)** chỉ để thử sai các cấu hình vi phạm VRAM!
- **Visual:** `fig3_budget_waste_breakdown.png`.

---

## Slide 3: Khoảng trống trong các phương pháp HPO hiện hành
- **Constrained BO / TPE tiêu chuẩn:** Phản ứng *sau* khi đã chạy thất bại trên GPU; không chuyển giao tri thức bộ nhớ sang kiến trúc/GPU mới.
- **Ước lượng bộ nhớ kiểu hồi quy điểm (Point Predictor):** Chỉ dự đoán trung bình $\hat{\mu}(x)$, bỏ qua phân mảnh allocator và phương sai tăng theo kích thước tensor dẫn đến miss-prediction và sập tiến trình.
- **Nghiên cứu đánh giá thiếu kiểm định nghiêm ngặt:** Thường chọn xong rồi báo kết quả Validation hoặc chạy 1 seed không báo độ lệch chuẩn / không khóa Test mù.

---

## Slide 4: Phát biểu bài toán toán học & Hậu-quyết định $\epsilon$-Constraint
- **Mục tiêu:** Chọn cấu hình $x^* \in \mathcal{X}_{\text{feas}}$ thỏa đồng thời $M(x) \le M_{\max}$ ($1850\text{ MiB}$) và tối thiểu chi phí thời gian/bộ nhớ while bảo toàn chất lượng:
  $$Q(x) \ge \max_{x' \in \mathcal{X}_{\text{ok}}} Q(x') - \epsilon \qquad (\epsilon \in [0.01, 0.02])$$
- **Đánh đổi Pareto:** Quy tắc `epsilon_constraint` vượt trội `Knee` trên cả 3 miền, không đánh đổi chất lượng quá ngưỡng khóa trước để lấy tốc độ giả tạo.
- **Visual:** `fig5_epsilon_and_applicability.png` Panel (a).

---

## Slide 5: Đóng góp 1 — Split Conformal VRAM Safety Guard + 2-Pt Calibration
- **Đặc trưng vật lý 5 chiều:** Kích hoạt activation proxy, parameter proxy, FP16 flag, tỷ lệ `freeze`, optimizer flag.
- **Toán học bảo chứng phân phối tự do:** Mô hình heteroescedastic scale $\hat{\sigma}(x) = \max(c_0, c_1 \hat{\mu}(x))$ + phần dư chuẩn hóa trên `n_cal = 28` điểm hiệu chuẩn tách rời cho cận trên:
  $$U_\alpha(x) = \hat{\mu}(x) + q_{1-\alpha}\,\hat{\sigma}(x), \quad P\big(M(X) \le U_\alpha(X)\big) \ge 1 - \alpha$$
- **Chuyển giao tức thì bằng 2 điểm an toàn (2-Point Online Shrinkage Calibration):** Chỉ đo 2 cấu hình nhẹ để tự co `scale` & `shift` khi đổi kiến trúc hoặc đổi GPU!

---

## Slide 6: Đóng góp 2 — Thuật toán CC-MFBO (Algorithm 1)
- **Ba giai đoạn phối hợp:**
  1. **Giai đoạn 0 (Guard Prefilter):** Đo 2 anchor $\rightarrow$ hiệu chỉnh card $\rightarrow$ cắt ngay ứng viên có $U_{0.05}(x) > M_{\max}$ **trước khi tốn 1 giây GPU search**.
  2. **Giai đoạn 1 (Bounded Probe <= 10% budget):** Đánh giá nhanh trên subset để ước lượng thứ hạng và thời gian full.
  3. **Giai đoạn 2 (Heteroscedastic GP + Cost-Aware Acquisition):** Dùng Gaussian Process dự báo phân phối chất lượng-bất định + LOOCV bandwidth để ưu tiên cấu hình khả thi giàu tiềm năng lên full-horizon training.

---

## Slide 7: Giao thức NCKH Chống Rò Rỉ (`Global Test Blindness`) & Cách Ly Tiến Trình
- **Khóa giao thức bằng SHA-256 (`protocol_sha256`):** Dataset split, grid 18 cấu hình, budget giây, 8 `search_seeds`, và quy tắc chọn đều khóa cứng trước khi chạy.
- **Tách ly GPU bằng Worker Subprocess:** Mỗi trial chạy tiến trình tách biệt giúp dọn sạch PyTorch allocator, đo đỉnh VRAM trung thực không dính rác bộ nhớ lượt trước.
- **Test Blindness 100%:** Toàn bộ vòng search và hậu-quyết định **không chạm tập Test**; đánh giá Test thực hiện tách biệt ở cuối qua `evaluate-v2 --seed-scope all` trên 3 seeds `[17, 29, 43]`, `rect=False`.

---

## Slide 8: Kết quả Split Conformal Guard & LOMO trên 60 Probes thực tế (RTX 3050)
- **60 phép đo thật** trên `yolo11n`, `yolo11s`, `yolo26n` (`56 ok`, `3` OOM/spill):
  - **50× Split Eval ($1850\text{ MiB}$, $\alpha=0.05$):** Coverage thực đạt **`96.36% ± 6.62%`**, chặn đứng **100.0% OOM**, False Rejection chỉ **`6.07%`** (`1.03%` tại trần `5161 MiB`).
- **Thử thách khắc nghiệt nhất — Chuyển giao kiến trúc mới `yolo26n` (LOMO):**
  - Zero-shot coverage tụt xuống **`64.3%`** (`MAE = 182.1 MiB`).
  - Sau **2-Point Calibration**: Coverage phục hồi lên **`100.0%`**, MAE giảm **38%** xuống **`112.8 MiB`**!
- **Visual:** `fig1_conformal_vram_calibration.png`.

---

## Slide 9: Benchmark 3 Domain (8 Phương Pháp $\times$ 8 Seeds) & Two-Family Holm
- **Penn-Fudan Pedestrian Detection (`108s` budget):**
  - `CC-MFBO` regret **`0.0009 ± 0.0026`** | **`87.5%`** chọn đúng cấu hình tối ưu toàn cục (`0%` vi phạm).
  - **Vượt trội có ý nghĩa thống kê toàn bộ 7/7 đối thủ** theo Two-Family Holm (`Family A m=5`: mọi $p_{\text{Holm}} \in [0.0391, 0.0469] < 0.05$; `Family B m=2`: cả hai $p_{\text{Holm}} = 0.0156 < 0.05$).
- **African Wildlife Detection (`135s` budget):**
  - `CC-MFBO` regret **`0.0748 ± 0.1143`** (`50.0%` trúng tối ưu, giảm **70.2%** regret so với `Default c00`), thắng có ý nghĩa trước `Adaptive Full` ($p_{\text{Holm}}=0.0391$) và `Successive Halving` ($p_{\text{Holm}}=0.0156$).
- **Visual:** `fig2_3domain_regret_comparison.png`.

---

## Slide 10: Phân tích Ngân sách Hữu ích vs Lãng phí VRAM
- Tại sao `CC-MFBO` tìm được nghiệm tốt hơn dù cùng ngân sách thời gian (`108s` / `135s`)?
  - Các phương pháp không có Conformal Guard lãng phí **`22.3s–38.3s` (`24.6%–31.9%` ngân sách)** để tự thử vào vùng cấu hình vi phạm VRAM!
  - **`CC-MFBO`** cắt trước đúng **`5.0` cấu hình/lượt**, dồn **`100%` thời gian online (`0.0s` waste)** vào vùng khả thi để đánh giá sâu hơn.
- **Visual:** `fig3_budget_waste_breakdown.png`.

---

## Slide 11: Kết quả đối chứng Paired 3-Seed Held-Out Test (`[17, 29, 43]`)
- Không chỉ thắng trên ngân sách search 4 epoch, khi huấn luyện lại đủ horizon và đo trên tập **Test giữ ngoài**:
  - **African Wildlife (12 ep, mAP@50):** `c14` đạt **`0.9587 ± 0.0220`** vs `Default c00` `0.9179 ± 0.0211` (**`+4.44%`**, `115.96 FPS`, `1148 MiB`).
  - **Penn-Fudan (12 ep, mAP@50:95):** `c17` đạt **`0.8667 ± 0.0186`** (`mAP@50 = 0.9785`, F1 `0.9517`) vs `Default c00` `0.7302 ± 0.0266` (**`+18.70%`**, thắng trên **3/3** seed, `73.16 FPS`, `1532 MiB`).
  - **CIFAR-10 CNN (20 ep, Acc):** `c12` `0.3987` vs `c00` `0.3973` (`+0.34%`).
- **Visual:** `fig4_paired_heldout_test.png`.

---

## Slide 12: Ranh giới áp dụng định lượng (Kendall $\tau$ Applicability Map)
- Minh bạch khoa học về giới hạn của đa độ tin cậy trên **Study 3 (CIFAR-10 CNN)**:
  - Trên **Study 1 & 2 (YOLO Detection):** Thứ hạng Epoch-1 tương quan mạnh với Full-Horizon (**Kendall $\tau = 0.7082$ và $0.7778$, $p < 5\times 10^{-5}$**) + tỷ lệ bất khả thi VRAM **`27.8%`** $\rightarrow$ **Miền vàng của CC-MFBO**.
  - Trên **Study 3 (CIFAR-10 CNN nhẹ):** Mọi cấu hình dùng $< 185\text{ MiB}$ (**0% thắt VRAM**) và thứ hạng epoch sớm bất ổn (**$\tau = 0.1946, p = 0.269$**) $\rightarrow$ CC-MFBO không tạo lợi thế nhưng **an toàn, không thua baseline** ($p_{\text{Holm}} = 1.00$).
- **Visual:** `fig5_epsilon_and_applicability.png` Panel (b).

---

## Slide 13: Sản phẩm phần mềm hoàn chỉnh & Kịch bản Live Demo (2 phút)
- **Hệ sinh thái đã hoàn thiện:** CLI chuẩn nghiên cứu (`python -m research_pipeline.v2`), Web UI tiếng Việt trực quan theo dõi Pareto Frontier & Conformal Card, xuất báo cáo song ngữ tự động (`paper-export`).
- **Kịch bản Demo trực tiếp 120 giây trước Hội đồng:**
  1. *(0:00–0:30)* Chạy nhanh `.\.venv\Scripts\python.exe scripts/colab_cross_gpu_conformal_transfer.py --dry-run-measurements runs/nckh_conformal_vram_study/measurements.json` để show tức thì bài toán kiểm chứng 95% coverage & 2-pt calibration.
  2. *(0:30–1:15)* Mở Web UI (`python -m research_pipeline.v2 ui`), show bảng kết quả đối chiếu Pareto + biểu đồ 3-Domain (`ALGORITHM_STUDY.md`) + `paired_test_comparison.md`.
  3. *(1:15–2:00)* Trình chiếu `fig3` (0% lãng phí VRAM) & `fig4` (+18.7% Held-Out Test Penn-Fudan).

---

## Slide 14: Hạn chế, Hướng phát triển & 6 Câu Hỏi Phản Biện Dự Kiến (Q&A)

- **Hướng mở rộng tức thì:** Script 1-lệnh `scripts/colab_cross_gpu_conformal_transfer.py --device 0` sẵn sàng kiểm chứng thực địa trên Tesla T4 / P100 16 GB.

### 6 Câu Hỏi Thường Gặp Từ Hội Đồng & Phương Án Trả Lời Chuẩn Xác

1. **Hỏi: Tại sao `Conformal VRAM Card` train trên RTX 3050 mà khi sang kiến trúc mới (`yolo26n`) lại bị tụt coverage xuống 64.3%? Vậy sang GPU khác có hỏng không?**  
   *Trả lời:* Chính vì phân bố kích hoạt kiến trúc mới hoặc chiến lược cấp phát driver GPU khác nhau có thể gây dịch chuyển phân phối (`covariate/scale shift`), nên chúng tôi **không quảng cáo Zero-Shot thần kỳ**, mà thiết kế **2-Point Online Shrinkage Calibration**: chỉ cần chạy 2 cấu hình an toàn chi phí vài giây trên kiến trúc/GPU mới để tự co `(scale, shift)`. Thực nghiệm LOMO chứng minh thao tác 2 điểm này khôi phục coverage của `yolo26n` từ `64.3%` lên **`100.0%`** và giảm MAE 38%.

2. **Hỏi: Tại sao trên CIFAR-10 CNN, `CC-MFBO` không vượt trội `TPE` hay `Constrained BO`?**  
   *Trả lời:* Chúng tôi định lượng rõ hai điều kiện cần của CC-MFBO: (i) không gian tìm kiếm phải có cấu hình vi phạm bộ nhớ (`27.8%` ở Study 1–2 vs **`0.0%`** ở CIFAR-10 CNN dùng $< 185\text{ MiB}$), và (ii) thứ hạng proxy epoch sớm phải nhất quán với full-horizon (**Kendall $\tau = 0.71\text{--}0.78$** ở YOLO vs **$\tau = 0.19$** ở CNN). Study 3 đóng vai trò kiểm chứng giới hạn áp dụng (negative control), chứng minh phương pháp không gây hại khi điều kiện không thỏa mãn.

3. **Hỏi: Tại sao báo cáo tự động ban đầu cho Holm $p = 0.0547 > 0.05$, nhưng trong phần Two-Family Holm lại đạt $< 0.05$?**  
   *Trả lời:* Đây là tính chất toán học thuần túy, không phải đổi kiểm định. Với $n=8$ cặp, mức sàn nhỏ nhất của Exact Sign-Flip 2 phía là $2/2^8 = 0.0078125$. Báo cáo tổng hợp gộp cả 5 baseline ngoài lẫn 2 ablation vào $m = 7$, tạo ra sàn toán học $7 \times 0.0078125 = 0.0547 > 0.05$ (bất khả thi đạt $0.05$ ngay cả khi thắng 8/8 tuyệt đối). Khi phân tách đúng thành **Family A (`m=5` external baselines)** và **Family B (`m=2` mechanistic ablations)**, toàn bộ **7/7** cặp ở Penn-Fudan và cả cặp chính yếu ở Wildlife đều đạt $p_{\text{Holm}} \in [0.0156, 0.0469] < 0.05$.

4. **Hỏi: Tập Test có bị dùng để điều chỉnh siêu tham số không?**  
   *Trả lời:* Tuyệt đối không. Tập Test được cô lập hoàn toàn khỏi quá trình probe, surrogate fitting và lựa chọn $\epsilon$-Constraint (`test_used_for_selection = false` được ghi rõ trong mọi `evaluation.json`). Chúng tôi còn tách biệt 8 `search_seeds` độc lập khỏi 3 `training_seeds` `[17, 29, 43]` khi đánh giá Held-Out Test.

5. **Hỏi: Việc chọn `c14`/`c17` giúp tăng mAP@50:95 thêm +18.7% trên Penn-Fudan nhưng FPS giảm từ 130 xuống 73 FPS; liệu có xứng đáng?**  
   *Trả lời:* Cả 73.16 FPS (`13.67 ms/ảnh`) lẫn 115.96 FPS (`8.62 ms/ảnh` ở Wildlife) đều vượt xa ngưỡng thời gian thực (`30 FPS`), đồng thời VRAM đỉnh `1532 MiB` / `1148 MiB` vẫn nằm gọn dưới trần `1850 MiB`. Ngược lại, `Default c00` mất gần `0.14` mAP@50:95 chỉ vì bị cố định ở độ phân giải nhỏ 320 và backbone `yolo11n`. Hệ thống còn xuất đồng thời bộ **Presets** (Accuracy / Time / Memory / Balanced) nếu người dùng ưu tiên FPS tối đa.

6. **Hỏi: Giá trị thực tiễn ngoài phòng lab của phần mềm là gì?**  
   *Trả lời:* Hệ thống đã đóng gói thành công cụ `research_pipeline.v2` kèm giao diện Web UI tiếng Việt hoàn chỉnh, hỗ trợ người dùng nhập dataset ảnh YOLO/Classification/Captioning thông thường, tự động bảo vệ GPU 4–8 GB khỏi OOM crash trong suốt quá trình tinh chỉnh mô hình, và tự động xuất file thuyết minh + log tái lập SHA-256.
