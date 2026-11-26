# BÁO CÁO NGHIÊN CỨU KHOA HỌC SINH VIÊN 2027

**Đề tài:** *Tối ưu hóa Cấu hình Huấn luyện Mô hình Thị giác Máy tính dưới Ràng buộc Ngân sách Thời gian và Bộ nhớ GPU bằng Phương pháp Tìm kiếm Lai Đa tầng Hiệu chỉnh Conformal (`Conformal-MF`) và Quy tắc $\varepsilon$-Constraint*

**Hệ thống thực nghiệm:** `yolo-budget-tuner / research_pipeline.v2` (`1.0.0`)  
**Mã nguồn công khai:** `https://github.com/Tungdota53/NHCKVANH-2027`

---

## TÓM TẮT (ABSTRACT)

Trong bối cảnh triển khai các mô hình học sâu thị giác máy tính (YOLO Object Detection và CNN Image Classification) trên phần cứng phổ thông có giới hạn tài nguyên (như GPU laptop 6 GB VRAM dưới cơ chế bộ nhớ dùng chung WDDM của Windows), việc chọn cấu hình huấn luyện (`kiến trúc mạng`, `kích thước ảnh`, `batch size`, `precision`, `tốc độ học`, `độ sâu đóng băng`) dưới ngân sách thời gian hạn hẹp gặp ba rào cản lớn: (1) thử nghiệm mù các cấu hình vượt ngưỡng VRAM gây tràn bộ nhớ hoặc suy giảm tốc độ nghiêm trọng do WDDM spill, làm lãng phí ngân sách; (2) tuyến tính hóa hàm mục tiêu đa tiêu chí làm thuật toán hy sinh quá mức độ chính xác (`mAP@50`) để đổi lấy tốc độ/VRAM; và (3) chiến lược đa tầng tĩnh có thể kém hiệu quả trên các miền không có rủi ro phần cứng (như phân loại ảnh CPU).

Nghiên cứu này đề xuất và kiểm chứng hệ thống tối ưu cấu hình có kiểm toán toàn trình gồm: **(i) Bộ lọc tiên nghiệm `ConformalVRAMGuard`** với hiệu chỉnh bất đối xứng split-conformal giúp đạt **100% tỷ lệ ngăn ngừa tràn/WDDM spill** trên tập hiệu chuẩn GPU thực tế; **(ii) Thuật toán `Risk-Gated conformal_mf`** kết hợp loại trừ Conformal trước khi chạy, thăm dò đa tầng ngắn có giới hạn (`2` điểm Giai đoạn 1, `mf_probe_fraction ∈ [0.18, 0.25]`), vòng lặp GP LOOCV xác nhận toàn phần Giai đoạn 2, và cơ chế tự chuyển hướng sang `TPE-BO` (`risk_gated_tpe_handoff`) khi bài toán không có card/ràng buộc VRAM; **(iii) Quy tắc chọn lọc $\varepsilon$-Constraint** giữ sai số chất lượng trong dải $\Delta Q \le \varepsilon$ trước khi tối ưu tốc độ và bộ nhớ. Thực nghiệm đối chứng ghép cặp trên **3 miền độc lập** với `n = 8` hạt giống tìm kiếm mỗi miền (`144` lượt chạy online, `36` cấu hình tham chiếu vét cạn) chứng minh `conformal_mf` đứng **hạng #1 trên cả 3/3 miền** (`U Regret = 0.00000` trên cả 2 tập Object Detection với `0.00` lần vi phạm VRAM và `0.00s` lãng phí; `U Regret = 0.00781` trên CPU CIFAR-10). Mô hình chọn lọc trên tập `african_wildlife_full` (`1.036` ảnh train / `216` ảnh val / `211` ảnh test) đạt **`Test mAP@50 = 0.9528 ± 0.0004`** và **`159.05 FPS`** trên GPU RTX 3050 6GB.

---

## 1. ĐẶT VẤN ĐỀ & MỤC TIÊU NGHIÊN CỨU

1. **Bối cảnh thực tế:** Trên các máy tính cá nhân và thiết bị biên (`edge/laptop GPU`), người dùng thường chỉ có **100–200 giây** để dò cấu hình huấn luyện phù hợp cho một tập dữ liệu mới, đồng thời chỉ có giới hạn VRAM thực dụng an toàn thấp hơn dung lượng danh định của card (do hệ điều hành và cơ chế WDDM shared memory gây tụt hiệu năng đột ngột trước khi kích hoạt ngoại lệ `CUDA OutOfMemoryError`).
2. **Hạn chế của các phương pháp hiện có:**
   - **Cấu hình mặc định (`default` / `c00`):** Chỉ tối ưu khi bộ siêu tham số mặc định ngẫu nhiên khớp với dữ liệu, dễ tụt hạng nghiêm trọng khi dữ liệu mới đòi hỏi kiến trúc hoặc tốc độ học khác (`0.0%` tỷ lệ trúng tối ưu trên cả 3 miền trong bộ benchmark đối chứng).
   - **Tìm kiếm ngẫu nhiên (`random`) & Bayesian chuẩn (`tpe_bo`):** Không có tri thức trước về ngưỡng VRAM thực thi của từng tổ hợp `(model, imgsz, batch, fp16/fp32)`, nên thường sa lầy `40.98s – 65.84s` vào các cấu hình vi phạm VRAM hoặc vượt ngân sách.
   - **Đa tầng không hiệu chỉnh (`mf_no_conformal`):** Dù rút ngắn thời gian từng lần thử nhờ ít epoch hơn, vẫn tiêu tốn `21.95s – 29.06s` cho các lần thử vi phạm VRAM và dễ bị nhiễu thứ hạng khi chuyển sang tác vụ CPU mạng nhẹ.
3. **Mục tiêu đề tài:** Thiết kế khung tối ưu cấu hình dưới ngân sách có chứng nhận kiểm toán, loại bỏ rủi ro lãng phí do tràn VRAM, bảo vệ chất lượng $\varepsilon$-Constraint, đạt hiệu quả thống kê vượt trội trên đa miền bài toán và hoàn thiện giao diện Web UI trực quan cho người dùng cuối.

---

## 2. CƠ SỞ TOÁN HỌC & PHƯƠNG PHÁP ĐỀ XUẤT

### 2.1. Bài toán tối ưu có ràng buộc ngân sách & tài nguyên

Cho không gian cấu hình $\mathcal{C} = \{c_0, c_1, \dots, c_{K-1}\}$, ngân sách thời gian thực $B > 0$, giới hạn bộ nhớ GPU $M_{\max} > 0$, hàm chất lượng thực trên toàn bộ dữ liệu/số epoch chuẩn $Q(c) \in [0, 1]$ (ví dụ `mAP@50` hoặc `Top-1 Accuracy`), thời gian huấn luyện $T(c)$ và mức đỉnh VRAM $V(c)$. Tập cấu hình khả thi là:
$$\mathcal{C}_{\text{feas}} = \{c \in \mathcal{C} \mid T(c) \le B,\ V(c) \le M_{\max}\}$$
Hàm mục tiêu tổng hợp chuẩn hóa $U(c)$ kết hợp độ đo chất lượng, tốc độ suy luận/huấn luyện và tiêu thụ VRAM, trong khi độ đo đánh giá thuật toán tìm kiếm là **sai số hối tiếc mục tiêu** $\text{Regret}_U = U^* - U(\hat{c}^*)$ và **sai số hối tiếc chất lượng** $\text{Regret}_Q = \max_{c \in \mathcal{C}_{\text{feas}}} Q(c) - Q(\hat{c}^*)$ trên chứng nhận toàn phần.

### 2.2. Bộ lọc tiên nghiệm `ConformalVRAMGuard` (Split-Conformal)

Từ $N_{\text{cal}}$ mẫu thăm dò thực nghiệm có đỉnh VRAM đo được $V_i$ và ước lượng cơ sở tĩnh $\hat{\mu}(c_i)$, ta tính phần dư lệch trên bất đối xứng trên tập hiệu chuẩn $\mathcal{D}_{\text{cal}}$:
$$r_i = V_i - \hat{\mu}(c_i), \qquad q_{\text{add},\alpha} = \text{Quantile}_{(1-\alpha)\left(1 + \frac{1}{|\mathcal{D}_{\text{cal}}|}\right)}(\{r_i\}_{i \in \mathcal{D}_{\text{cal}}})$$
Cận trên dự đoán bảo toàn với mức tin cậy $1 - \alpha$ cho cấu hình mới $c$ là:
$$U_\alpha(c) = \hat{\mu}(c) + q_{\text{add},\alpha}$$
Mọi cấu hình có $U_\alpha(c) > M_{\max}$ được gắn nhãn **lọc Conformal (`Conformal prune`)** và loại khỏi không gian tìm kiếm trước khi khởi tạo tiến trình huấn luyện thật, tiết kiệm toàn bộ thời gian `OOM/spill`.

### 2.3. Thuật toán lai có cổng rủi ro `Risk-Gated conformal_mf`

Để khắc phục nhược điểm "một công thức cho mọi bài toán" của đa tầng thuần túy, `conformal_mf` ([scheduler.py](research_pipeline/v2/scheduler.py)) thực thi theo 3 bước:
1. **Cổng rủi ro & Phân luồng (`Risk Gate`):** Kiểm tra điều kiện có chứng chỉ Conformal và giới hạn `--max-memory`. Nếu **không** tồn tại ràng buộc bộ nhớ phần cứng (`card is None` hoặc `max_memory <= 0`), thuật toán ghi nhận sự kiện `risk_gated_tpe_handoff` và chuyển toàn bộ ngân sách sang vòng lặp **`TPE-BO`** trên không gian đầy đủ.
2. **Giai đoạn 1 — Thăm dò rút gọn có giới hạn (`Stage 1 Fidelity Probe`):** Trên không gian còn lại $\mathcal{C}_{\text{safe}} = \mathcal{C} \setminus \mathcal{C}_{\text{pruned}}$, chọn tối đa `max_stage1_probes = 2` cấu hình (điều chỉnh số epoch theo `mf_probe_fraction` giới hạn trong `[0.18, 0.25]`) để thu thập tín hiệu chất lượng và chi phí ban đầu trong quỹ thời gian nhỏ nhất.
3. **Giai đoạn 2 — Xác nhận toàn phần hướng dẫn bởi GP LOOCV (`Stage 2 Adaptive Full-Confirmation`):** Sử dụng mô hình thay thế **Gaussian Process với Leave-One-Out Cross-Validation (`GP LOOCV`)** (`rank_candidates`) kết hợp điểm thăm dò đã thu thập để chọn tuần tự các ứng viên triển vọng nhất trên $\mathcal{C}_{\text{safe}}$ và huấn luyện toàn phần cho tới khi đạt ngưỡng ngân sách.

### 2.4. Quy tắc chọn lọc cuối $\varepsilon$-Constraint

Thay vì lấy $\arg\max_c U(c)$ thuần túy vốn dễ ưu tiên cấu hình rất nhẹ nhưng hụt độ chính xác, quy tắc `epsilon_constraint` ([scheduler.py](research_pipeline/v2/scheduler.py)) xác định tập ứng viên bảo toàn chất lượng:
$$\mathcal{C}_\varepsilon = \{c \in \mathcal{C}_{\text{done,full}} \mid Q_{\max} - Q(c) \le \varepsilon\}$$
và chỉ chọn cấu hình có $U(c)$ lớn nhất bên trong $\mathcal{C}_\varepsilon$, nhờ đó cân bằng chặt chẽ giữa chất lượng mô hình và hiệu năng phần cứng.

---

## 3. KIẾN TRÚC HỆ THỐNG & CÁC BẤT BIẾN KIỂM TOÁN

* **Tách lớp bảo toàn di sản (`0.10` / `yolo_tuner`):** Gói lõi tham chiếu `yolo_tuner/` (19 tệp) và định nghĩa chuẩn `0.10` ([algorithm_010_studies.json](configs/algorithm_010_studies.json), SHA-256 `d246015b...`) giữ nguyên bất biến 100%, không bị chỉnh sửa trong quá trình phát triển `v2`.
* **Chứng nhận toàn phần (`Full-Data / Full-Horizon / Full-Seed`):** Chỉ những cấu hình hoàn thành đầy đủ epoch chuẩn, đủ dữ liệu và đủ hạt giống mới đủ điều kiện tranh hạng chung cuộc (`finalists`). Lượt chạy bị ngắt/thất bại giữ `regret = null` trong xuất báo cáo.
* **Đồng bộ chuẩn đánh giá (`Detection mAP@50`):** Cưỡng chế `rect=False` trên cả `val` và `test`, kiểm toán đồng nhất `best_checkpoint_epoch == best_epoch`, và tách biệt hoàn toàn thư mục đánh giá `Test` (`evaluate-v2`) khỏi quá trình tối ưu siêu tham số.
* **Giao diện người dùng Web UI (`research_pipeline/v2/web_ui`):** Cung cấp 6 màn hình chuyên biệt bằng tiếng Việt có bảo vệ CSRF trên mọi `POST`, tự động nạp danh sách trọng số `.pt` cục bộ và thẻ hiệu chuẩn `configs/conformal_vram_card_*.json`, hiển thị trước dòng lệnh CLI và các cột an toàn VRAM (`Vi phạm VRAM TB`, `Lãng phí VRAM (s)`, `Conformal lọc`) trên màn hình đối chứng Benchmark.

---

## 4. THIẾT LẬP THỰC NGHIỆM & KẾT QUẢ

### 4.1. Thực nghiệm hiệu chuẩn `ConformalVRAMGuard` (Study 2 — RTX 3050 6GB)

Từ `28` điểm thăm dò thực tế trên GPU `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`vram_free_mb = 5161.0`, `vram_total_mb = 6143.5`), thu được `3` sự kiện tràn bộ nhớ WDDM spill ([summary.json](runs/nckh_conformal_vram_study/summary.json), biểu đồ [conformal_vram_calibration.svg](runs/nckh_conformal_vram_study/conformal_vram_calibration.svg)):
* Trên `50` lần chia lặp ngẫu nhiên (`13 fit / 8 cal / 7 test`, `α ∈ {0.05, 0.10}`, `q_add = 1047.22 MiB`):
  * **Tỷ lệ ngăn OOM/spill đạt `100.0%` (`oom_prevention_rate = 1.0`)**.
  * Sai số dự đoán VRAM: `MAE = 364.21 MiB`, `RMSE = 485.87 MiB`, độ rộng khoảng TB `548.24 MiB`, độ phủ thực nghiệm **`88.86% ± 16.40%`**, tỷ lệ loại nhầm `25.23%`.
* Đánh giá **Leave-One-Model-Out (LOMO)** khi ẩn hoàn toàn một kiến trúc YOLO khỏi tập hiệu chuẩn:
  * Độ phủ Zero-shot: `100%` (`yolo11n`, `n=8`), `87.5%` (`yolo11s`, `n=8`), `83.33%` (`yolo26n`, `n=6`).
  * Hiệu chỉnh 2 điểm neo (`calibrated_2pt`): nâng độ phủ lên `100%` trên `yolo11s` và `yolo26n`, nhưng `yolo11n` đạt `75.0%` do chỉ dùng `2` neo chưa bao phủ hết độ dốc `batch × imgsz`.

### 4.2. Thực nghiệm đối chứng thuật toán 3 miền độc lập (`n = 8` search seeds)

Mỗi miền được đánh giá trên `8` hạt giống tìm kiếm `[17, 29, 43, 59, 73, 89, 101, 127]` so sánh `6` thuật toán (`conformal_mf`, `mf_no_conformal`, `adaptive_full`, `tpe_bo`, `random`, `default`), xuất từ `algorithm-report` ([ALGORITHM_STUDY.md](runs/nckh_2027_final_algorithm_report/ALGORITHM_STUDY.md)) và `paper-export` ([tables.tex](runs/paper_exports/nckh_2027_final_3domain_vi/tables.tex)). Hai miền Detection dùng lưới [nckh_non_default_opt_grid.json](configs/nckh_non_default_opt_grid.json) (khởi điểm `c00` không tối ưu, `c07..c11` dùng `fp32` vượt `1800 MiB`); miền Classification dùng `--space core` trên CPU.

| Miền thực nghiệm (ID) | Ngân sách / Vét cạn | Thuật toán #1 | Tỷ lệ khả thi | U Regret ± ĐLC | Q Regret TB | Trúng tối ưu | Vi phạm VRAM / Lãng phí | Kiểm định ghép cặp vs `default` |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. `african_wildlife_benchmark`** (`5117f0c3feebb51d`, Detection) | `150s` / `526.47s` (`28.5%`) | **`conformal_mf`** | **8/8** | **`0.00000 ± 0.00000`** | **`0.00000`** | **100.0%** | **`0.00` / `0.00s`** (lọc `5.0`) | Wilcoxon & SF `p=0.0078`, Holm `p=0.0312` |
| **2. `penn_fudan_research`** (`9d8ae3860c9f3d71`, Detection) | `125s` / `431.52s` (`29.0%`) | **`conformal_mf`** | **8/8** | **`0.00000 ± 0.00000`** | **`0.00172`** | **100.0%** | **`0.00` / `0.00s`** (lọc `5.0`) | Wilcoxon & SF `p=0.0078`, Holm `p=0.0312` |
| **3. `cifar10_research`** (`833f3cb3052b5a98`, Classification CPU) | `50s` / `138.92s` (`36.0%`) | **`conformal_mf`** *(Risk-Gated)* | **8/8** | **`0.00781 ± 0.00865`** | **`0.01500`** | **50.0%** | `0.00` / `0.00s` | Wilcoxon & SF `p=0.0156`, Holm `p=0.0625` (vs `random` SF `p=0.0391`) |

**So sánh chi tiết với các đường cơ sở quan trọng:**
* Trên `african_wildlife_benchmark`: `mf_no_conformal` có U Regret `0.02454` và lãng phí **`29.06s`** (`1.25` lần vi phạm VRAM/hạt); `tpe_bo` có U Regret `0.10285` (`50%` opt hit, lãng phí **`42.18s`**); `random` chỉ đạt `7/8` khả thi, U Regret `0.10472`, lãng phí **`65.84s`**; `default` cố định ở `0.19631` (`0%` opt hit).
* Trên `penn_fudan_research`: `tpe_bo` và `random` chỉ đạt `50%` opt hit (`U Regret = 0.03471` và `0.03374`), đồng thời lãng phí lần lượt **`40.98s`** và **`53.72s`** cho các lần vi phạm VRAM; `conformal_mf` đạt `100%` opt hit và `0.00s` lãng phí.
* Trên `cifar10_research`: `tpe_bo` (`0.01125`), `mf_no_conformal` (`0.01378`), `random` (`0.01529`), `default` (`0.01752`) và `adaptive_full` (`0.02248`) đều xếp sau `conformal_mf` (`0.00781`). Cơ chế `risk_gated_tpe_handoff` chứng minh tính cần thiết khi không dùng ràng buộc phần cứng.

### 4.3. Huấn luyện quy mô đầy đủ & Đánh giá tập Test độc lập (Study 3)

Trên tập `african_wildlife_full` (`1,036` train / `216` val / `211` test, `12` epoch, `2` seed `[17, 29]`, `[AUDIT PASS]` với `11` worker):
* Quy tắc **`epsilon_constraint (0.03)`** lựa chọn cấu hình **`c00`** (`Val mAP@50 = 0.9237 ± 0.0088`, `442.0 MiB`) thay cho `c01` (vốn có `U = 0.3550` cao hơn `0.3300` nhưng `Val mAP@50` tụt xuống `0.8846`, mất **`3.91%`**).
* Kết quả đánh giá trên **tập Test giữ lại (`211` ảnh độc lập)** trong [EVALUATION.md](runs/nckh_2027_study3_wildlife_full/20261126_013218_819548/evaluation/20261126_015534_759413/EVALUATION.md):
  * **`Test mAP@50 = 0.9528 ± 0.0004`** (`0.952761`: seed 17 `0.9525`, seed 29 `0.9530`, biên độ Val→Test `+0.0291`).
  * `Test mAP@75 = 0.8646 ± 0.0010` | `Test mAP@50:95 = 0.7677 ± 0.0030` | `Test F1 = 0.9093`.
  * Tốc độ suy luận đo tại `batch=4, fp32, rect=False`: **`6.299 ms/ảnh` (`159.05 FPS`)**.

---

## 5. RỦI RO HIỆU LỰC & HƯỚNG PHÁT TRIỂN (THREATS TO VALIDITY)

1. **Giới hạn suy diễn thống kê:** Bộ kiểm định ghép cặp (`n = 8` hạt giống tìm kiếm mỗi miền) đánh giá biến thiên của thuật toán tìm kiếm trên cùng lưới và phần cứng mục tiêu; chưa phải kiểm định khái quát hóa xuyên phần cứng trên nhiều dòng GPU khác nhau.
2. **Giới hạn thực nghiệm GPU đơn:** Hiệu chuẩn Conformal mới được đo trực tiếp trên `RTX 3050 6GB` (Windows WDDM). Để áp dụng sang các GPU khác (`RTX 4060`, `T4`, `A100`), người dùng cần chạy lại `scripts/run_conformal_vram_study.py` để tạo thẻ hiệu chuẩn riêng và cân nhắc dùng `> 2` điểm neo nếu muốn LOMO đạt 100% ở mọi biến thể nhẹ.
3. **Độ trễ thăm dò ở kịch bản không có rủi ro VRAM:** Nếu tập ứng viên hoàn toàn nhẹ và an toàn, `adaptive_full` có thể hội tụ nhanh hơn `conformal_mf` khoảng `~19s` vì không cần chạy 2 lượt thăm dò Giai đoạn 1.

---

## 6. KẾT LUẬN

Nghiên cứu đã xây dựng và kiểm chứng thành công quy trình tối ưu cấu hình huấn luyện thị giác máy tính trong ràng buộc ngân sách **`conformal_mf` + `epsilon_constraint`**. Bằng chứng thực nghiệm trên 3 miền độc lập cho thấy phương pháp loại bỏ **100% thời gian lãng phí do vi phạm VRAM**, duy trì **100% độ khả thi**, đứng **hạng #1 trên cả 3 miền**, và mang lại mô hình phát hiện đối tượng đạt **`95.28% Test mAP@50`** ở **`159.05 FPS`**, sẵn sàng cho triển khai nghiên cứu và ứng dụng qua CLI lẫn Web UI tiếng Việt.
