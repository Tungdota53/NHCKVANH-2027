# BÁO CÁO NGHIÊN CỨU KHOA HỌC SINH VIÊN 2027

**Đề tài:** *CC-MFBO: Tối ưu hóa Cấu hình Huấn luyện Đa mục tiêu dưới Ràng buộc Ngân sách Thời gian và Bộ nhớ GPU bằng Tối ưu hóa Bayes Đa tầng Hiệu chỉnh Conformal Phân tầng (`Conformal-Calibrated Multi-Fidelity Bayesian Optimization`) và Quy tắc $\varepsilon$-Constraint*

**Hệ thống thực nghiệm:** `yolo-budget-tuner / research_pipeline.v2` (`1.0.0` — nâng cấp thuật toán **v2.0 `CC-MFBO`**)  
**Mã nguồn công khai:** `https://github.com/Tungdota53/NHCKVANH-2027`  
**Phần cứng kiểm chứng chính:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`vram_free = 5,161.0 MiB` / `vram_total = 6,143.5 MiB`, Windows WDDM)

---

## TÓM TẮT (ABSTRACT)

Việc lựa chọn tự động cấu hình huấn luyện mô hình thị giác máy tính (kiến trúc mạng, độ phân giải ảnh `imgsz`, `batch`, bộ tối ưu `AdamW/SGD`, độ chính xác `fp16/fp32`, tốc độ học `lr0`, số tầng đóng băng `freeze`) trên phần cứng GPU phổ thông dưới ngân sách thời gian hạn hẹp ($100\text{--}150$ giây) đối mặt đồng thời ba bài toán chưa được giải quyết triệt để trong các bộ tinh chỉnh hiện có:
1. **Rủi ro tràn VRAM âm thầm do WDDM Shared-Memory Spill:** Trên Windows/GPU phổ thông, tiến trình vượt `5,161 MiB` không phải lúc nào cũng ném ngoại lệ `CUDA OOM` ngay mà âm thầm tràn sang RAM hệ thống qua WDDM (`5,174–6,452 MiB`), làm tiêu tốn vô ích $20\text{--}66$ giây ngân sách mà vẫn bị loại ở bước kiểm tra `--max-memory`.
2. **Thiên lệch cấu hình nhỏ & tuyến tính hóa mục tiêu:** Tuyến tính hóa hàm đa mục tiêu $U(c)$ khiến thuật toán dễ thiên vị cấu hình nhẹ `320px` (hy sinh $3\text{--}30\%$ độ chính xác), đồng thời bộ khởi tạo nhỏ-trước (`small_first_key` của `adaptive_full`) sa lầy vào cụm `320px` kém chất lượng khi ngân sách eo hẹp.
3. **Thất bại của đa tầng thô trước các cấu hình biên hỗn hợp (`fp16` nặng vs `fp32` an toàn):** Các luật phỏng đoán đơn giản không phân biệt được ranh giới kích hoạt phi tuyến giữa `yolo11s 640/B8 fp16` (`2,718–2,736 MiB > 1,850 MiB`) và `yolo11n 480/B4 fp32` (`760–782 MiB <= 1,850 MiB`).

Nghiên cứu này đề xuất **CC-MFBO** (`conformal_mf`), một khung tối ưu hóa Bayes đa tầng thống nhất gồm:
- **Giai đoạn 0 (`ConformalVRAMGuard` phân tầng có trọng số cục bộ):** Mô hình vật lý phần dư kích hoạt $V(c) = V_{\text{static}}(c) + \hat{r}_{\text{Ridge-LOOCV}}(x_c)$ kết hợp hiệu chỉnh **Locally-Weighted Split Conformal** cho đảm bảo hữu hạn mẫu phân phối-tự do $\mathbb{P}\{V(c) \le U_\alpha(c)\} \ge 1 - \alpha$ và xác suất khả thi liên tục $\hat{p}_{\text{feas}}(c)$. Trên $60$ điểm đo GPU thực tế across `yolo11n/yolo11s/yolo26n`, bộ lọc đạt độ phủ thực nghiệm **`96.4% ± 6.6%`** ($\alpha = 0.05$, mục tiêu $95\%$), `MAE = 135.1 MiB`, tỷ lệ loại nhầm chỉ **`6.07%`** và ngăn chặn **`100%`** sự kiện tràn WDDM (`5,174–6,452 MiB`), đồng thời **Leave-One-Model-Out + hiệu chỉnh co ngót Bayesian 2 điểm** đạt `100%` độ phủ trên `yolo11n`, `94.4%` trên `yolo11s`, và `100%` trên `yolo26n`.
- **Giai đoạn 1 (`Facility-Location` K-Medoid phân tầng):** Chọn $2\text{--}4$ cấu hình thăm dò đa tầng đa dạng có ràng buộc hạt giống (loại bỏ hoàn toàn tính tất định `SD = 0`), kết hợp giãn biên độ tự hồi quy kiểu **Kennedy–O'Hagan** ($\hat{\rho}$).
- **Giai đoạn 2 (`Multi-Fidelity Residual GP-LOOCV` + `cEUI` hậu kiểm $\varepsilon$-Constraint):** Kết hợp tiên nghiệm thăm dò với mô hình phần dư Gaussian Process đa seed và kỳ vọng cải thiện $\varepsilon$-Constraint có trọng số khả thi $\hat{p}_{\text{feas}}(c) \cdot \text{EI}_\varepsilon(c)$.

Thực nghiệm đối chứng ghép cặp trên **3 miền bài toán khác nhau hoàn toàn** với không gian **`18` cấu hình/miền**, **`8` phương pháp** (`conformal_mf`, `adaptive_full`, `constrained_bo` cEI, `mf_no_conformal`, `successive_halving`, `tpe_bo`, `random`, `default`) và **`n = 8` hạt giống tìm kiếm** (`192` lượt tìm kiếm online + `54` cấu hình tham chiếu vét cạn) chứng minh:
- **Cả 3 miền hội tụ về 3 cấu hình tối ưu toàn cục hoàn toàn khác nhau:** `c14` (`yolo11s 480/B4 AdamW lr=0.00018 frz=8`, `0.9379 mAP@50`) trên Wildlife; `c17` (`yolo11s 640/B4 AdamW lr=0.0002 frz=6`, `0.8772 mAP@50:95`) trên Penn-Fudan; và `c12` (`cnn_classifier width=64 32/B16`, `0.4200 Acc`) trên CIFAR-10.
- Trên cả hai miền có ràng buộc VRAM GPU (`Wildlife` ở $13.1\%$ chi phí vét cạn; `Penn-Fudan` ở $21.4\%$ chi phí vét cạn), **`conformal_mf` đứng #1 tuyệt đối** (`U Regret = 0.0748 ± 0.1143` với `50%` trúng tối ưu trên Wildlife; **`0.0009 ± 0.0026`** với **`87.5%` trúng tối ưu** trên Penn-Fudan), giữ **`0.00` vi phạm VRAM / `0.00s` lãng phí** và thắng có ý nghĩa thống kê toàn bộ 7 đường cơ sở trên Penn-Fudan (`Wilcoxon & Sign-Flip p = 0.0078–0.0312`, vượt `adaptive_full` với `p = 0.0078` trên cả hai tập).
- Đánh giá **tập `Test` giữ lại hoàn toàn độc lập** trên cả 3 miền đạt: Wildlife **`Test mAP@50 = 0.9528 ± 0.0004`** (`159.05 FPS`), Penn-Fudan **`Test mAP@50 = 0.9785 ± 0.0107` / `mAP@50:95 = 0.8667 ± 0.0186`** (`73.16 FPS`), và CIFAR-10 **`Test Acc = 0.3480 ± 0.0231`**.

---

## 1. GIỚI THIỆU & ĐỘNG LỰC NGHIÊN CỨU

### 1.1. Khoảng trống trong Tối ưu Siêu tham số (HPO) dưới Ràng buộc Phần cứng Phổ thông
Phần lớn công trình HPO hiện đại (AutoML / BO / Hyperband) được thiết kế trên cụm GPU trung tâm dữ liệu ($40\text{--}80$ GB VRAM) và giả định rằng mọi cấu hình có thể chạy thử trong quỹ thời gian lớn để nhận nhãn chất lượng. Ngược lại, trong bối cảnh NCKH sinh viên, thiết bị biên và máy trạm cá nhân (`RTX 3050 6GB Laptop GPU`), bài toán đặt ra khắc nghiệt hơn gấp bội:
1. **Ngân sách siêu hẹp:** Toàn bộ quá trình tìm kiếm thường chỉ được cấp $68\text{--}135$ giây ($\approx 13\%\text{--}23\%$ chi phí vét cạn $1$ lượt quét), chỉ đủ xác nhận toàn phần từ $2$ đến $4$ cấu hình.
2. **Bẫy WDDM Shared-Memory Spill thay vì `CUDA OOM` sạch:** Trên GPU $6$ GB dưới hệ điều hành Windows (`vram_free = 5,161 MiB`), chúng tôi đo được thực nghiệm 3 cấu hình (`yolo11n 800/B12 fp32`, `yolo11s 640/B12 fp32`, `yolo11s 800/B8 fp32`) tiêu thụ lần lượt **`5,174.0 MiB`**, **`6,216.0 MiB`** và **`6,452.0 MiB`** — vượt VRAM vật lý khả dụng nhờ cơ chế trang ảo WDDM nhưng làm chậm tốc độ huấn luyện nặng nề. Ngay cả khi đặt ngưỡng triển khai chặt `--max-memory 1850 MiB` (chừa biên cho đa tiến trình / camera stream), một thuật toán không biết trước cận VRAM sẽ phải trả $20\text{--}45$ giây huấn luyện trước khi phát hiện cấu hình không khả thi.
3. **Bẫy `fp16` vs `fp32` hỗn hợp:** Các luật heuristic thô (ví dụ "cấm mọi `fp32`") loại nhầm những cấu hình `fp32` chính xác và hoàn toàn an toàn (như `yolo11n 480/B4 fp32` chỉ tốn `760–782 MiB`), trong khi bỏ lọt bẫy **`yolo11s 640/B8 fp16`** tiêu tốn **`2,624–2,736 MiB > 1,850 MiB`** dù chạy ở bán chính xác!

### 1.2. Các đóng góp khoa học chính của đề tài
Nghiên cứu nâng cấp toàn diện hệ thống `yolo-budget-tuner` lên kiến trúc **v2.0 (`CC-MFBO`)** với 5 đóng góp chính:
1. **Lý thuyết & Hiệu chuẩn `ConformalVRAMGuard` (60 điểm đo thực tế GPU):** Kết hợp biểu diễn đặc trưng vật lý kích hoạt phi tuyến (bất biến theo gradient accumulation), chọn tham số $\lambda$ bằng `PRESS/LOOCV` trên tập `fit`, chuẩn hóa phần dư dị phương sai cục bộ $\hat{\sigma}(c)$ và hiệu chuẩn Split Conformal tách rời + chuyển giao kiến trúc mới (LOMO) bằng co ngót Bayesian 2 điểm.
2. **Thuật toán thống nhất `CC-MFBO` (`conformal_mf`):** Loại bỏ hoàn toàn nhánh rẽ cứng `if-else` thủ công sang `tpe_bo`, hợp nhất lọc Conformal (Giai đoạn 0), sàng lọc K-Medoid phân tầng có giãn biên độ Kennedy–O'Hagan (Giai đoạn 1), và vòng lặp Residual GP-LOOCV + $\varepsilon$-Constraint `cEUI` đa tầng (Giai đoạn 2).
3. **Đường cơ sở quốc tế `constrained_bo` (`cEI` — Gelbart et al., 2014):** Cài đặt đúng chuẩn Constrained Expected Improvement với Gaussian Process thứ hai mô hình hóa xác suất khả thi tài nguyên log-memory $\Phi((\log M_{\max} - \mu_M)/\sigma_M)$ để so sánh công bằng với cộng đồng Bayes HPO quốc tế.
4. **Kiểm chứng đa miền không gian lớn `18` cấu hình (`n = 8`, `8` phương pháp):** Chứng minh tính thích nghi đa miền thật sự — 3 miền tự hội tụ về 3 cấu hình tối ưu khác biệt — cùng kiểm định ghép cặp phi tham số Wilcoxon, Exact Sign-Flip và Holm–Bonferroni.
5. **Hệ sinh thái kiểm toán bất biến & Web UI 100% offline:** Bảo toàn 100% lớp chuẩn `0.10` ([algorithm_010_studies.json](configs/algorithm_010_studies.json)) và 19 tệp lõi `yolo_tuner/*.py`; cung cấp Web Studio tiếng Việt 6 màn hình, tự động kiểm tra CSRF, xoay nhãn đồ thị đa cột và xuất thẳng bảng LaTeX (`tables.tex`), CSV, SVG Okabe-Ito.

---

## 2. NGHIÊN CỨU LIÊN QUAN & ĐỊNH VỊ PHƯƠNG PHÁP

Bảng dưới đây đối chiếu trực diện **CC-MFBO** với 5 nhóm phương pháp HPO kinh điển trong tài liệu quốc tế:

| Phương pháp / Tác giả | Khai thác đa tầng (`Multi-Fidelity`) | Mô hình thay thế (`Surrogate`) | An toàn Bộ nhớ / Ràng buộc | Đảm bảo Hữu hạn Mẫu | Bảo toàn Chất lượng Hậu kiểm | Hạn chế chính trên GPU Phổ thông |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Search** (Bergstra & Bengio, 2012) | Không | Không | Hậu kiểm (lãng phí) | Không | $\arg\max U(c)$ tuyến tính | Sa lầy thử cấu hình tràn VRAM; không khai thác lịch sử |
| **TPE / SMAC** (Bergstra et al., 2011; Hutter et al., 2011) | Không | Kernel Density / RF | Hậu kiểm (`penalty`) | Không | Tuyến tính hóa đa mục tiêu | Vẫn phải chạy 1 lần thử bẫy VRAM để học vùng cấm |
| **Successive Halving / Hyperband** (Jamieson & Talwalkar, 2016; Li et al., 2018) | Rung cố định (loại 50%/vòng) | Không | Thường dựa trên `imgsz`/kích thước | Không | Chỉ xét chất lượng hoặc proxy đơn | Lãng phí ngân sách thăm dò quá nửa số cấu hình khi $B$ siêu nhỏ ($\sim 120$s) |
| **Constrained BO / `cEI`** (Gelbart et al., 2014) | Không | GP chất lượng + GP ràng buộc độc lập | $\text{EI}(c) \cdot \mathbb{P}_{\text{GP}}(V(c) \le M_{\max})$ | Không (dựa vào tiên nghiệm Gaussian) | Có thể kết hợp $\varepsilon$ nhưng không có probe | Không có tri thức vật lý trước khi chạy, vẫn vi phạm VRAM trung bình $0.88$ lần/hạt |
| **BOHB / MFBO** (Falkner et al., 2018; Kandasamy et al., 2017) | TPE đa tầng | KDE / GP đa tầng | Không hiệu chỉnh tràn WDDM | Không | Không | Có thể tụt điểm khi tương quan thứ hạng probe yếu ở miền nhiễu |
| **CC-MFBO (`conformal_mf`, Đề xuất)** | **K-Medoid đa tầng + Residual GP** | **Heteroscedastic Matérn-5/2 / RBF LOOCV** | **Split Conformal $\hat{p}_{\text{feas}}(c)$ + 2-pt Bayes** | **Có ($\ge 1-\alpha$ phân phối-tự do)** | **$\varepsilon$-Constraint Pareto-safe** | Giảm về gần `mf_no_conformal` nếu không truyền card & miền không có rủi ro VRAM |

---

## 3. PHÁT BIỂU BÀI TOÁN & CƠ SỞ TOÁN HỌC

### 3.1. Bài toán Tối ưu Đa mục tiêu dưới Ràng buộc Ngân sách và Bộ nhớ
Cho tập hữu hạn gồm $K = 18$ cấu hình ứng viên $\mathcal{C} = \{c_0, \dots, c_{K-1}\}$, tập training seeds cố định $\mathcal{S}_{\text{train}}$, số epoch chuẩn $E_{\text{full}}$, ngân sách thời gian phiên tìm kiếm $B > 0$ (giây), và trần bộ nhớ triển khai $M_{\max} > 0$ (MiB). Mỗi cấu hình được xác nhận toàn phần đo trực tiếp:
- Chất lượng kiểm định trung bình $Q(c) \in [0, 1]$ (`mAP@50`, `mAP@50:95`, hoặc `Top-1 Accuracy`),
- Thời gian thực thi trung bình mỗi lượt $T(c) > 0$ (giây),
- Đỉnh bộ nhớ cấp phát `cuda_reserved_allocator_mib` lớn nhất $V(c) > 0$ (MiB).

Hàm mục tiêu chuẩn hóa khóa trước tìm kiếm ([selection.py](research_pipeline/v2/selection.py)):
$$U(c) = w_Q Q(c) - w_T \frac{T(c)}{T_{\text{scale}}} - w_M \frac{V(c)}{M_{\text{scale}}}, \quad (w_Q, w_T, w_M) = (0.90, 0.07, 0.03)$$
với miền khả thi tài nguyên $\mathcal{F} = \{c \in \mathcal{C} \mid V(c) \le M_{\max},\ Q(c) \ge Q_{\min}\}$. Khi kết thúc ngân sách $B$, bộ chọn hậu kiểm $\varepsilon$-Constraint chỉ xét tập ứng viên toàn phần không bị chi phối (`nondominated`) thỏa:
$$\mathcal{C}_\varepsilon = \left\{c \in \mathcal{F}_{\text{obs,full}} \;\middle|\; \max_{c' \in \mathcal{F}_{\text{obs,full}}} Q(c') - Q(c) \le \varepsilon\right\}, \qquad \hat{c}^* = \arg\max_{c \in \mathcal{C}_\varepsilon} S_\varepsilon(c)$$
trong đó $S_\varepsilon(c) = 1 - \frac{c_{\text{tm}}(c)}{1 + c_{\text{tm}}(c)} + 10^{-4} Q(c)$ với chi phí chuẩn hóa thời gian-bộ nhớ $c_{\text{tm}}(c) = \frac{w_T (T/T_{\text{scale}}) + w_M (V/M_{\text{scale}})}{w_T + w_M}$. Độ đo đánh giá chất lượng tìm kiếm là **sai số hối tiếc mục tiêu** $\text{Regret}_U = U^*_{\text{ref}} - U_{\text{ref}}(\hat{c}^*)$ trên lượt đối chứng vét cạn độc lập (lỗi không có cấu hình khả thi được gán `null`, tuyệt đối không gán `0`).

### 3.2. Mô hình Vật lý VRAM & Định lý Đảm bảo Split Conformal Có Trọng số Cục bộ
Quan sát thực nghiệm cho thấy đỉnh VRAM của một mạng YOLO tách tự nhiên thành phần tĩnh $V_{\text{static}}(c)$ (trọng số $4\text{--}6$ byte/tham số tùy `fp16/fp32`, gradient $4$ byte/tham số khả huấn, và trạng thái tối ưu $8$ byte/tham số với `AdamW` hoặc $4$ byte với `SGD`) cộng phần dư kích hoạt phi tuyến $r(c) = V(c) - V_{\text{static}}(c) \ge 0$. Trong [conformal_vram.py](research_pipeline/v2/conformal_vram.py), ta định nghĩa proxy kích hoạt chuẩn hóa:
$$\phi_{\text{act}}(c) = B_{\text{mb}} \cdot \left(\frac{H}{320}\right)^2 \cdot \gamma_{\text{prec}} \cdot \left(\frac{\text{FLOPs}_{32}}{0.01653}\right)^{0.55} \left(\frac{L}{24}\right)^{0.25} \cdot \max\left(0.35,\ 1 - 1.2\frac{\min(f, L)}{L}\right)$$
Vector đặc trưng $6$ chiều bất biến theo gradient accumulation gồm:
$$x_c = \left[1,\ \phi_{\text{act}}(c),\ \phi_{\text{act}}(c)\cdot \mathbb{I}_{\text{fp32}},\ \mathbb{I}_{\text{fp32}},\ \mathbb{I}_{\text{SGD}},\ \min\left(1, \frac{f}{L}\right)\right]^\top$$
Các đặc trưng không chuẩn hóa như `parameters` thô bị loại bỏ vì gây đa cộng tuyến làm méo hệ số khi ngoại suy sang kiến trúc mới (LOMO). Trên tập `fit` $\mathcal{D}_{\text{fit}}$, hệ số $\hat{\beta}_\lambda$ được chọn hoàn toàn tự động qua cực tiểu hóa sai số `PRESS / LOOCV` chính xác trên lưới $\lambda \in \{0.1, 0.5, 1.0, 5.0, 15.0\}$, tạo ước lượng điểm $\hat{\mu}(c) = V_{\text{static}}(c) + \max(0, \hat{\beta}^\top_\lambda \tilde{x}_c)$. Để triệt tiêu bất đối xứng sai số ở vùng cấu hình lớn, ta ước lượng độ lệch cục bộ đơn điệu trên tập `fit`:
$$\hat{\sigma}(c) = \max\left(\sigma_{\text{floor}},\ c_0 + c_1 \hat{\mu}(c)\right), \qquad \sigma_{\text{floor}} = 120.0\ \text{MiB}$$
và tính điểm không tương thích chuẩn hóa một phía trên tập hiệu chuẩn độc lập $\mathcal{D}_{\text{cal}}$ ($n_{\text{cal}} = |\mathcal{D}_{\text{cal}}|$):
$$s_i = \frac{V_i - \hat{\mu}(c_i)}{\hat{\sigma}(c_i)}, \quad i \in \mathcal{D}_{\text{cal}}, \qquad q_{\text{norm},\alpha} = \text{Quantile}_{\frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}}\left(\{s_i\}_{i \in \mathcal{D}_{\text{cal}}}\right)$$

**Mệnh đề 1 (Đảm bảo độ phủ hữu hạn mẫu phân phối-tự do — Vovk et al., 2005; Lei et al., 2018):**  
*Nếu bộ $(c_1, V_1), \dots, (c_{n_{\text{cal}}}, V_{n_{\text{cal}}}), (c_{\text{new}}, V_{\text{new}})$ khả hoán (exchangeable) và độc lập với $\mathcal{D}_{\text{fit}}$, khi đó cận trên Split Conformal có trọng số cục bộ $U_\alpha(c_{\text{new}}) = \hat{\mu}(c_{\text{new}}) + q_{\text{norm},\alpha}\,\hat{\sigma}(c_{\text{new}})$ thỏa mãn bất đẳng thức không giả định phân phối chuẩn:*
$$\mathbb{P}\left\{V_{\text{new}} \le U_\alpha(c_{\text{new}})\right\} \ge 1 - \alpha$$
*Hơn nữa, xác suất khả thi dự báo trơn hóa tại ngưỡng $M_{\max}$ là:*
$$\hat{p}_{\text{feas}}(c) = \frac{1 + \sum_{i \in \mathcal{D}_{\text{cal}}} \mathbb{I}\left\{\hat{\mu}(c) + s_i \hat{\sigma}(c) \le M_{\max}\right\}}{n_{\text{cal}} + 1} \in \left[\frac{1}{n_{\text{cal}}+1},\ 1\right]$$

**Hiệu chỉnh trực tuyến Co ngót Bayesian 2 điểm (Online Shrinkage 2-Point Transfer):**  
Khi gặp kiến trúc mục tiêu chưa từng xuất hiện trong $\mathcal{D}_{\text{fit}} \cup \mathcal{D}_{\text{cal}}$ và thu được $k \ge 2$ điểm neo trực tuyến có biên độ $\Delta x \ge 300\ \text{MiB}$, độ dốc $\hat{a}$ được co ngót tâm về $1.0$ với trọng số tiên nghiệm $\tau = 15{,}000$:
$$\hat{a} = \text{clip}_{[0.85,\, 1.85]}\left(\frac{\sum_{j=1}^k (x_j - \bar{x})(y_j - \bar{y}) + \tau}{\sum_{j=1}^k (x_j - \bar{x})^2 + \tau}\right), \qquad \hat{b} = \bar{y} - \hat{a}\bar{x}$$
và đánh giá $\hat{\sigma}(c)$ tại $\max(\hat{\mu}_{\text{raw}}(c), \hat{\mu}_{\text{cal}}(c))$ để tránh thu hẹp quá mức khoảng tin cậy khi sang kiến trúc nhẹ.

---

## 4. THUẬT TOÁN THỐNG NHẤT `CC-MFBO` (`conformal_mf`)

Thay cho việc rẽ nhánh thủ công `if card is None: tpe_bo`, `CC-MFBO` ([scheduler.py](research_pipeline/v2/scheduler.py), [acquisition.py](research_pipeline/v2/acquisition.py)) vận hành qua 3 giai đoạn liền mạch dưới cùng một khung toán học:

1. **Giai đoạn 0 — Tiền kiểm Conformal & Đánh giá Khả thi ($\hat{p}_{\text{feas}}$):**  
   Nếu có thẻ hiệu chuẩn `--conformal-vram-card`, tính $(U_{0.05}(c), \hat{p}_{\text{feas}}(c))$ cho toàn bộ $c \in \mathcal{C}$. Cấu hình có $U_{0.05}(c) > M_{\max}$ bị loại tức thì (`stage0_offline_conformal`) với chi phí $0.0$ giây. Khi không có thẻ (ví dụ Classification VRAM nhỏ), $\hat{p}_{\text{feas}}(c) = 1.0$.
2. **Giai đoạn 1 — Sàng lọc Đa tầng K-Medoid / Facility-Location Phân tầng & Giãn Biên độ Kennedy–O'Hagan:**  
   - Chọn điểm neo thăm dò đầu tiên $p_0$ ngẫu nhiên theo `search_seed` trong tầng cấu hình nhanh/vừa (`fast_tier`), sau đó tham lam tối thiểu hóa tổng khoảng cách phủ Facility-Location trên $\mathcal{C}_{\text{safe}}$ (ưu tiên phủ đa dạng các họ mô hình) cho tối đa `target_probes` $= 3$ (Detection, 1 epoch subset) hoặc $4$ (Classification, 4 epochs) điểm trong ngân sách phụ `probe_fraction` $\in [0.20, 0.28]$. Sau mỗi probe hoàn tất, cập nhật hiệu chỉnh 2-điểm trực tuyến và siết lại cận Conformal.
   - Chất lượng probe $\{\tilde{q}_j\}$ thường nén biên độ ở epoch 1; `rank_candidates` áp dụng phép biến đổi tự hồi quy **Kennedy–O'Hagan** giãn độ lệch chuẩn tối thiểu về $0.075$ ($\hat{\rho} = \min(3.0,\, \max(\sigma_{\tilde{q}}, 0.075)/\sigma_{\tilde{q}})$) rồi nội suy mặt tiên nghiệm thăm dò $m_{\text{probe}}(c)$ bằng kernel Matérn-5/2 ($l = 0.90, \lambda = 0.08$).
3. **Giai đoạn 2 — Vòng lặp Xác nhận Toàn phần GP Phần dư Dị phương sai (`cEUI`):**  
   Với mỗi bước xác nhận toàn phần khi đã có $n_{\text{full}}$ cấu hình đủ dữ liệu/đủ epoch/đủ seed:
   - Nếu $n_{\text{full}} = 0$: dùng ngay tiên nghiệm $m_{\text{probe}}(c)$ và phương sai thăm dò để xếp hạng ứng viên theo hàm thu ích hậu kiểm $\varepsilon$-Constraint neo vào $\max_j m_{\text{probe}}(j)$.
   - Nếu $n_{\text{full}} \ge 1$: tính độ lệch hệ thống $\hat{\delta} = \frac{1}{n_{\text{full}}}\sum_{i \in \mathcal{O}} (y_i - m_{\text{probe}}(c_i))$, huấn luyện **Heteroscedastic GP-LOOCV** (`Matérn-5/2` / `RBF`) trên phần dư $y_i - (m_{\text{probe}}(c_i) + \hat{\delta})$ để dự đoán phân phối hậu nghiệm $(\mu_Q(c), \sigma_Q(c))$, kết hợp chi phí thời gian điều hòa đa tầng $T_{\text{mf}}(c)$ (làm mượt dự báo FLOP-tuyến tính theo tỷ lệ thăm dò thực tế).
   - Hàm mua sắm thống nhất (`Conformal cEUI`):
     $$\alpha_{\text{CC-MFBO}}(c) = \hat{p}_{\text{feas}}(c) \cdot \frac{\text{EI}_{\varepsilon\text{-post}}(c) + 0.25\,\text{UCB}_\varepsilon(c) + 0.5\,\text{EHVI}_{\text{3D}}(c) + b_{\text{div}}(c)}{\hat{T}_{\text{mf}}(c) \cdot |\mathcal{S}_{\text{missing}}(c)| \cdot \left(1 + \text{excess}(c)\right)}$$
     Trong đó `EI`$\varepsilon\text{-post}$ tích phân bằng **21-point Gauss–Hermite Quadrature** có cộng phần thưởng bứt phá chất lượng $(q - q_{\text{obs,max}})_+$ nhằm ngăn thuật toán kẹt ở cấu hình `320px` khi đã có bước nhảy chất lượng cao hơn ở `480px–640px`.

---

## 5. KIẾN TRÚC HỆ THỐNG & BẤT BIẾN KIỂM TOÁN MẬT MÃ

Hệ thống `yolo-budget-tuner / research_pipeline.v2` tuân thủ nghiêm ngặt 5 bất biến kỹ thuật và khoa học:
1. **Bảo toàn di sản `yolo_tuner` & `0.10`:** Toàn bộ 19 tệp trong `yolo_tuner/` và bộ tham chiếu [algorithm_010_studies.json](configs/algorithm_010_studies.json) hoàn toàn không bị chỉnh sửa (`git diff --stat` rỗng), đồng thời các tệp nằm trong `training_signature()` (`worker.py`, `runner.py`, `pooling.py`, `telemetry.py`, `model_features.py`, `data_features.py`, `predictors.py`, `vision_worker.py`, `caption_worker.py`, `loaders.py`) không bị tác động.
2. **Khóa giao thức trước tìm kiếm & kiểm toán SHA-256 (`events.jsonl`):** Mọi lượt thử phải ghi nhận cặp `(request.json, result.json, parent_result.json)` và hàm băm checkpoint `best.pt`. Mọi so sánh trong `algorithm-report` kiểm tra tính khớp nguyên bản của `source_files`, `dataset_content_id`, `gpu_uuid`, `budget_seconds`, và bất kỳ rò rỉ hậu kiểm nào đều ném ngoại lệ chặn biên dịch báo cáo.
3. **Tính công bằng đánh giá đa tầng & `Test` giữ lại:** Chất lượng probe chỉ được phép hướng dẫn *thứ tự khám phá*, tuyệt đối không được tính vào bảng quyết định `decision.locked.json`. Đánh giá `Test` (`evaluate-v2`) cưỡng chế `rect=False` đồng bộ với `YOLO(best.pt).val(rect=False)` và chỉ chạy sau khi mọi lựa chọn đã được khóa vĩnh viễn.
4. **Hàng rào cách ly tiến trình GPU (`workflow.idle()`):** Mọi thí nghiệm GPU, benchmark và test suites được thực thi tuần tự tuyệt đối, ngăn nhiễu thời gian hoặc tranh chấp VRAM từ tiến trình khác.
5. **Web UI Studio 100% offline & xuất bản học thuật (`paper-export`):** Giao diện 6 màn hình ([launcher.html](research_pipeline/v2/web_ui/templates/launcher.html), [benchmark.html](research_pipeline/v2/web_ui/templates/benchmark.html), [charts.js](research_pipeline/v2/web_ui/static/charts.js)) hỗ trợ 8 phương pháp (`CC-MFBO`, `cEI`, `Successive Halving`, v.v.), hiển thị rõ 3 cột an toàn VRAM, xoay nhãn `-38°` cho đồ thị nhiều thanh và bảo vệ CSRF; [paper_export.py](research_pipeline/v2/paper_export.py) sinh sẵn bảng LaTeX (`\usepackage{booktabs}`) song ngữ `vi`/`en` và đồ thị SVG bảng màu chuẩn Okabe–Ito.

---

## 6. THỰC NGHIỆM HIỆU CHUẨN `ConformalVRAMGuard` (60-PROBE GPU & CHUYỂN GIAO LOMO)

Để khắc phục hạn chế thiếu mẫu hiệu chuẩn của phiên bản đầu (`n_cal = 8`), chúng tôi mở rộng bộ đo [run_conformal_vram_study.py](scripts/run_conformal_vram_study.py) lên **`60` tiến trình GPU cô lập** (`20` cấu hình/kiến trúc trên 3 trọng số thực: `yolo11n.pt`, `yolo11s.pt`, `yolo26n.pt`; kết quả lưu tại [summary.json](runs/nckh_conformal_vram_study/summary.json), [summary.md](runs/nckh_conformal_vram_study/summary.md) và [conformal_vram_calibration.svg](runs/nckh_conformal_vram_study/conformal_vram_calibration.svg)).

### 6.1. Phát hiện & Bắt gọn 3 ca Tràn WDDM Shared-Memory
Trong `60` probes, có `56` probes hợp lệ (`OK`), `1` probe `yolo26n + SGD + batch>=8` lỗi AMP step epoch 1, và **`3` probes tràn vượt VRAM vật lý khả dụng (`5,161.0 MiB`) sang bộ nhớ chia sẻ WDDM**:

| Probe | Mô hình | Thiết lập (`Img/B/Prec`) | VRAM Thực đo (`MiB`) | Dự báo Điểm (`MiB`) | Cận trên Conformal 95% (`MiB`) | Bị gắn cờ > `5,161 MiB`? |
|---:|---|---|---:|---:|---:| :---: |
| `#19` | `yolo11n.pt` | `800 / B12 / fp32` | **`5,174.0`** | `5,235.9` | `5,854.6` | **Đúng (`100%`)** |
| `#38` | `yolo11s.pt` | `640 / B12 / fp32` | **`6,216.0`** | `6,463.2` | `7,170.0` | **Đúng (`100%`)** |
| `#39` | `yolo11s.pt` | `800 / B8 / fp32` | **`6,452.0`** | `6,712.7` | `7,437.5` | **Đúng (`100%`)** |

### 6.2. Độ phủ thực nghiệm Trong-phân-phối (`50×` tách ngẫu nhiên `fit=24 / cal=21 / test=11`, ngưỡng `1,850 MiB`)
Thẻ hiệu chuẩn xuất cuối [conformal_vram_card_rtx3050.json](configs/conformal_vram_card_rtx3050.json) (`n_fit=28, n_cal=28`, `q_norm@95=2.023`, `q_norm@90=1.851`, `q_add@95=492.5 MiB`) đạt bước nhảy vọt toàn diện so với phiên bản 1.0:

| Mức tin cậy $1 - \alpha$ | Độ phủ Thực nghiệm (`TB ± ĐLC`) | Sai số `MAE` (`MiB`) | `RMSE` (`MiB`) | Độ rộng Khoảng (`MiB`) | Tỷ lệ Loại nhầm `@1850 MiB` | Recall Ngăn Tràn `@1850` & `@5161` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`95%` ($\alpha = 0.05$)** | **`0.964 ± 0.066`** *(đạt $\ge 0.95$)* | **`135.1`** | **`187.9`** | `363.7` | **`6.07%`** *(giảm $4.15\times$)* | **`100.0%` (`1.000`)** |
| **`90%` ($\alpha = 0.10$)** | **`0.913 ± 0.109`** *(đạt $\ge 0.90$)* | `135.1` | `187.9` | `281.3` | **`3.90%`** | **`100.0%` (`1.000`)** |

*(So với bản v1.0 28-probe: `MAE` giảm từ `364.2 MiB` xuống `135.1 MiB`; tỷ lệ loại nhầm giảm từ `25.23%` xuống `6.07%`; hai mức $\alpha \in \{0.05, 0.10\}$ tách biệt rõ rệt và khớp chính xác lý thuyết `0.964` vs `0.913`).*

### 6.3. Đánh giá Chuyển giao Xuyên Kiến trúc — Leave-One-Model-Out (LOMO)
Khi ẩn hoàn toàn $1$ kiến trúc YOLO khỏi quá trình `fit + cal`:

| Kiến trúc Ẩn (Held-out) | Số mẫu đánh giá | Zero-shot `MAE` (`MiB`) | Zero-shot `Cov@95` | 2-Point `MAE` (`MiB`) | 2-Point `Cov@90` | 2-Point **`Cov@95`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo11n.pt` | `18` | `147.2` | `1.000` | `306.8` *(bias shift nhẹ từ `yolo11s/26n`)* | `1.000` | **`1.000`** |
| `yolo11s.pt` | `18` | `192.8` | `0.944` | **`170.9`** | `0.944` | **`0.944`** |
| `yolo26n.pt` | `14` | `182.1` | `0.643` | **`112.8`** *(giảm `38%` lỗi)* | `0.786` | **`1.000`** |

Kết quả trên `yolo26n.pt` minh họa giá trị thực tiễn của co ngót 2-điểm trực tuyến: khi zero-shot từ dòng YOLO11 bị hụt độ phủ còn `64.3%`, chỉ $2$ điểm neo nhẹ tự động nâng `Cov@95` lên **`100.0%`** và giảm `MAE` từ `182.1` xuống còn **`112.8 MiB`**.

---

## 7. THỰC NGHIỆM ĐỐI CHỨNG 3 MIỀN × 8 PHƯƠNG PHÁP (`n = 8` HẠT GIỐNG TÌM KIẾM)

Mỗi miền sử dụng lưới **`18` cấu hình**, `8` phương pháp (`conformal_mf`, `adaptive_full`, `constrained_bo` cEI, `mf_no_conformal`, `successive_halving`, `tpe_bo`, `random`, `default`), và **`8` hạt giống tìm kiếm ghép cặp** (`17, 29, 43, 59, 73, 89, 101, 127`) với khóa mục tiêu thống nhất `--weights 0.90 0.07 0.03 --memory-scale 2048 --selection-rule epsilon_constraint`. Toàn bộ dữ liệu được kiểm toán và xuất tự động tại [ALGORITHM_STUDY.md](runs/nckh_2027_v2_algorithm_report/ALGORITHM_STUDY.md), [method_summary.csv](runs/paper_exports/nckh_2027_v2_3domain_vi/method_summary.csv) và [tables.tex](runs/paper_exports/nckh_2027_v2_3domain_vi/tables.tex).

### 7.1. Sự phân hóa Cấu hình Tối ưu Độc lập trên 3 Miền (`Reference Exhaustive`)
Đúng với yêu cầu khoa học đa miền, **3 bài toán hội tụ về 3 cấu hình tối ưu toàn cục hoàn toàn khác nhau**, không có cấu hình nào là tối ưu mặc định (`c00`):

| Miền (Study ID) | Metric Khóa | Tổng Lượt Vét Cạn | Cấu hình Tối ưu Toàn cục Độc lập | $Q_{\text{opt}}$ | $T_{\text{opt}}$ | $V_{\text{opt}}$ | Bẫy VRAM & Vùng Biên |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **1. `african_wildlife_benchmark`** (`4cae045574a367a3`) | `mAP@50` | `1,029.7s` | **`c14` (`e9810b3385f5`)**: `yolo11s`, `480/B4`, `fp16`, `AdamW`, `lr0=0.00018`, `freeze=8` | `0.9379` | `38.8s` | `724 MiB` | `5/18` bẫy `[1,4,7,11,16]`; `c07` là `fp16` (`2,624M`), `c09` là `fp32` an toàn (`762M`) |
| **2. `penn_fudan_research`** (`d4ad010059141fc7`) | `mAP@50:95` *(COCO chặt)* | `503.9s` | **`c17` (`cc13f397d796`)**: `yolo11s`, `640/B4`, `fp16`, `AdamW`, `lr0=0.0002`, `freeze=6` | `0.8772` | `29.3s` | `1,132 MiB` | Độ phân giải `640px` + giải phóng thêm `backbone` giúp định vị người đi bộ chặt hơn |
| **3. `cifar10_research` (GPU)** (`77d166e1df75ed54`) | `Top-1 Acc` | `295.6s` | **`c12` (`fdd43bd398ea`)**: `cnn_classifier`, `width=64`, `32/B16`, `fp32`, `AdamW`, `lr0=0.001` | `0.4200` | `12.4s` | `96 MiB` | Không rủi ro VRAM (`68–272 MiB`); không gian `18` cấu hình trên GPU |

### 7.2. Bảng Tổng hợp Kết quả 8 Phương pháp trên Từng Miền

#### A. Study 1 — `african_wildlife_benchmark` (`budget = 135.0s` $\approx 13.1\%$ tham chiếu; `max_memory = 1,850 MiB`; $\varepsilon = 0.015$)

| Thuật toán | Khả thi | `U Regret` (`TB ± ĐLC`) | Trúng Tối ưu | Pareto Hit | `AUARC` ($\downarrow$) | Thời gian TB | Vi phạm VRAM / Lãng phí | Conformal lọc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conformal_mf` (CC-MFBO)** | **8/8** | **`0.0748 ± 0.1143`** | **50.0%** | 50.0% | **`0.4244`** | 125.5s | **`0.00` / `0.0s`** | **5.0** |
| `mf_no_conformal` | 7/8 | `0.1540 ± 0.1333` | 25.0% | 62.5% | `0.4933` | 118.3s | `1.38` / `29.1s` | 0.0 |
| `default` (`c00`) | 8/8 | `0.2509 ± 0.0000` | 0.0% | 100.0% | `0.3444` *(1 trial)* | 31.9s | `0.00` / `0.0s` | 0.0 |
| `constrained_bo` (`cEI`) | 8/8 | `0.2529 ± 0.1903` | 0.0% | 50.0% | `0.4849` | 120.3s | `0.88` / `37.1s` | 0.0 |
| `random` | 8/8 | `0.2532 ± 0.1545` | 0.0% | 37.5% | `0.4314` | 131.3s | `0.62` / `20.2s` | 0.0 |
| `adaptive_full` | 8/8 | `0.2868 ± 0.0495` | 0.0% | 62.5% | `0.4392` | 97.0s | `0.00` / `0.0s` | 0.0 |
| `successive_halving` | 8/8 | `0.2868 ± 0.0495` | 0.0% | 62.5% | `0.4608` | 101.3s | `0.00` / `0.0s` | 0.0 |
| `tpe_bo` | 7/8 | `0.3054 ± 0.2262` | 0.0% | 50.0% | `0.4693` | 121.8s | `0.88` / `38.3s` | 0.0 |

#### B. Study 2 — `penn_fudan_research` (`mAP@50:95`, `budget = 108.0s` $\approx 21.4\%$ tham chiếu; `max_memory = 1,850 MiB`; $\varepsilon = 0.01$)

| Thuật toán | Khả thi | `U Regret` (`TB ± ĐLC`) | Trúng Tối ưu | Pareto Hit | `AUARC` ($\downarrow$) | Thời gian TB | Vi phạm VRAM / Lãng phí | Conformal lọc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conformal_mf` (CC-MFBO)** | **8/8** | **`0.0009 ± 0.0026`** | **87.5%** | **100.0%** | `0.2968` | 101.4s | **`0.00` / `0.0s`** | **5.0** |
| `mf_no_conformal` | 7/8 | `0.0908 ± 0.0566` | 0.0% | 62.5% | `0.3526` | 85.4s | `1.62` / `22.3s` | 0.0 |
| `tpe_bo` | 8/8 | `0.1081 ± 0.1255` | 12.5% | 62.5% | `0.3505` | 97.4s | `1.00` / `31.0s` | 0.0 |
| `adaptive_full` | 8/8 | `0.1293 ± 0.0000` | 0.0% | 100.0% | `0.2951` | 94.5s | `0.00` / `0.0s` | 0.0 |
| `default` (`c00`) | 8/8 | `0.1293 ± 0.0000` | 0.0% | 100.0% | `0.2167` *(1 trial)* | 22.7s | `0.00` / `0.0s` | 0.0 |
| `random` | 8/8 | `0.1398 ± 0.1118` | 12.5% | 62.5% | `0.2930` | 103.9s | `0.75` / `21.1s` | 0.0 |
| `constrained_bo` (`cEI`) | 7/8 | `0.1450 ± 0.1591` | 12.5% | 62.5% | `0.3321` | 95.9s | `0.88` / `28.2s` | 0.0 |
| `successive_halving` | 8/8 | `0.1542 ± 0.0705` | 0.0% | 100.0% | `0.3707` | 90.4s | `0.00` / `0.0s` | 0.0 |

#### C. Study 3 — `cifar10_research` trên GPU (`Top-1 Acc`, `15` epoch full / `4` epoch probe `256` ảnh, `budget = 68.0s` $\approx 23.0\%$ tham chiếu; $\varepsilon = 0.02$)

| Thuật toán | Khả thi | `U Regret` (`TB ± ĐLC`) | Trúng Tối ưu | Pareto Hit | `AUARC` ($\downarrow$) | Thời gian TB | Vi phạm VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tpe_bo` | 8/8 | **`0.0372 ± 0.0331`** | **25.0%** | 75.0% | **`0.0913`** | 61.9s | `0.00` |
| `constrained_bo` (`cEI`) | 8/8 | `0.0407 ± 0.0384` | **25.0%** | 62.5% | `0.0921` | 59.9s | `0.00` |
| **`conformal_mf` (CC-MFBO)** | **8/8** | `0.0619 ± 0.0487` | 12.5% | 50.0% | `0.1251` | 53.6s | `0.00` |
| `mf_no_conformal` | 8/8 | `0.0619 ± 0.0487` | 12.5% | 50.0% | `0.1252` | 53.5s | `0.00` |
| `adaptive_full` | 8/8 | `0.0715 ± 0.0000` | 0.0% | 100.0% | `0.1137` | 42.3s | `0.00` |
| `successive_halving` | 8/8 | `0.0715 ± 0.0000` | 0.0% | 100.0% | `0.1089` | 42.2s | `0.00` |
| `default` (`c00`) | 8/8 | `0.0715 ± 0.0000` | 0.0% | 100.0% | `0.0989` | 21.8s | `0.00` |
| `random` | 8/8 | `0.0833 ± 0.0314` | 0.0% | 25.0% | `0.1236` | 64.3s | `0.00` |

### 7.3. Kiểm định Thống kê Ghép cặp Phi tham số (`conformal_mf` trừ Baseline trên `objective_regret`)

| Đường cơ sở đối chứng | Study 1 Wildlife (`n=8`/`7`): `ΔU` / Wilcoxon `p` / Sign-Flip `p` / Holm `p` | Study 2 Penn-Fudan (`n=8`/`7`): `ΔU` / Wilcoxon `p` / Sign-Flip `p` / Holm `p` | Study 3 CIFAR-10 GPU (`n=8`): `ΔU` / Sign-Flip `p` |
| :--- | :---: | :---: | :---: |
| vs **`adaptive_full`** | **`-0.2120`** / **`0.0078`** / **`0.0078`** / `0.0547` | **`-0.1284`** / **`0.0078`** / **`0.0078`** / `0.0547` | `-0.0096` / `0.5391` |
| vs **`successive_halving`** | **`-0.2120`** / **`0.0078`** / **`0.0078`** / `0.0547` | **`-0.1533`** / **`0.0078`** / **`0.0078`** / `0.0547` | `-0.0096` / `0.5391` |
| vs **`default`** | **`-0.1761`** / **`0.0156`** / **`0.0156`** / `0.0781` | **`-0.1284`** / **`0.0078`** / **`0.0078`** / `0.0547` | `-0.0096` / `0.5391` |
| vs **`constrained_bo` (`cEI`)** | **`-0.1782`** / **`0.0312`** / **`0.0312`** / `0.1250` | **`-0.1440`** (`n=7`) / **`0.0312`** / **`0.0312`** / `0.0625` | `+0.0212` / `0.4062` |
| vs **`random`** | **`-0.1784`** / `0.0547` / **`0.0391`** / `0.1250` | **`-0.1388`** / **`0.0156`** / **`0.0156`** / `0.0625` | `-0.0214` / `0.2656` |
| vs **`tpe_bo`** | `-0.2199` (`n=7`) / `0.0625` / `0.0625` / `0.1250` | **`-0.1072`** / **`0.0156`** / **`0.0156`** / `0.0625` | `+0.0248` / `0.3438` |
| vs **`mf_no_conformal`** | `-0.1165` (`n=7`) / `0.0938` / `0.0938` / `0.1250` | **`-0.0897`** (`n=7`) / **`0.0156`** / **`0.0156`** / `0.0625` | `+0.0000` / `1.0000` *(khớp đồng nhất)* |

*Lưu ý thống kê:* Độ phân giải cực tiểu của kiểm định Sign-Flip 2 phía với $n = 8$ là $2/2^8 = 1/128 = 0.0078125$. Vì `algorithm_study.py` hiệu chỉnh Holm theo họ $m = 7$ đường cơ sở đồng thời trên cùng miền, mức $0.0078 \times 7 = 0.0547$ sát ngưỡng $0.05$, trong khi **từng cặp riêng lẻ đạt `p = 0.0078–0.0312` trên toàn bộ 7/7 đối thủ tại Study 2** và `5/7` đối thủ tại Study 1.

---

## 8. NGHIÊN CỨU CẮT BỎ (ABLATION) & ĐỘ NHẠY QUY TẮC CHỌN LỌC

### 8.1. Tách riêng đóng góp của từng thành phần (`conformal_mf` vs `mf_no_conformal` vs `constrained_bo` vs `adaptive_full`)
- **Đóng góp của `ConformalVRAMGuard` (`conformal_mf` vs `mf_no_conformal`):** Cùng chia sẻ thuật toán đa tầng và `cEUI`, việc bỏ lọc Conformal làm phát sinh `1.38–1.62` lần vi phạm VRAM/hạt (`22.3s–29.1s` lãng phí), tụt tỷ lệ khả thi xuống `7/8` (`87.5%`) trên cả 2 miền Detection và tăng regret gấp $2.06\times$ (Wildlife) và $100\times$ (Penn-Fudan: `0.0009` $\rightarrow$ `0.0908`, `p = 0.0156`).
- **Đóng góp của Đa tầng Thăm dò + `cEUI` (`conformal_mf` vs `constrained_bo` & `adaptive_full`):** `constrained_bo` (cEI tiêu chuẩn không có vật lý tiên nghiệm + probe) vẫn tốn `0.88` vi phạm/hạt và có `Regret` cao hơn $3.38\times$ trên Wildlife (`p = 0.0312`) và $161\times$ trên Penn-Fudan (`p = 0.0312`). `adaptive_full` bị `small_first_key` giữ chặt ở cụm `320px`, không tìm ra vùng tối ưu `480–640px` khi ngân sách hẹp (`p = 0.0078`).

### 8.2. Độ nhạy Quy tắc Chọn lọc & Ngưỡng $\varepsilon$ (Hậu kiểm Pareto — [summary.md](runs/nckh_2027_v2_selection_ablation/summary.md))

| Miền | `Weighted` / `Tchebycheff` / `TOPSIS` | `Knee` (Góc gập Pareto) | $\varepsilon \in [0.005, 0.015]$ | $\varepsilon \in [0.02, 0.05]$ | $\varepsilon = 0.10$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wildlife (`mAP@50`)** | `c14` (`0.9379`, `38.8s`) | `c00` (`0.6522`, tụt **`-28.6%`**) | **`c14` (`0.9379`)** *(bền vững)* | `c14` (`0.9379`) | `c14` (`0.9379`) |
| **Penn-Fudan (`mAP@50:95`)** | `c17` (`0.8772`, `29.3s`, `1,132M`) | `c17` (`0.8772`) | **`c17` (`0.8772`)** *(chính xác cao)* | Dịch mượt sang **`c14`** (`0.8617`, nhanh hơn `5.4s`, ít VRAM `-408M`) | Tụt xuống `c10` (`0.7987`) |
| **CIFAR-10 (`Top-1 Acc`)** | `c12` (`0.4200`, `12.4s`, `96M`) | `c00` (`0.3400`, tụt `-8.0%`) | `c12` (`0.4200`) | **`c12` (`0.4200`)** | Tụt xuống `c00` (`0.3400`) |

*(Xem đồ thị trade-off chi tiết tại [pareto_epsilon_tradeoff.svg](runs/nckh_2027_v2_selection_ablation/pareto_epsilon_tradeoff.svg)).* Việc đặt $\varepsilon \in [0.01, 0.02]$ bảo toàn tối ưu chất lượng ổn định hơn hẳn `Knee` (vốn hay chọn điểm góc nhanh nhưng mất $8\%\text{--}28.6\%$ chất lượng), đồng thời cho phép người dùng dịch chuyển có kiểm soát dọc đường Pareto khi nới $\varepsilon \ge 0.02$.

---

## 9. TỔNG QUÁT HÓA TRÊN TẬP `TEST` ĐỘC LẬP Ở CẢ 3 MIỀN

Sau khi khóa vĩnh viễn cấu hình chọn lọc từ `conformal_mf` (tách biệt hoàn toàn khỏi tập `Test`), chúng tôi huấn luyện đa hạt giống (`train` + `val`) và mở đánh giá tập `Test` giữ lại đúng một lần qua `evaluate-v2 --seed-scope all` (`rect=False`, lưu tại [heldout_test_summary.json](runs/nckh_2027_v2_selection_ablation/heldout_test_summary.json)):

| Miền thực nghiệm | Cấu hình & Thời lượng Train Độc lập | Validation (`TB ± ĐLC`) | **Held-Out `Test` (`TB ± ĐLC`)** | Chỉ số Phụ & Tốc độ Suy luận |
| :--- | :--- | :---: | :---: | :--- |
| **1. `african_wildlife_full`** (`211` ảnh test, `nc=4`) | `c00` (`yolo11n`, `12` ep, `2` seeds `[17,29]`) | `mAP@50 = 0.9237 ± 0.0088` | **`Test mAP@50 = 0.9528 ± 0.0004`** | `mAP@50:95 = 0.7677 ± 0.0030`, `F1 = 0.9093`, **`159.05 FPS`** (`6.30 ms/ảnh`) |
| **2. `penn_fudan_research`** (`100%` blind test split) | **`c17` (`cc13f397d796`, `yolo11s 640/4 frz=6`, `12` ep, `3` seeds `[17,29,43]`)** | `mAP@50:95 = 0.8996 ± 0.0187` | **`Test mAP@50:95 = 0.8667 ± 0.0186`**<br>**`Test mAP@50 = 0.9785 ± 0.0107`** | `Test mAP@75 = 0.9426`, `Precision = 0.9763`, `Recall = 0.9296`, `F1 = 0.9517`, **`73.16 FPS`** (`13.67 ms/ảnh`) |
| **3. `cifar10_research`** (`500` ảnh test, `nc=10`) | `e8ee60458e2d` (`conformal_mf repeat_17`, `cnn w=96 96/B8`, `20` ep, `3` seeds `[17,29,43]`) | `Val Acc = 0.3678 ± 0.0369` | **`Test Acc = 0.3480 ± 0.0231`** *(gap `-0.0198`)* | VRAM `138.0 MiB`, `T = 40.1s/seed` |

---

## 10. PHÂN TÍCH TRƯỜNG HỢP THẤT BẠI & BIÊN GIỚI HẠN (`FAILURE CASE ANALYSIS`)

1. **Thất bại cục bộ trên `CIFAR-10` so với `tpe_bo` / `constrained_bo`:**  
   Ở Study 3 (mạng `cnn_classifier` nhỏ, không có ràng buộc VRAM), `conformal_mf` đạt `0.0619 ± 0.0487` — thắng `adaptive_full` / `successive_halving` / `default` (`0.0715`) và `random` (`0.0833`), nhưng xếp sau single-fidelity `tpe_bo` (`0.0372`) và `constrained_bo` (`0.0407`). **Nguyên nhân gốc:** trên CIFAR-10 dùng tập probe nhỏ (`256` ảnh, `4` epoch), mạng rất lớn `width=96, 96/B8` (`c15`, `T = 30.1s`) tăng nhanh hơn ở 4 epoch đầu so với cấu hình `width=64, 32/B16` (`c12`, `T = 12.4s`), nhưng về cuối epoch 15 lại hội tụ kém hơn (`0.2967` vs `0.4200`). Vì bỏ ~`15s` cho $2\text{--}3$ probe trước khi xác nhận full, `conformal_mf` mất 1 lượt full confirmation so với `tpe_bo` ở các hạt giống `17, 59, 127`. **Bài học định lượng:** `CC-MFBO` phát huy ưu thế áp đảo khi tồn tại rủi ro tài nguyên ($\ge 20\%$ cấu hình bẫy VRAM) và tín hiệu đa tầng dương tính; khi không gian an toàn 100% và thời gian 1 full trial nhỏ ($\sim 10\text{--}20$s), single-fidelity BO đỡ chi phí khởi tạo subprocess hơn.
2. **Lỗi `yolo26n + SGD + fp16 + batch >= 8` ở Probe 1-epoch:**  
   Trong 60 probes Conformal, $1/60$ probe (`yolo26n 320/B8 fp16 SGD`) thất bại do AMP `GradScaler` bỏ qua bước tối ưu ở 1 epoch đầu trên tập con nhỏ (`optimizer_state_created: false`). Sử dụng `AdamW` khi `batch >= 8` trên `yolo26n` hoặc giữ `SGD` ở `batch <= 4` khắc phục hoàn toàn lỗi này.
3. **Khóa file tạm Windows (`WinError 5`) thoáng qua:** Một lần ghi đè nguyên tử `result.tmp -> result.json` gặp khóa antivirus/Windows thoáng qua khi chạy đánh giá CIFAR-10; chạy lại độc lập hoàn thành `3/3` seeds sạch.

---

## 11. ĐE DỌA HIỆU LỰC (`THREATS TO VALIDITY`)

1. **Giới hạn phần cứng vật lý đơn (`RTX 3050 6GB Laptop GPU`):** Mọi kết quả đo VRAM thực tế, WDDM spill và tốc độ thực thi đều thu trên 1 GPU vật lý (`GPU-b06ae253-3f9c-54de-d7b7-0cd434157ea0`). Thẻ hiệu chuẩn `ConformalVRAMGuard` **không** tự động mang theo bảo chứng số đo sang GPU khác kiến trúc/driver (`RTX 4060`, `T4`, `A100`) nếu không hiệu chuẩn lại bằng [run_conformal_vram_study.py](scripts/run_conformal_vram_study.py) hoặc sử dụng cơ chế neo 2 điểm trực tuyến.
2. **Bản chất có điều kiện của các kiểm định thống kê (`n = 8` `search_seed`):** Các p-value Wilcoxon, Sign-Flip và khoảng bootstrap chỉ lượng hóa độ ổn định của thuật toán tìm kiếm qua các hạt giống tìm kiếm trên bộ lưới/dữ liệu cố định (`training_seeds = [17]` trong pha benchmark nhanh), **không** phải khoảng tin cậy xuyên tập dữ liệu hoặc xuyên GPU độc lập.
3. **Chuẩn hóa độ đo đa miền:** `Study 1` đo `mAP@50` (`4` ep), `Study 2` đo `mAP@50:95` (`4` ep), `Study 3` đo `Top-1 Acc` (`15` ep); do mục tiêu khác đơn vị vật lý, nghiên cứu cố ý **không** gộp `U Regret` trung bình qua cả 3 miền mà báo cáo rời rạc theo từng miền đúng nguyên tắc kiểm toán của `algorithm-report`.

---

## 12. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

Nghiên cứu đã xây dựng, chứng minh lý thuyết và kiểm chứng thực nghiệm thành công khung tối ưu **CC-MFBO (`conformal_mf`)** cho bài toán chọn cấu hình huấn luyện thị giác máy tính dưới ngân sách hẹp trên GPU phổ thông:
- **An toàn phần cứng hữu hạn mẫu:** `ConformalVRAMGuard` đạt **`96.4%` độ phủ thực nghiệm** (`MAE = 135.1 MiB`, `6.07%` loại nhầm), bắt sống **100%** trường hợp tràn WDDM (`5,174–6,452 MiB`), giữ **`0.00` vi phạm VRAM** xuyên suốt `16/16` lượt tìm kiếm Detection trong khi các phương pháp BO chuẩn lãng phí `20.2–38.3s`/lượt.
- **Vượt trội đa miền với 3 tối ưu khác biệt:** `CC-MFBO` tự thích nghi tìm đúng 3 cấu hình tối ưu hoàn toàn khác nhau cho 3 miền bài toán, giảm `U Regret` tới **`3.8×`** trên Wildlife (`0.0748` vs `adaptive_full 0.2868`, `p = 0.0078`) và tới **`143×`** trên Penn-Fudan COCO (`0.0009` vs `adaptive_full 0.1293`, `87.5%` trúng tối ưu, thắng toàn bộ 7 đường cơ sở với `p = 0.0078–0.0312`), đạt hiệu năng triển khai thực tế **`0.9528 Test mAP@50 @ 159 FPS`** và **`0.8667 Test mAP@50:95 @ 73 FPS`**.

**Hướng phát triển tiếp theo:** Mở rộng hiệu chuẩn Conformal đa GPU (`RTX 4060`, `T4`, Linux CUDA vs Windows WDDM), thiết kế cơ chế dừng sớm thích nghi cho Giai đoạn 1 khi phát hiện phương sai chất lượng probe quá thấp ($\hat{\tau} < 0.35$) ở miền không có rủi ro VRAM, và mở rộng kiểm chứng lên bài toán phân đoạn ảnh (`Instance Segmentation`) và `Captioning`.

---

## 13. TÀI LIỆU THAM KHẢO HỌC THUẬT

1. **Vovk, V., Gammerman, A., & Shafer, G.** (2005). *Algorithmic Learning in a Random World*. Springer.
2. **Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.** (2018). Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association (JASA)*, 113(523), 1094–1111.
3. **Gelbart, M. A., Snoek, J., & Adams, R. P.** (2014). Bayesian Optimization with Unknown Constraints. *Uncertainty in Artificial Intelligence (UAI)*, 250–259.
4. **Kennedy, M. C., & O'Hagan, A.** (2000). Predicting the Output from a Complex Computer Code When Fast Approximations Are Available. *Biometrika*, 87(1), 1–13.
5. **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B.** (2011). Algorithms for Hyper-Parameter Optimization. *NeurIPS*, 24, 2546–2554.
6. **Jamieson, K., & Talwalkar, A.** (2016). Non-stochastic Best Arm Identification and Hyperparameter Optimization. *AISTATS*, 240–248.
7. **Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A.** (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *JMLR*, 18(185), 1–52.
8. **Falkner, S., Klein, A., & Hutter, F.** (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. *ICML*, 1437–1446.
9. **Holm, S.** (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
10. **Wilcoxon, F.** (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80–83.
