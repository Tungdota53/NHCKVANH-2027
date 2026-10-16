# Thuật toán tối ưu cấu hình adaptive_full — 0.10.0

## 1. Câu hỏi nghiên cứu và đóng góp được kiểm chứng

Trong grid cấu hình chạy trên máy đang đo, với budget tìm kiếm B hữu hạn, chọn cấu hình feasible cân bằng validation quality Q, thời gian T và bộ nhớ M. Đầu ra không phải một bảng preset theo tên GPU.

Tên `adaptive_full` là tên phương pháp triển khai trong đề tài, không tự chứng minh tính mới so với mọi công trình đã công bố. Thành phần cần đánh giá là sự kết hợp: xác nhận đủ seeds trước khi học quality; acquisition thích nghi có chi phí/độ bất định/độ mới; dành tài nguyên cho toàn bộ seeds còn thiếu; lưu dấu quyết định và kiểm chứng bằng raw evidence. Không kế thừa định lý no-regret chỉ bằng cách dùng kernel.

Các baseline reserve/random/default/SH giữ riêng. Code phương pháp chính: `research_pipeline/v2/acquisition.py` và `scheduler.py`; giao diện mặc định mới gọi `adaptive_full`.

## 2. Định nghĩa bài toán

Khóa trước: train/val splits, task/metric, hardware/runtime, hữu hạn C, horizon E, training seeds S, B, constraints, weights/scales và acquisition parameters. Không dùng test để chọn cấu hình/checkpoint/weights.

Với mỗi c đủ toàn bộ full-data E epochs × S seeds:

```
Q(c)    = mean_s validation_quality(c,s)
Qmin(c) = min_s  validation_quality(c,s)
T(c)    = mean_s cold_worker_wall_seconds(c,s)
Tmax(c) = max_s  cold_worker_wall_seconds(c,s)
M(c)    = max_s  primary_memory_counter(c,s)
U(c)    = wA Q(c) - wT T(c)/Tref - wM M(c)/Mref
```

Weights chuẩn hóa tổng bằng 1; constraints dùng Qmin/Tmax/M, không dùng dự báo để cấp chứng nhận. CUDA counter là allocator reserved MiB, không là toàn bộ VRAM; CPU là process-tree RSS. T có startup/validation/checkpoint trong worker. Tổng online cost còn gồm preparation/controller/verification/export và được báo riêng; không cộng worker time hai lần.

Mục tiêu thực nghiệm `argmax U(c)` trên các cấu hình đã full-confirm và feasible. Chưa full toàn C: best-observed. Đủ toàn C: empirical optimum của **grid/horizon/seeds/objective đã đo**, không phải optimum kỳ vọng, hội tụ hoặc mọi GPU/dataset. Trọng số là ưu tiên người dùng, không phải hằng số tối ưu do thuật toán suy ra.

0.10 chỉ hỗ trợ weighted acquisition cho adaptive methods; TOPSIS/Tchebycheff/knee vẫn có ở các baseline. Không im lặng tìm theo U rồi quảng cáo acquisition tối ưu một rule khác.

## 3. Biểu diễn cấu hình và mô hình quality

Numeric features: log/image/batch/width/hidden/LR/parameter descriptors và các continuous fields được hỗ trợ. Chuẩn hóa bằng min/max của **grid đã khóa**, không bằng metric tương lai. Categorical features dùng mismatch model-content identity/family/encoder/precision/optimizer/pretrained. Loại trường hằng. Khoảng cách d² là trung bình squared numeric differences + categorical mismatches, trong [0,1].

Fit targets chỉ đến từ các cấu hình hoàn tất đủ S/E. Module tự kiểm tra lại raw workers `mode=train`, status, horizon, optimizer steps, finite metric, actual config và checkpoint identity marker; tính lại mean Q. Không tin partial row mean. Một config là một training point, không coi các seed là các config độc lập. Constraint-failure full vẫn có thể cung cấp quality target; chỉ feasible full mới làm incumbent.

```
k(c,c') = exp(-d(c,c')² / (2 l²))
Kλ       = K + λ I
μ(c)     = clip(mean(y) + k(c,X)' Kλ^-1 (y - mean(y)), 0, 1)
v(c)     = clip(1 - k(c,X)' Kλ^-1 k(c,X), 0, 1)
n(c)     = min_i d(c, observed_config_i)
s²       = variance(config_means) + mean_squared_ridge_residual
u(c)     = clip(sqrt(v(c)) + n(c)*sqrt(s²), 0, 1)
Q+(c)    = min(1, μ(c) + β*u(c))
```

Đây là kernel ridge point surrogate với **heuristic uncertainty**, không phải calibrated posterior, confidence interval hoặc xác suất an toàn. u không mô hình hóa đầy đủ seed-noise/heteroscedasticity. Solver lỗi số học fallback được ghi rõ; không xuất NaN như dự báo hợp lệ.

Thiếu 2 config full: chưa fit kernel quality, dùng novelty/cost fallback. Vì vậy phải báo số acquisition steps thật sự có `surrogate.fitted=true`, không đếm bootstrap thành học thích nghi.

## 4. Acquisition có chi phí

Từ observed full workers, CostModel ước lượng thời gian một full seed; tách non-loop overhead với loop scaling khi có telemetry. Width exponent học từ contrasts online nếu đủ; nếu không dùng heuristic công khai. Không nhìn reference để sửa dự báo.

Với số seeds còn thiếu m(c):

```
Cremaining(c) = predicted_full_worker_seconds(c) * m(c)
U+(c)        = wA Q+(c) - wT T_hat(c)/Tref - wM M_hat(c)/Mref
I+(c)        = max(0, U+(c) - best_observed_feasible_U)
p(c)         = 1/(1 + predicted_resource_excess)
A(c)         = [I+(c) + η*n(c)] * p(c) / Cremaining(c)
```

Số dương rất nhỏ xử lý hòa điểm. Thiếu incumbent dùng mốc U=0; ít hơn 2 config full dùng `[1 + η*n(c)]*p(c)/Cremaining(c)`. Thời gian partial config dùng mean full workers đã đo × missing seeds; các ứng viên khác dùng forecast. Unknown T/M nhận median/default **planning placeholder**, có cờ unknown; không là bằng chứng feasibility. `predicted_resource_excess` là tổng `max(0,M_hat/Mlimit-1)` và `max(0,T_hat/Tlimit-1)`, chỉ lấy các giới hạn được đặt. Điểm tài nguyên vượt trần chỉ bị soft penalty trong acquisition; admission/verification nằm ở coordinator và measurement thật.

Các tham số khóa mặc định: λ=0.05, l=1, β=0.25, η=0.05. Đây là lựa chọn thiết kế có regularization trên features đã chuẩn hóa; cần sensitivity/ablation để đánh giá, không được gọi là hệ số tối ưu phổ quát hoặc suy ra từ định lý.

## 5. Coordinator và bảo vệ bằng chứng

```
while còn budget dành cho worker + finalization:
    xây full measurements; fit không dùng probes/learning curves/reference
    loại khỏi hành động: đã đủ seeds, đã thử trong execution này,
                       hoặc đã có OOM/config mismatch/evidence failure
    tạm hoãn các action có all-missing-seed forecast không vừa budget
    ưu tiên hoàn tất partial full confirmations từ resume
    nếu chưa có full config: metadata-small bootstrap (seeded tie order)
    nếu có: chọn argmax A(c) trên các action còn lại
    journal toàn bộ ranking/fit/forecast/reason; lưu state
    train full E, lần lượt đủ missing seeds; cập nhật sau mỗi worker
    acquire lại sau mỗi config; không permanent prune theo quality sớm
verify đầy đủ dữ liệu/source/checkpoint; raw audit; commit selection
```

Không có subset-quality rung trong adaptive_full. Probes cũ giữ trong evidence nhưng không replay vào adaptive CostModel. Cost model chỉ nhận các worker đầy đủ chứng cứ; timeout chỉ là censored lower bound, không được coi là phép đo full rẻ. Mỗi config được mở tối đa một lần mỗi execution để tránh loop retry lỗi. Resume có thể thử lại timeout nhưng không tự nối model/optimizer/RNG giữa epoch.

Affordability: planning time × missing seeds + cleanup/finalization phải vừa remaining budget. Unknown-cost bootstrap chia cap đều cho missing seeds. Forecast sai vẫn có thể tạo timeout/partial/no selection. Heuristic quá bảo thủ cũng có thể bỏ lỡ config thực tế chạy vừa. Không có guarantee hard deadline hoặc bảo đảm thăm hết slow starters dưới B nhỏ.

Mỗi action kèm acquisition receipt trong events/state/report; chỉ measured full outcomes quyết định recommendation. Không gắn nhãn measured cho Q+ hoặc M_hat.

## 6. Các tính chất có thể lập luận

1. **Không leakage trực tiếp:** API acquisition không nhận test/reference; rows có thể chứa learning curves nhưng thuật toán không đọc/dùng early quality, target lấy lại từ full worker records. Unit tests chèn hostile probe/partial/forecast quality để kiểm tra invariance. Điều này không phủ nhận prior development đã từng xem dataset.
2. **Recommendation có chứng cứ:** selection chỉ xét config đủ all-S/full-E/actual-config/finite metrics/checkpoint và constraints. Surrogate không tự tạo incumbent.
3. **Incumbent không giảm:** nếu evidence đã hoàn tất bất biến, chỉ append full configs mới và giữ nguyên objective, max measured feasible U không giảm. Đây là tính chất chọn max, không là regret bound hay dự đoán chất lượng tương lai.
4. **Finite execution:** mỗi config được mở tối đa một lần trong một execution; không retry vô hạn. Nếu mọi worker hoàn tất và forecasts/budget cho phép thăm toàn grid thì chọn empirical argmax; không suy rằng mọi B đều đủ.

Mỗi acquisition có O(C²*d) mixed distances/kernel, O(n³ + n²*C) dense solve với n config đã full-confirm; retained arrays O(C² + n*C) cùng history, transient broadcast memory có thể O(C²*d). Grid hiện tối đa512; chưa có sparse/incremental GP hoặc worker pool. Mọi overhead controller được tính vào online clock.

## 7. Ablation và thực nghiệm

- `adaptive_no_cost`: bỏ **denominator acquisition**, không bỏ time/memory trong objective hoặc affordability guard.
- `adaptive_no_uncertainty`: β bonus bằng0, giữ prediction/novelty.
- `adaptive_no_diversity`: η bonus bằng0, không bỏ feature distances khỏi kernel/uncertainty.
- default/random/reserve/SH là baseline riêng; default giữ đúng config mặc định adapter tại index0.

Thiết kế khóa trước ở `ALGORITHM_010_EXPERIMENTS.md`. Tất cả online decisions khóa trước independent reference, cùng grid/E/S/objective/runtime/budget. Reference đủ toàn grid mới tính empirical regret. B phải nhỏ hơn **measured exhaustive reference cost** để analysis gọi là budget-limited, kiểm tra sau chứ không dùng oracle để admission.

`algorithm-report` audit raw và paired studies; tách no-selection/reference disagreement khỏi conditional regret, đếm dataset/GPU/CPU đúng scope, bootstrap chỉ khi đủ5 usable search repeats. CI có điều kiện theo một fixed study/reference, không là CI trên toàn bộ dataset/GPU; chỉ đổi seed label không chứng minh độc lập. Không tự gắn superiority.

## 8. Nguồn nền tảng và phần không kế thừa

- [Snoek et al., Practical Bayesian Optimization, NeurIPS2012](https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html): surrogate/acquisition và chi phí mỗi phép thử là nền tảng đã có; phương pháp này không được quảng cáo là phát minh ra cost-aware HPO.
- [Srinivas et al., GP-UCB](https://arxiv.org/abs/0912.3995): regret theory cần giả định GP/RKHS và cơ chế confidence cụ thể. KRR+uncalibrated u ở đây **không thỏa/không chứng minh các điều kiện đó**, nên không kế thừa no-regret theorem.
- [Li et al., Hyperband, JMLR2018](https://jmlr.org/papers/v18/16-558.html): adaptive allocation/early stopping là baseline nghiên cứu liên quan. adaptive_full không phải Hyperband; đổi chi phí bằng full confirmations để tránh trực tiếp loại theo early quality, không hứa speedup tương đương.

Kết quả thực tế, kể cả thất bại, ở `ALGORITHM_010_VALIDATION.md`. Code/test pass là nghiệm thu implementation, không thay thế chứng minh thực nghiệm về hiệu quả thuật toán.
