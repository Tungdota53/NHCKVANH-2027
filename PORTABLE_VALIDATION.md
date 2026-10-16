# Kiểm chứng nâng cấp portability 0.5

## Thay đổi đã thực thi

- `adaptive_verified`: Q proxy không loại ứng viên; dùng profile thời gian để lập lịch, giữ khám phá các LR, xác nhận bằng full data/full horizon. Dự báo thời gian là heuristic, không phải confidence bound.
- `verify-config`: fit label/vocabulary và đo hardware/dataset hiện tại, tổng hợp đủ seed trước lựa chọn; không dùng nghiệm ViECAP/CIFAR làm đáp án cho dataset mới.
- Ba mức: empirical optimum trong grid đã đo đủ / full-trained best observed nhưng chưa đo hết / không có lựa chọn verified. Không đồng nhất chúng với tối ưu toàn cục.
- Grid expanded 6 cấu hình thay grid4 cũ; có khóa precision và source/runtime protocol.
- `transfer-report`: kiểm tra request/result gốc, định danh nội dung/split thay vì chỉ đường dẫn; đếm physical GPU bằng UUID, không lấy hai runtime CPU/CUDA làm hai GPU.
- Sửa CUDA logical-index/NVML physical-index: lấy UUID từ CUDA properties rồi đối chiếu NVML theo UUID, không đo nhầm adapter khi device ordering hoặc CUDA_VISIBLE_DEVICES remap. Thiếu UUID hoặc NVML không đối chiếu được vẫn có thể training, nhưng physical identity bị đánh dấu chưa verified. MIG instances không tự tăng số physical GPUs khi nhận được parent identity.

Các run0.4 và số liệu thất bại trước được giữ nguyên. Heuristic mới lấy bài học từ ViECAP/CIFAR nên không gọi cải tiến trên chính hai bộ đó là blind validation.

## Miền dữ liệu mới: Fashion-MNIST

Nguồn [Zalando chính thức](https://github.com/zalandoresearch/fashion-mnist), tải qua torchvision và kiểm MD5. Script `scripts/prepare_fashion_research.py`: giữ nhãn gốc, seed101, stratified1000train/300val/500test; train/val từ original train, test từ original test, không trùng nội dung. Đây là subset, không phải full Fashion-MNIST hoặc SOTA. Test metric không dùng để chọn cấu hình.

Khóa trước thực nghiệm: CNN classifier ảnh32, batch16, effective32, AdamW, precision FP32, 8epoch, seeds17/29/43; width16/32 × LR0.001/0.0003/0.003. Objective balanced weights0.6/0.25/0.15, anchors600s/1024MiB, minQ0. CPU/GPU dùng cùng nội dung/grid/fidelity/seeds/objective và cùng max-memory2286.4968750000003MiB đã ghi trong CPU protocol; không chỉnh objective theo kết quả GPU.

CPU dùng torch2.14.1+cpu; CUDA torch2.6.0+cu124 trên RTX3050 Laptop6GB. Cùng FP32 storage dtype/deterministic mode **không** bảo đảm đồng nhất số học/kernels giữa runtime/hardware. Không quy mọi khác biệt thành hiệu ứng GPU. Không chạy CPU/GPU training đồng thời khi đo.

## CPU: đã hoàn tất independent research

Study `runs/research_fashion_05/20261012_233601_316234`: 6config ×3seed = **18/18 full-reference runs**. Search default/random/adaptive được khóa trước reference. Random và adaptive mỗi phương pháp còn full-train18 trial online riêng; tổng study CPU có **54 lượt full-data đủ8epoch**, cộng18 proxy trial, không dùng proxy làm oracle.

| Config | Q mean ± SD (val accuracy) | Full time mean ± SD(s) | Max RSS MiB |
|---|---:|---:|---:|
| c00 width16/LR0.001 | 0.714444 ± 0.018359 | 16.19 ± 0.54 | 290.2 |
| c01 width32/LR0.001 | 0.774444 ± 0.038490 | 17.99 ± 1.30 | 295.0 |
| c02 width16/LR0.0003 | 0.632222 ± 0.038345 | 17.16 ± 0.89 | 290.6 |
| c03 width32/LR0.0003 | 0.732222 ± 0.020092 | 18.08 ± 0.86 | 294.7 |
| c04 width16/LR0.003 | 0.750000 ± 0.012019 | 16.75 ± 1.14 | 290.3 |
| c05 width32/LR0.003 | 0.783333 ± 0.038442 | 18.45 ± 1.16 | 295.5 |

Nghiệm balanced thực nghiệm: **c05**, val accuracy78.33% ±3.84điểm%; không phải test accuracy hoặc chứng nhận ứng dụng. Khác nghiệm c01/LR0.001 của CIFAR grid4 cũ; hai grid khác nhau nên không gọi chênh lệch đó là một effect thuần dataset.

| Method | Feasible | Balanced regret mean ± SD | Search mean ± SD(s) |
|---|---:|---:|---:|
| Default | 100% | 0.039602 ±0 | 0 |
| Random full | 100% | 0.007859 ±0.009571 | 93.97 ±17.70 |
| Adaptive verified | 100% | 0.006172 ±0.010691 | 114.11 ±9.95 |

Adaptive giảm mean regret quan sát so Random **0.001686**, nhưng chậm hơn và SD lớn; **không kết luận statistical superiority/SOTA** từ3seed. Adaptive single-seed chọn c05/c05/c04, Random c05/c01/c04. Production `verify-config` khác ở chỗ tổng hợp đủ3seed trước khi xuất cấu hình; single-seed ranking còn sai so mục tiêu mean3seed.

Proxy Q MAE0.300122, time MAPE31.00%, RSS relative error1.94%. Không dùng Q proxy kém tin cậy làm certificate. Study elapsed942.14s, independent offline reference313.86s, common setup3.78s. Audit đối chiếu18 request/result gốc; `RESEARCH_AUDITED.md` và `study.audited.json` trong study.

## CUDA local verification

Đã hoàn tất `runs/verified_fashion_gpu_05/20261012_235605_198545`: **18/18 lượt full-data đủ8epoch**, cộng18 proxy trial; cùng dữ liệu/grid/epoch/seed/FP32/objective với CPU. Audit đối chiếu18 request/result gốc. Mọi full confirmation tính vào online budget240s mỗi seed; không có offline oracle miễn phí.

Status `empirical_optimal_in_declared_space`; chọn **c05: width32, LR0.003, batch16, effective32, FP32**. Val accuracy **78.33% ±3.84điểm%**, full time18.24±0.36s, peak CUDA reserved70MiB. RSS CPU và CUDA allocator không phải cùng loại memory measurement. Study elapsed472.656s, chi phí worker full-confirm326.577s, common setup5.39s; profile/startup cũng tiêu ngân sách. Không có oracle độc lập trong lượt production nên không báo regret hoặc superiority từ chính lượt này.

Sau CPU study hoàn tất, sửa riêng hardware-identity logger để xử lý CUDA/NVML remapping rồi mới khởi động GPU study. Không đổi model/optimizer/metric/training algorithm giữa hai lượt; checksum/source snapshot từng run lưu khác biệt logger. Đã đối chiếu UUID thực RTX3050: CUDA UUID và physical NVML UUID khớp. UUID được ẩn trong bản tài liệu công khai; giá trị gốc vẫn giữ trong artifact cục bộ, không thay đổi số GPU vật lý đã kiểm chứng.

## Kiểm tra ngân sách thiếu và resume thật

Run `runs/verified_fashion_partial_05/20261013_000741_971416` dùng60s/seed, vẫn giữ đủ6config×3seed×8epoch. Ban đầu chỉ2 lượt full hoàn tất, không cấu hình nào đủ3seed: **`no_verified_feasible_selection`**, không xuất `selected.json`. Bằng chứng trước resume giữ tại `budget_initial.json` và `RESEARCH.budget_initial.md`.

Resume thêm198s, không replay search, chỉ khởi động lại lượt full còn thiếu từ initialization. Sau resume có **14 lượt full thành công**, cộng1 lượt timeout được giữ nguyên trong audit; dù log timeout ghi đã tới epoch8, status vẫn không được nâng thành thành công. Chỉc02 vàc03 đủ3seed. Chọnc03 với Qmean0.732222 và status **`fully_trained_best_observed`**, `reference_complete=false`: không chọnc05 chỉ từ2seed có Q cao hơn, không chứng nhận toàn grid.

`resume.json` ghi prior elapsed160.671s + resume active198.078s =358.749s; downtime không tính, search không chạy lại. Chi phí worker full confirmations/attempts online230.108s, offline0; ngân sách bổ sung không bị giấu dưới cap60s ban đầu. Audit đối chiếu15 request/result, transfer registry chỉ đếm14 lượt thành công và không dùng run partial để tăng complete-domain coverage.

## Chuyển môi trường và đường chạy end-to-end

`runs/transfer_05/20261013_001932_401031/TRANSFER.md` đối chiếu5 run gồm ViECAP CUDA, CIFAR CPU, Fashion CPU, Fashion CUDA và Fashion partial: **4 run hoàn chỉnh, 3 dataset content/split, 1 GPU vật lý**. Có60 full rows được kiểm ở4 run hoàn chỉnh;14 rows thành công của run partial được ghi riêng. Cặp Fashion CPU/CUDA cùng protocol số học khai báo đều chọnc05, runtime khác được ghi rõ; không phải hai GPU độc lập hoặc chứng minh predictor chuyển miền.

Đã chạy thật `train --selection` lấy lựa chọn Fashion CUDA sang CPU, đồng thời đổi horizon thành1epoch. Training hoàn tất1/1, sinh checkpoint/report tại `runs/deploy_portability_05/20261013_002015_887652/training_report.json`; CLI cảnh báo khác hardware/runtime/horizon/config và **không giữ chứng nhận tối ưu cũ**. Đây là kiểm tra luồng thực thi/cảnh báo, không phải benchmark chất lượng1epoch.

## Kiểm thử và cài đặt

- **93/93 tests** qua trên `.venv` CUDA và `.venv-cpu`, gồm scheduler không loại late-starter, ba mức bằng chứng, online không có oracle ẩn, split/content identity, CPU/GPU không đếm trùng, parser isolation, CUDA UUID/NVML remapping và identity fallback.
- `compileall` thành công; `uv pip check` báo48 packages tương thích trong mỗi môi trường.
- Package source và editable metadata cùng **0.5.0** ở cả hai môi trường. Cài lại package bằng `--no-deps`, không thay torch/torchvision để làm đẹp phép so sánh.
- Các run/checkpoint cũ được giữ nguyên. Số đo cũ không bị nâng thành bằng chứng chuyển GPU mới.

Sau khi tất cả phép đo/resume/deployment kết thúc, sửa riêng metadata CLI: `verify-config --help` ghi rõ chi phí online và chỉ nhận policy production, không kế thừa dòng "offline reference is separate" của `research`. Thêm regression test cho help/choices và chạy lại đủ93tests. Không đổi model/optimizer/scheduler/training tensor code. Source snapshot vẫn giữ nguyên bản thực sự đã đo; source guard bảo thủ sẽ báo khác `cli.py` khi dùng lại lựa chọn cũ, hoặc từ chối resume cũ cho tới khi phục hồi đúng snapshot. Không sửa checksum lịch sử để giả vờ cùng source.

## Giới hạn không được bỏ qua

Chỉ một GPU vật lý hiện có. GPU verification đã hoàn tất nhưng chỉ chứng minh đo lại local trên CUDA hiện tại, **không chứng minh chuyển sang GPU khác đã thành công**. Cần đưa cùng CLI sang GPU khác và thu raw measurements; transfer-report giữ `unseen_gpu_validation=not_established` khi chưa có bằng chứng đó. Không có đảm bảo tối ưu mọi dataset/máy/kiến trúc/epoch. `verify-config` hiện hỗ trợ caption/classification/reconstruction; detection còn dùng nhánh tuning riêng, chưa có full-reference adapter cho chứng nhận mới.
