# Đo lại cấu hình trên GPU hoặc dataset mới — bản 0.5

## Một lệnh cho môi trường mới

```powershell
.venv\Scripts\python.exe train_config.py verify-config --data "D:/datasets/my_dataset" --device auto
```

Không sử dụng nghiệm cũ của RTX3050 hoặc ViECAP làm đáp án cho dữ liệu/máy mới. Lệnh discover lại dữ liệu, kiểm tra split/hash, dò runtime hiện tại, khóa objective/grid, đo chi phí trên máy đó và full-train cấu hình mới. Vocabulary/labels lấy từ train của dataset hiện tại. Không fit predictor từ vài dòng dữ liệu rồi tuyên bố chuyển miền đã được chứng minh.

Mặc định: grid `expanded`, 3 seed17/29/43, horizon8 epoch, cap10 phút mỗi seed, total timeout3 giờ. Đổi số epoch bằng `--full-epochs`; đây là một protocol mới, không kế thừa chứng nhận8epoch. Ví dụ:

```powershell
.venv\Scripts\python.exe train_config.py verify-config --data "ViECAP-Image-Captioning-Vietnamese-master" --full-epochs 20 --search-budget 30 --timeout-hours 6 --caption-encoder resnet18 --offline
```

30 phút là cap tìm kiếm mỗi seed, không phải cam kết mọi dataset sẽ đo đủ trong30phút. Không giảm epoch/hyperparameter ngầm để tạo báo cáo thành công. CPU là fallback khi runtime phù hợp; thiếu runtime, thiếu validation hoặc task chưa có adapter sẽ báo lỗi. Các lệnh `auto`/`tune` cũ giữ nguyên để tương thích và không được nâng nhãn thành tối ưu verified.

`verify-config` hiện có full-reference adapter cho **captioning, classification, reconstruction**. Detection tiếp tục dùng `auto`/`tune` riêng, chưa có chứng nhận full-grid ở lệnh mới. Định dạng khác cần adapter; ảnh không nhãn chỉ cho bài toán reconstruction, không tự trở thành bài toán có nhãn mà người dùng định huấn luyện. Kết quả kiểm chứng thực tế0.5: [PORTABLE_VALIDATION.md](PORTABLE_VALIDATION.md).

## Adaptive verified: không loại cấu hình học chậm

1. Profile **mọi ứng viên** cùng subset/rung cao nhất (mặc định4epoch). Q proxy chỉ dùng cho chẩn đoán, không dùng để loại ứng viên hoặc chứng nhận chất lượng cuối.
2. Dự kiến chi phí sweep = tổng full-time ngoại suy ×1.25 +2giây/ứng viên. Đây là heuristic scheduling margin, **không phải** confidence bound hoặc bảo đảm timeout.
3. Nếu projected sweep nằm trong budget còn lại: đo full-data/full-horizon mọi cấu hình. Nếu không: khám phá phân tầng LR trước khi quay lại cùng LR, lần lượt full-confirm các cấu hình còn có thể thử trong ngân sách. Cấu hình chưa thử/OOM/timeout vẫn là unresolved, không bị coi là kém chất lượng.
4. Chỉ các trial **full data + đủ horizon + đủ mọi seed** mới được đưa vào lựa chọn. Không dùng proxy làm lựa chọn đã verified. Production giữ thứ tự candidate cố định giữa các seed để budget ít vẫn có khả năng xác nhận cùng một cấu hình đủ seed; nghiên cứu baseline vẫn xoay/shuffle theo seed.
5. Tất cả profile/startup/train/validation/checkpoint tính vào budget online. **Không có exhaustive oracle offline miễn phí** trong `verify-config`. Giá của tính chắc chắn trong một grid hữu hạn là phải đo đầy đủ grid đó.

Heuristic mới được phát triển từ các thất bại đã thấy trên ViECAP/CIFAR; không gọi kết quả cải tiến trên hai bộ đó là blind held-out validation. Không phải thuật toán tối ưu toàn cục hoặc SOTA mới.

## Ba trạng thái phân biệt rõ

| Trạng thái | Bằng chứng | Ý nghĩa |
|---|---|---|
| `empirical_optimal_in_declared_space` | Mọi cấu hình × mọi seed đã full-train đủ horizon, có cấu hình feasible | Nghiệm tốt nhất theo objective cố định trong tập đã khai báo và số đo hiện tại |
| `fully_trained_best_observed` | Có cấu hình full-confirm đủ seed, nhưng chưa đo hết không gian | Có thể train/deploy cấu hình đã xác nhận; **chưa chứng nhận tối ưu** |
| `no_verified_feasible_selection` | Chưa có cấu hình đủ seed/horizon và đạt constraints | Không tạo lựa chọn đã verified; báo thiếu budget/không khả thi |

Mean/SD/min/max seed là biến thiên quan sát, không là confidence interval cho seed/máy tương lai. Caption Q là(BLEU4+chrF)/200 của checkpoint validation tốt nhất trong horizon; classification là val accuracy epoch cuối; reconstruction dùng quality_score riêng. Không có một cấu hình tối ưu duy nhất khi mục tiêu quality/time/memory/weights khác nhau.

Grid expanded classification/reconstruction: width16/32 × LR0.001/0.0003/0.003, ảnh32, batch16, effective32. Caption: hidden128/256/384 × LR0.001/0.0003, batch8/effective64, cùng encoder/preprocessing của lượt đo. Có thể mở rộng bằng `--grid` JSON trong phạm vi hyperparameter thực sự được trainer hỗ trợ. Grid nhỏ không bao phủ mọi kiến trúc/optimizer/batch/image size.

`--precision fp32` giúp khóa cùng precision khi so CPU/CUDA. Không coi FP16GPU/FP32CPU hoặc RSSCPU/allocatorGPU là cùng phép đo vật lý. Nếu model/dataset quá lớn, cần grid/budget/resource limits phù hợp; không thể bảo đảm chạy trên mọi máy thiếu RAM/VRAM.

## Xem, audit và tiếp tục

```powershell
.venv\Scripts\python.exe train_config.py research-show --run "runs/verified/<run>"
.venv\Scripts\python.exe train_config.py research-audit --run "runs/verified/<run>"
.venv\Scripts\python.exe train_config.py research-resume --run "runs/verified/<run>" --timeout-hours 3
```

`research-resume` không replay search như thể budget chưa tiêu. Sau khi decisions khóa, nó chỉ full-train lại các config/seed chưa hoàn tất, cùng source/hardware/runtime/data; restart initialization thay vì nối checkpoint thiếu thời gian đo. Chi phí bổ sung được ghi riêng, không giấu trong budget ban đầu. Mất quyền kiểm tra process còn sống thì từ chối resume; không giả định process đã chết. Artifact source snapshot cũ cho phép phục hồi đúng version nếu cần tiếp tục sau nâng cấp mã.

Sau khi đổi machine/data/epoch/config/source/runtime, `train --selection` tiếp tục hạ optimality scope hoặc từ chối nếu lựa chọn không đủ full-confirm. Muốn có kết luận mới phải dùng `verify-config` trên môi trường mới, không chỉ copy `selected.json`.

Chạy training từ cấu hình đã xuất:

```powershell
.venv\Scripts\python.exe train_config.py train --selection "runs/verified/<run>/selected.json" --device auto --output runs/production
```

Không thêm `--epochs` nếu muốn giữ nguyên horizon đã đo. Cấu hình chỉ được verified trong grid/objective của run; không gọi checkpoint của một seed riêng là optimum chất lượng phổ quát.

## Báo cáo nhiều môi trường không tạo bằng chứng giả

```powershell
.venv\Scripts\python.exe train_config.py transfer-report --runs "runs/research/<run_cpu>" "runs/verified/<run_gpu>" --output runs/transfer
```

Report đối chiếu protocol/decisions và request/result gốc, tính lại optimum từ full measurements. Dataset content ID bỏ đường dẫn nhưng giữ SHA ảnh, label/caption và split; GPU vật lý chỉ đếm khi có UUID thật. CPU và CUDA trên cùng laptop **không** là hai GPU/máy độc lập. GPU thiếu UUID không được tự tạo ID để tăng coverage.

CUDA logical ordinal được đối chiếu UUID với NVML thay vì giả định bằng physical ordinal; thiếu đối chiếu thì đánh dấu physical identity chưa verified. MIG instance IDs không được lấy làm các GPU vật lý độc lập khi có parent UUID. Đây là kiểm tra consistency từ dữ liệu do chương trình thu, không phải cryptographic hardware attestation chống báo cáo giả thủ công.

Paired backend evidence đòi cùng nội dung/split, grid (không tính đường dẫn cache), horizon, seed, metric và objective/constraints. Runtime khác vẫn phải ghi rõ: không quy mọi chênh lệch thành hiệu ứng GPU. Không gộp Q hoặc memory giữa task/backend không cùng metric. Dataset subsets khác của cùng nguồn cũng không tự trở thành các miền nguồn độc lập.

Nếu chỉ có một UUID CUDA, report ghi `unseen_gpu_validation=not_established`. Dù đo nhiều GPU cũng không có bảo đảm tối ưu với mọi GPU/dataset tương lai. Bằng chứng cần để nghiên cứu tiếp: nhiều GPU vật lý/máy độc lập, dataset nguồn mới, nhiều budget/horizon/grid, outer test chưa dùng, và predictor/calibration riêng nếu muốn đưa ra khoảng dự đoán chuyển miền.

## Dữ liệu kiểm chứng mới

`scripts/prepare_fashion_research.py` lấy Fashion-MNIST từ [repo chính thức của Zalando](https://github.com/zalandoresearch/fashion-mnist), qua torchvision kiểm MD5. Giữ nhãn gốc, chia train/val từ original train và test từ original test, loại trùng nội dung giữa các split, lưu chỉ số nguồn. Mặc định1000train/300val/500test, seed101; đây là subset, không phải benchmark toàn bộ60.000train. Test metric không dùng để chọn cấu hình.
