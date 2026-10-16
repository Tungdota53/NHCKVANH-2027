# Phương pháp và các điểm cần sửa trong đề xuất

Nhánh thực nghiệm mới cho captioning/classification/reconstruction được mô tả tại [RESEARCH_METHOD.md](RESEARCH_METHOD.md): objective cố định, decisions khóa trước exhaustive full-reference, nhiều seed, baseline và regret/Pareto/HV. Nhánh này không dùng early metric làm nghiệm đối chứng hoặc gọi cấu hình `auto` là tối ưu toàn cục. Phần bên dưới giữ thiết kế predictor/detection cũ, không phải protocol mới.

Phiên bản 0.2 đã bổ sung pipeline tự nhận diện task, CPU/CUDA/MPS và các adapter dataset. Xem `PIPELINE.md` cho phương pháp hiện tại áp dụng cho captioning/classification/reconstruction/detection và `PIPELINE_VALIDATION.md` cho phép đo thực tế. Các phần dưới là thiết kế nghiên cứu chuyên biệt của nhánh detection, vẫn được giữ để so sánh và mở rộng predictor offline.

## Định nghĩa bài toán

Với cấu hình c, dataset D, phần cứng H và protocol P:

- A(c,D,P): mAP50-95 sau huấn luyện đầy đủ và validation trên split cố định.
- T(c,D,H,P): wall time toàn bộ, kể cả data loading, validation và checkpoint.
- V(c,D,H,P): peak bộ nhớ huấn luyện/validation; phân biệt allocator PyTorch với tổng bộ nhớ thiết bị.
- B: ngân sách tìm kiếm. B không phải ngân sách huấn luyện đầy đủ.

Tìm tập cấu hình không bị trội theo tối đa A, tối thiểu T,V. Nếu có ràng buộc, dùng cận dưới accuracy và cận trên memory/time đã hiệu chỉnh; thiếu predictor đủ điều kiện thì trả trạng thái chưa chứng nhận, kèm phép đo. Preset Balanced là quy tắc quyết định có công bố trọng số; nó không phải tối ưu duy nhất cho mọi người.

## Điều chỉnh các giả định chính

| Trong tài liệu | Cách xử lý |
|---|---|
| Hai điểm batch 1 và 2 đủ suy ra batch 16/32 an toàn | Hai điểm chỉ tạo đề xuất. Allocator, kernel/workspace, nhãn nhiều vật thể và validation có thể tạo điểm gãy. Giới hạn bước ngoại suy và bắt buộc chạy thật batch đề xuất. |
| Batch lớn nhất vừa VRAM là cấu hình tốt nhất | Profile throughput; batch lớn hơn có thể tăng ít hoặc giảm tốc độ. Chọn batch gần throughput tối đa với VRAM thấp hơn. |
| Gradient accumulation giữ kết quả học tương đương | Chỉ gần tương đương. BatchNorm, augmentation, số optimizer step, warmup và rounding làm khác kết quả. Ghi batch thực, nbs và accumulation sau warmup. |
| mAP50/loss sớm dự đoán được accuracy cuối | Chỉ là proxy xếp hạng. Learning curve có thể giao nhau. Muốn dự đoán final mAP50-95 phải có số đo full training và kiểm định trên dataset/GPU chưa dùng để fit. Loss không so trực tiếp giữa các họ model có scale khác. |
| Loại cấu hình bằng accuracy dự đoán ngay từ đầu | Với sai số lớn dễ loại nhầm cấu hình tốt. Bản đầu chỉ loại bằng tính hợp lệ/VRAM đo; đánh giá các ứng viên còn lại trước khi halving. Nghiên cứu sau có thể dùng dominance chắc chắn với các khoảng đồng thời. |
| 10–30 cấu hình × vài epoch nằm trong 20 phút | Không bảo đảm, đặc biệt trên GPU laptop. Tính cả startup, profile, val, checkpoint. Áp deadline ngoài process; thu hẹp ứng viên hoặc trả kết quả một phần có provenance. |
| Knee point theo khoảng cách tới mặt phẳng cực biên luôn xác định | 3D có thể suy biến, nhiều knee hoặc nhạy chuẩn hóa. Bản đầu dùng augmented Tchebycheff, ghi rõ trọng số. So sánh knee/TOPSIS trong thực nghiệm riêng. |
| LR tăng tuyến tính theo effective batch là quy tắc chung | Phụ thuộc optimizer/protocol. Giữ AdamW và effective batch mục tiêu; không áp linear scaling một cách mặc định. Thử LR ba điểm nếu còn thời gian. |
| Vượt 95% VRAM thì đổi batch và resume ngay | Cần làm mới dataloader, optimizer/scaler/checkpoint và xử lý val. Bản đầu chỉ recovery OOM có giới hạn ở tiến trình mới, từ checkpoint epoch gần nhất, giữ nbs và báo rõ phần bị chạy lại. |
| Conformal trên Colab bảo đảm VRAM máy bất kỳ | Coverage đòi hỏi phân phối phù hợp. GPU/driver/torch/kernel và dataset mới gây dịch miền. Kiểm tra miền, calibration theo nhóm, đo lại online và không gọi margin thủ công là khoảng tin cậy. |
| Online phải dùng predictor đóng gói | Cần chế độ cold start đo thật khi chưa có số đo offline. Không thể tạo một predictor đã kiểm định chỉ từ mô tả bài toán. |

## Thiết kế offline để có kết luận nghiên cứu

1. Cố định detection task, metric, epoch/horizon, dữ liệu train/val, preprocessing, pretrained source và optimizer. Ghi phiên bản torch, Ultralytics, driver, GPU, seed và scheduler. Freeze phải so theo cấu trúc thực của từng họ YOLO; cùng số layer chưa chắc cùng tỷ lệ tham số.
2. Thu số đo trên nhiều dataset có lớp/kích thước/độ khó khác nhau và nhiều GPU. Không chỉ đo cùng một COCO subset trên Colab. Colab có thể đổi GPU giữa phiên; ghi GPU của từng phép đo.
3. Khởi đầu với thiết kế rải đều khả thi (Sobol hoặc thiết kế factorial rút gọn), đo cả cấu hình thất bại/OOM. OOM là thông tin censored/biên khả thi, không gán số memory/accuracy tùy ý.
4. Tách theo dataset/source và GPU cho đánh giá chuyển miền. Tách train/calibration/test trước khi chọn mô hình; không fit trên calibration/test. Làm group conformal, đo empirical coverage và độ rộng khoảng riêng từng nhóm. Một đảm bảo pointwise không tự trở thành đảm bảo cho hàng chục cấu hình được chọn thích nghi; dùng simultaneous calibration/Bonferroni hoặc kiểm định finalist độc lập khi cần cam kết.
5. Đánh giá ridge/gradient boosting trước. Chỉ thêm GP/qNEHVI nếu dữ liệu và chi phí đủ để chứng minh lợi ích. Early learning-curve features cần predictor riêng fit từ cùng protocol; không trộn prior và early mAP tùy ý rồi giữ nguyên khoảng conformal cũ.
6. Dùng cận VRAM/thời gian và kiểm tra xác nhận online để chuyển máy. Mô hình tuyến tính theo batch nên local theo model × imgsz × precision × freeze × optimizer, có điểm xác nhận; không dùng một slope chung cho toàn bộ họ YOLO.

## Thực nghiệm cần báo cáo

Baseline bắt buộc: mặc định Ultralytics cùng pretrained/dataset; AutoBatch với mức bộ nhớ đã chọn; random search cùng wall budget; Successive Halving không predictor; phiên bản đề xuất có predictor. Bayesian multi-objective chỉ thêm khi cùng ngân sách và có thực thi đáng tin cậy.

Đánh giá full train các finalist, không dùng early mAP làm đáp án. Giữ test cuối độc lập, không dùng test để tuning. Với mỗi dataset/GPU chạy ít nhất ba seed khi tài nguyên cho phép; báo trung bình và độ phân tán của mAP cuối, wall time, peak VRAM, tỷ lệ OOM, chi phí tìm kiếm và độ ổn định cấu hình.

Metric trọng tâm:

- Search overhead / full-training time và số phút vượt ngân sách.
- Regret accuracy so với cấu hình tốt nhất được đo trong ngân sách đối chứng; không gọi là global optimum nếu không đo đủ không gian.
- Pareto hypervolume với cùng reference point, feasibility rate và tỷ lệ giữ được cấu hình tốt sau halving.
- MAE/relative error thời gian, VRAM; empirical interval coverage và interval width; đánh giá riêng in-domain và held-out GPU/dataset.
- Chất lượng small-object theo metric phù hợp nếu đây là ưu tiên ứng dụng.

Ablation: bỏ profile online; bỏ margin; chỉ batch 1/2 so với có điểm xác nhận; batch tối đa so với throughput; bỏ freeze; bỏ LR refinement; chia ngẫu nhiên theo row so với chia nhóm dataset; proxy mAP50 so với predictor learning curve.

## Phạm vi bản mã nguồn hiện tại

Đã có: đọc hardware, lấy mẫu multilabel cố định, memory/throughput probes có optimizer state, giới hạn ngoại suy, deadline process, successive halving cùng fidelity, Pareto/preset, LR refinement, xuất cấu hình, full training và recovery OOM, ghi nhận full runs, ridge + group conformal, từ chối OOD và ngưỡng accuracy không chứng nhận được.

Chưa có: predictor pretrained từ số đo thật, GP/GBDT/learning-curve extrapolation, qNEHVI, multi-GPU, kiểm tra ảnh gần trùng hoặc video groups tự động, resume bitwise-identical, hay bằng chứng thực nghiệm về ưu thế accuracy trên dataset thật. Đây là các công việc nghiên cứu cần dữ liệu và đánh giá, không phải các kết quả có thể khẳng định từ tài liệu.

Nguồn kỹ thuật: [Ultralytics train API](https://docs.ultralytics.com/modes/train/), [Ultralytics trainer](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py), [PyTorch memory_reserved peak](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html), [PyTorch CUDA memory API](https://docs.pytorch.org/docs/stable/cuda.html).
