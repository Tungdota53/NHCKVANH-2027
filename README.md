# Pipeline tự động chọn cấu hình huấn luyện ảnh

## Bản hiện tại: 0.10.0 — thuật toán adaptive_full

Phương pháp mới chọn cấu hình tiếp theo bằng kernel-ridge quality surrogate từ **full-data/full-horizon/đủ seeds đã đo**, kết hợp optimistic improvement, uncertainty/novelty heuristic và chi phí xác nhận các seeds còn thiếu. Không dùng early-quality để loại ứng viên. Training là phép đo của bộ tối ưu, không phải đầu ra nghiên cứu duy nhất.

Mặc định menu/optimize dùng `adaptive_full` với weighted objective; `reserve`, `random`, `default`, `successive_halving` và ba `adaptive_no_*` là đối chứng/ablation. Không tuyên bố phát minh ra cost-aware HPO, no-regret bound, hoặc bảo đảm thắng Random/mọi GPU/dataset.

Thiết kế và phương trình: [ALGORITHM_010_METHOD.md](ALGORITHM_010_METHOD.md); kế hoạch: [ALGORITHM_010_PLAN.md](ALGORITHM_010_PLAN.md); thực nghiệm khóa trước: [ALGORITHM_010_EXPERIMENTS.md](ALGORITHM_010_EXPERIMENTS.md); nghiệm thu: [ALGORITHM_010_VALIDATION.md](ALGORITHM_010_VALIDATION.md).

### Lấy mã nguồn từ GitHub

Dùng Python 3.12 trên Windows cho cấu hình cài đặt bên dưới. Sau khi clone, chọn **một** backend:

```powershell
git clone https://github.com/Tungdota53/NHCKVANH-2027.git
cd NHCKVANH-2027
.\setup-v2.ps1 -Backend cpu
# Hoặc GPU NVIDIA tương thích CUDA 12.4: .\setup-v2.ps1 -Backend cu124
.\cli.ps1
```

Repository gồm mã nguồn, tests, cấu hình và tài liệu; **không kèm dataset, weights, môi trường ảo hoặc thư mục `runs/`**. Các đường dẫn `runs/...` trong tài liệu nghiệm thu/manifest là provenance của thực nghiệm cục bộ, không phải artifact có sẵn sau clone. Chuẩn bị dataset riêng trước khi optimize; các cấu hình dùng local weights cần có file tương ứng hoặc dùng model identifier được backend hỗ trợ. Không thể audit lại raw experiments chỉ từ báo cáo Markdown.

Chạy kiểm thử sau khi cài (thay interpreter nếu chọn backend CUDA):

```powershell
.venv-v2-cpu\Scripts\python.exe -m unittest discover -s tests -q
```

```powershell
.\cli.ps1
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\images" --method adaptive_full --budget-seconds 1200
# Chỉ audit/phân tích các benchmark studies đã hoàn tất, không training:
.venv\Scripts\python.exe -m research_pipeline.v2 algorithm-report --manifest studies.json --output runs/algorithm_analysis_new
```

`report.json` có acquisition_policy và acquisition_history (bootstrap/partial priority/model-fit/ranking/reason). Chỉ measured full outcomes được chọn. Uncertainty chưa hiệu chuẩn; budget là soft end-to-end, có thể không đủ khởi tạo/xác nhận một cấu hình. Không được coi exit0 hoặc nhiều tests pass là chứng minh thuật toán hiệu quả hơn baseline.

## Nền thực thi 0.9.1 được giữ tương thích

Pipeline nghiên cứu: trong **tập cấu hình đã khai báo**, chọn cân bằng validation quality / thời gian / bộ nhớ trên phần cứng đang đo. Chỉ full-data/full-horizon/đủ seeds mới được chọn. Không hứa tối ưu toàn cục, mọi GPU hoặc mọi loại dataset.

```powershell
.\cli.ps1
# Hoặc gọi trực tiếp, không cần activate:
.venv\Scripts\python.exe -m research_pipeline.v2
.venv\Scripts\python.exe -m research_pipeline.v2 inspect --data "D:\datasets\my dataset"
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\my dataset" --device auto --budget-seconds 1200
```

Mặc định menu/lệnh mới: **20 phút**, 20 epoch/full lượt, 3 seeds, grid core12. Đây không phải dự báo ngân sách đủ để đo hết grid. Đầu vào hỗ trợ detection, classification ảnh, captioning có nhãn/validation độc lập; **chưa hỗ trợ bảng CSV/tabular hay mọi framework AI**.

0.9.1 sửa OOD predictor fallback, bảo vệ ngân sách SH, preset Pareto-safe, hash streaming/cache kiểm chứng, cache ảnh decode có trần RAM, DataLoader Windows-spawn, pinned transfer CUDA, package entrypoint và lựa chọn interpreter. Backend mới nằm riêng trong v2; không sửa source/checkpoint legacy. Kế hoạch: [UPGRADE_091_PLAN.md](UPGRADE_091_PLAN.md); phương pháp: [METHOD_091.md](METHOD_091.md); nghiệm thu: [UPGRADE_091_VALIDATION.md](UPGRADE_091_VALIDATION.md); cách dùng: [CLI_GUIDE.md](CLI_GUIDE.md).

```powershell
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\my dataset" --workers auto --image-cache-mb 64 --pin-memory auto
# Chỉ dựng lại benchmark đã hoàn tất; KHÔNG training:
.venv\Scripts\python.exe -m research_pipeline.v2 benchmark-report --run "runs/benchmark/<study>"
```

Cache64MiB là trần tổng decoded-image caches, còn bị chặn ở10% RAM khả dụng; không bao gồm model/queue/pinned buffers. Detection chưa bật RAM cache YOLO không giới hạn. Auto workers=0 cho CPU/dataset nhỏ; tối đa2 cho CUDA từ8192ảnh. Đây là policy bảo thủ, **không phải số workers tối ưu đã đo**. Đo throughput/RSS trước khi tăng workers trên Windows.

Quy tắc weighted/Tchebycheff/TOPSIS/knee, Pareto/presets và predictor point prior trong0.9 được giữ lại. Predictor OOD phải đo lại; predicted khác measured. Knee không dùng weights trừ fallback Tchebycheff. Budget là soft end-to-end có dự phòng audit, không hard deadline; full hash đầu/cuối bắt buộc, cache metadata giữa trials không là chống giả mạo.

## Lịch sử và các nhánh tương thích

Những số đo/lệnh dưới đây thuộc phiên bản và scope ghi rõ, **không là nghiệm thu0.9.1**. Lệnh `--data` không có `optimize` vẫn chạy legacy; dùng các lệnh v2 bên trên cho nghiên cứu hiện tại.

**Sửa lỗi CLI 0.8.1:** phân biệt CSV caption với bảng CSV/TSV, giải thích đúng lỗi thiếu ảnh/file nén và cho nhập lại dataset ngay tại bước1. `marketing_campaign.csv` là dữ liệu bảng: hiện chưa có nhánh tabular, không tự coi bảng là dữ liệu ảnh hoặc đoán cột mục tiêu. 207 tests; xem [CLI_GUIDE.md](CLI_GUIDE.md).

**CLI 0.8.0:** chạy `.venv\Scripts\python.exe train_config.py` (không tham số) hoặc `.\cli.ps1` để mở menu tiếng Việt. Có hướng dẫn chọn dataset/budget theo phút, xem lại và xác nhận trước training, tiến độ theo epoch, quản lý lượt chạy, audit/resume/train/test riêng, lỗi kèm cách xử lý và chế độ JSON cho script. Các lệnh cũ/checkpoint vẫn giữ nguyên; protocol cũ khóa source sẽ không bị ép resume bằng mã mới. Hướng dẫn: [CLI_GUIDE.md](CLI_GUIDE.md); kiểm chứng: [CLI_VALIDATION.md](CLI_VALIDATION.md).

Bản **0.7.1 / research v2**: `python train_config.py optimize --data <dataset> --device auto`. Training thật, budget chung giữa seeds, incumbent/reserve, profile có giới hạn, phase telemetry, event ledger, audit/resume và benchmark reference độc lập. Hướng dẫn: [PIPELINE_V2.md](PIPELINE_V2.md); phương pháp: [PROTOCOL_V2.md](PROTOCOL_V2.md); kết quả: [UPGRADE_VALIDATION.md](UPGRADE_VALIDATION.md); tiến độ/phần chưa hoàn thành: [UPGRADE_PROGRESS.md](UPGRADE_PROGRESS.md). Không suy ra tối ưu mọi GPU/dataset hoặc predictor có safety guarantee từ việc pipeline chạy được.

Nhánh nghiên cứu detection đã thêm qua extension `research_pipeline`: `python train_config.py research --data <detection_dataset> --task detection`. Có actual-config verification, full-reference độc lập, đối chứng/ablation và sensitivity. Xem [RESEARCH_PLAN.md](RESEARCH_PLAN.md), [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), [DETECTION_RESEARCH.md](DETECTION_RESEARCH.md) và [DETECTION_VALIDATION.md](DETECTION_VALIDATION.md). Package0.5 bên dưới được giữ bất biến để bảo toàn job/checkpoint cũ; không dùng số liệu pilot để tuyên bố optimum mọi GPU/dataset.

Bản **0.5** thêm `verify-config --data <dataset> --device auto`: profile tại máy/dataset hiện tại, không loại cấu hình học chậm theo Q sớm, bắt buộc full-confirm nhiều seed và phân biệt **optimal trong grid đã đo đủ** với **best observed chưa đo hết**. `transfer-report` đối chiếu nhiều lượt chạy theo data hash/GPU UUID, không coi CPU/CUDA cùng laptop là nhiều GPU. Hướng dẫn và giới hạn: [PORTABLE_VERIFICATION.md](PORTABLE_VERIFICATION.md).

Đã chạy thật bản0.5 trên Fashion-MNIST subset: grid6×3seed×8epoch ở CPU và RTX3050, đều chọn CNN width32/LR0.003; val accuracy78.33%±3.84điểm%, không phải test accuracy. Đã kiểm tra thiếu budget/resume và93/93 tests ở cả hai runtime. Xem [PORTABLE_VALIDATION.md](PORTABLE_VALIDATION.md) cho số đo, chi phí và giới hạn; GPU vật lý khác vẫn chưa được kiểm chứng.

Đã thêm nhánh **nghiên cứu xác định cấu hình tối ưu**: `train_config.py research --data <dataset>`. Nhánh này khóa objective/grid/protocol trước search, so sánh default/random/Successive Halving/Pareto racing, full-train mọi cấu hình với nhiều seed, xuất regret/feasibility/HV và sai số proxy. Xem [RESEARCH_METHOD.md](RESEARCH_METHOD.md) để chạy thực nghiệm và hiểu phạm vi “tối ưu trong tập cấu hình đã đo”, khác với tìm kiếm vận hành bên dưới.

Bản 0.4 đã kiểm chứng thật **24 full-reference runs** (ViECAP CUDA + CIFAR-10 subset CPU, mỗi bộ 4 cấu hình × 3 seed × 8 epoch), audit đầy đủ và 78/78 tests ở cả hai runtime. Nghiệm balanced: ViECAP hidden256/batch32/LR0.0003; CIFAR CNN width32/LR0.001. Racing chưa chứng minh superiority: thua default về regret trên ViECAP và thua Random full trên CIFAR. Xem [RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md) cho số đo/giới hạn; không coi đây là tối ưu toàn cục hoặc kiểm chứng20epoch.

Lệnh tương thích legacy nhận đường dẫn dataset và tự chọn task, thiết bị, model, batch, precision và learning rate:

```powershell
.venv\Scripts\python.exe train_config.py --data "ViECAP-Image-Captioning-Vietnamese-master"
```

Thêm `--train` để sau search chạy full training và tự đánh giá test caption (nếu có test). Ví dụ: `.venv\Scripts\python.exe train_config.py --data "ViECAP-Image-Captioning-Vietnamese-master" --train`.

Hỗ trợ YOLO detection (YAML/thư mục nhãn), COCO boxes/captions, VOC XML, LabelMe rectangles, caption CSV, phân loại ImageFolder và ảnh chưa gán nhãn. Caption v2 dùng encoder pretrained ResNet18 (hoặc attention CNN khi offline không có weights), GRU đọc ngữ cảnh ảnh ở từng bước và spatial cross-attention. Search/checkpoint chọn theo caption sinh trên validation, có đối chứng, LR scheduler và early stopping; không chỉ chọn theo loss. Ảnh không nhãn đi vào reconstruction pretraining. Máy có CUDA/MPS sẽ được chọn nếu runtime chạy được, nếu không dùng CPU. Kết quả gồm `selected.json`, `train.yaml`, `REPORT.md`, lịch sử phép đo và preset. Checkpoint caption cũ vẫn đọc được.

Đã train/đánh giá bản v2 thật trên ViECAP: test BLEU-4 **33,95** / chrF **46,04**, so với bản cũ **23,20** / **41,14** và đối chứng không nhìn ảnh **28,05** / **43,20**. Đây không phải % caption đúng hoặc bảo đảm tối ưu toàn cục. Chi tiết và đường dẫn checkpoint ở `PIPELINE_VALIDATION.md`.

Xem **[PIPELINE.md](PIPELINE.md)** cho cách dùng mới và phạm vi thực thi; **[PIPELINE_VALIDATION.md](PIPELINE_VALIDATION.md)** cho kết quả kiểm thử. Các phần bên dưới là hướng dẫn chi tiết cho nhánh YOLO detection cũ vẫn hoạt động.

Đây là bản thử nghiệm thực thi bài toán trong [tài liệu gốc](https://docs.google.com/document/d/1PAzKRcWxnIWE0GDGnTQyB8YhTbFNH4_d9u7xvb4ggA0/edit): chọn Model, Image size, Batch, Precision, Freeze, Optimizer và LR dựa trên máy và dataset. Tìm kiếm tạo Pareto và bốn preset Accuracy / Time / Memory / Balanced.

Nhánh YOLO hỗ trợ **object detection, một GPU CUDA/MPS hoặc CPU, dataset YOLO cục bộ**. Không cần bộ dự đoán offline để chạy. Khi chưa có bộ dự đoán, độ chính xác được ghi là **mAP50 thử nghiệm**, không phải độ chính xác cuối. Mã nguồn có công cụ thu thập số đo full training và huấn luyện predictor offline với khoảng dự đoán conformal theo nhóm dataset.

## Chạy trên Windows

Môi trường `.venv` tại máy này hiện có PyTorch 2.6.0+cu124 và đã chạy được training trên RTX 3050. Môi trường `.venv-cpu` riêng dùng PyTorch CPU và cũng đã chạy pipeline tự động thành công. Cả hai có thể dùng `--device cpu`; không cần cài lại CUDA để chạy thử.

Chạy ngay một lượt thử ngắn bằng dataset tổng hợp đã tạo:

```powershell
.venv\Scripts\yolo-tune.exe tune --data runs/smoke_data/data.yaml --models yolo11n.yaml --imgsz 128 --freeze 0 --max-batch 2 --device cpu --full-epochs 3 --rungs 1 2 --budget 2 --force-search
```

Kết quả kiểm tra và đường dẫn artifact có trong `VALIDATION.md`.

Mở PowerShell tại thư mục dự án. Một môi trường `.venv` được dùng riêng cho dự án.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124 --system-certs
uv pip install --python .venv\Scripts\python.exe -e . --index-url https://pypi.org/simple --system-certs
.venv\Scripts\python.exe -m yolo_tuner inspect
```

Chọn bản CUDA/PyTorch tương thích theo [trang cài đặt PyTorch](https://pytorch.org/get-started/locally/) nếu máy khác cần bản khác. Thư viện Ultralytics có giấy phép riêng; xem [nguồn chính thức](https://github.com/ultralytics/ultralytics).

Dataset cần YAML chỉ tới ảnh/nhãn local đã tồn tại; mã không tự tải dataset. Ví dụ:

```yaml
path: D:/datasets/my_dataset
train: images/train
val: images/val
names: [person, helmet]
```

Nhãn tương ứng nằm tại `labels/train`, `labels/val`, định dạng `class x_center y_center width height`, tọa độ chuẩn hóa 0–1. Ảnh không có nhãn được hiểu là ảnh âm tính. Phải chia train/val từ trước, theo video/nguồn/chủ thể nếu có.

```powershell
.venv\Scripts\python.exe -m yolo_tuner tune --data D:/datasets/my_dataset/data.yaml --budget 20
```

Mặc định thử `yolo11n.pt` và `yolo11s.pt`, ảnh 480/640, freeze 0/10, tối đa 8 ứng viên, các mức thử 1/2/4 epoch. File `.pt` có thể được Ultralytics tải từ nguồn chính thức khi chưa có; đặt đường dẫn checkpoint local để không cần tải. Với RTX 3050 6 GB nên bắt đầu từ model n/s; batch luôn được đo trước khi chọn.

Một lần chạy tạo thư mục mới trong `runs/`, gồm:

- `report.json`: phần cứng, tập con, phép đo, lịch sử, Pareto, preset và giới hạn của dự đoán.
- `selected.json`: cấu hình được chọn và nguồn dataset.
- `train.yaml`: cấu hình Ultralytics tương ứng.
- `subset/`: danh sách ảnh train/val cố định, seed và thống kê.
- `trials/`: request, result và log từng tiến trình.

Lệnh tune chỉ tìm kiếm. Huấn luyện toàn bộ bằng lệnh riêng, nằm ngoài ngân sách tìm kiếm:

```powershell
.venv\Scripts\python.exe -m yolo_tuner train --selection runs/<lan_chay>/selected.json
```

Full training lưu checkpoint mỗi epoch. Khi OOM, tiến trình được kết thúc, batch giảm một nửa, `nbs` giữ nguyên và thử resume checkpoint gần nhất; nếu chưa có checkpoint thì khởi động lại. Đây là tiếp tục gần tương đương, không bảo đảm đồng nhất số học hoặc BatchNorm với batch cũ.

## Thuật toán đã triển khai

1. Đọc RAM, CPU, đĩa, GPU, VRAM còn trống và khả năng precision. FP16 là mặc định trên GPU thích hợp; BF16 là tùy chọn, kiểm tra khả năng hỗ trợ. Kiểm tra AMP thực tế của trainer được ghi lại.
2. Kiểm tra dataset, từ chối ảnh trùng đường dẫn giữa train/val. Lấy mẫu ảnh có nhiều nhãn bằng heuristic ưu tiên lớp hiếm, ảnh có vật thể nhỏ và ảnh âm tính. Seed và danh sách ảnh dùng chung cho mọi cấu hình. Chỉ báo small-object ở đây là diện tích tương đối <1%, không phải định nghĩa COCO theo pixel.
3. Dành tối đa 35% ngân sách (trần 5 phút) để profile. Đo batch 1/2, sau đó đề xuất một batch lớn hơn nhưng không vượt 2 lần batch đã đo. Phép thử bao gồm backward và trạng thái optimizer. Đo peak reserved và allocated của PyTorch; dành mặc định 20% VRAM trống làm dự phòng. Đề xuất batch phải được chạy thật. Trong các batch đã đo an toàn, chọn batch nhỏ nhất đạt ít nhất 95% throughput tốt nhất.
4. Đo baseline. Nếu full training được ước tính dưới một giờ thì trả baseline, trừ khi dùng `--force-search`. Baseline này vẫn không có bảo đảm về accuracy cuối.
5. Successive Halving trên các cấu hình đã profile. Tất cả ứng viên cùng rung học cùng số epoch, cùng tập con, seed, optimizer, LR và horizon scheduler. Mỗi rung khởi động lại từ checkpoint đầu để tránh sai khác scheduler/resume; toàn bộ chi phí khởi động lại tính vào ngân sách. Mặc định xếp hạng bằng mAP50 thử nghiệm, giữ khoảng một nửa.
6. Timeout do tiến trình cha quản lý, tính cả khởi động, validation và tải model. Kết thúc worker để giải phóng VRAM. Khi một rung bị gián đoạn, trả cohort của rung hoàn tất gần nhất; nếu rung đầu chưa hoàn tất thì chỉ dùng các ứng viên đã hoàn tất và ghi rõ giới hạn.
7. Thời gian full training = số epoch × (thời gian train subset × tỷ lệ số ảnh train + thời gian val subset × tỷ lệ số ảnh val), cộng thời gian khởi động worker đo được và một lần validation cuối. Chi phí checkpoint full training chưa hiệu chỉnh riêng. Đây là ngoại suy điểm, chưa có bảo đảm sai số. Pareto dùng accuracy proxy/score, thời gian và peak VRAM. Balanced dùng khoảng cách Tchebycheff chuẩn hóa tới điểm lý tưởng, trọng số bằng nhau; không gọi đây là knee point 3D duy nhất.
8. `--tune-lr` thử 0.5×/1×/2× LR khi đủ ngân sách dự kiến. Chỉ áp dụng kết quả khi cả ba phép thử hoàn tất ở cùng fidelity.

`--max-time 7200` giới hạn thời gian full training ước tính. `--min-accuracy 0.65` yêu cầu **final mAP50-95**: khi chưa có predictor offline đủ điều kiện, chương trình báo không chứng nhận được và không tạo `selected.json`, thay vì so ngưỡng này với mAP50 sớm.

## Predictor offline

Thu thập full training trên nhiều dataset, GPU và cấu hình. Dùng cùng `dataset_id` cho mọi cấu hình/seed thuộc một dataset; không đổi ID để tăng giả tạo số nhóm độc lập.

Để đo nhiều cấu hình tự động trên một dataset (tác vụ dài, ví dụ trần 6 giờ), dùng script dưới đây. Chạy lại trên các dataset/GPU khác và cùng file CSV. OOM và timeout được giữ trong `collection.json`; những lần trainer tự đổi batch không đưa vào tập fit. Script dùng grid factorial hữu hạn với thứ tự ngẫu nhiên theo seed, làm baseline trước khi bổ sung Bayesian optimization.

```powershell
.venv\Scripts\python.exe scripts/collect_offline.py --data D:/datasets/my_dataset/data.yaml --dataset-id my_dataset --hours 6 --max-runs 12 --csv measurements.csv
```

Hoặc ghi một lần full training đã thực hiện qua lệnh `train`:

```powershell
.venv\Scripts\python.exe -m yolo_tuner export-offline --report runs/<tim_kiem>/report.json --training-report runs/full/<huan_luyen>/training_report.json --dataset-id my_dataset --csv measurements.csv
.venv\Scripts\python.exe -m yolo_tuner fit-offline --csv measurements.csv --output surrogate.json --alpha 0.1
.venv\Scripts\python.exe -m yolo_tuner tune --data D:/datasets/my_dataset/data.yaml --surrogate surrogate.json --min-accuracy 0.65
```

Schema CSV có ở `examples/offline_columns.csv`. Export chỉ chấp nhận full training hoàn tất và không có restart/đổi batch, để số đo có nghĩa với một cấu hình cố định.

Trường `runtime_protocol` ghi phiên bản PyTorch, Ultralytics, driver và protocol của chương trình. Predictor từ chối phiên bản chưa có trong dữ liệu fit, để không tái dùng khoảng dự đoán sau khi đổi môi trường một cách âm thầm.

Fit dùng ridge regression cho final mAP50-95, peak VRAM và full-training time. Tách theo **dataset_id**, giữ nhóm test độc lập; calibration dùng residual lớn nhất trong mỗi nhóm dataset và quantile có hiệu chỉnh hữu hạn mẫu. Với alpha=0.1 cần ít nhất 9 nhóm calibration, thêm nhóm train và test. Cần nhiều dataset thật để kiểm định chuyển miền; hàng trăm cấu hình trên chỉ một dataset không thay thế được số nhóm độc lập.

Artifact lưu hệ số, phạm vi đặc trưng, nhóm split, MAE và coverage trên test. Khi gặp model/GPU/precision chưa thấy hoặc đặc trưng ngoài miền đã học, predictor từ chối suy luận. Nếu mọi ứng viên thuộc miền, lựa chọn cuối có thể dùng cận dưới final mAP50-95. Ngưỡng accuracy đạt theo predictor chỉ được gọi là **predicted feasible**, không phải full training verified. Các khoảng này là marginal theo nhóm dưới giả thiết trao đổi được; chưa có bảo đảm đồng thời cho cấu hình được chọn sau tìm kiếm thích nghi. Online vẫn đo VRAM/thời gian thật; predictor offline không thay thế kiểm tra bộ nhớ.

## Kiểm tra nhanh

Output CLI được chia thành bốn phần theo thứ tự đọc: **Phần cứng**, **Dataset**, **Kế hoạch tìm kiếm** và
**Kết quả đề xuất**. Mỗi trial hiện trạng thái, model, kích thước ảnh, batch, mAP50 proxy, epoch và thời gian.
Preset được chọn có dấu `●`; cảnh báo phương pháp và đường dẫn artifact nằm cuối. Terminal hỗ trợ ANSI sẽ có màu;
khi redirect vào file hoặc đặt biến môi trường `NO_COLOR`, output tự chuyển sang văn bản thuần.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/create_smoke_dataset.py
.venv\Scripts\python.exe -m yolo_tuner tune --data runs/smoke_data/data.yaml --models yolo11n.yaml --imgsz 128 160 --freeze 0 --precision fp32 --max-batch 2 --budget 6 --full-epochs 3 --rungs 1 2 --force-search
```

Smoke dataset gồm hình chữ nhật tổng hợp, model YAML khởi tạo ngẫu nhiên. Nó kiểm tra luồng GPU thực sự, log, timeout, lựa chọn và training; số mAP của nó không có giá trị chứng minh hiệu quả trên dataset của người dùng.

Xem `METHOD.md` để biết các điểm đã sửa trong đề xuất và kế hoạch đánh giá nghiên cứu. Chưa có dataset của bạn và chưa có tập số đo offline thực, nên không thể kết luận cấu hình nào chính xác nhất cho bài toán của bạn.
