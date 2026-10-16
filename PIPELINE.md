# Pipeline tự động chọn cấu hình trên máy và dataset hiện tại

## Nhánh nghiên cứu 0.4

Nếu mục tiêu là **nghiên cứu phương pháp xác định cấu hình tối ưu**, dùng `research`, không coi kết quả `auto` là chứng minh tối ưu. `research` khóa objective/không gian/seed/protocol, so sánh baseline trong cùng budget cap, rồi đo exhaustive full-reference độc lập để tính regret/feasibility/Pareto/hypervolume và sai số proxy. Xem [RESEARCH_METHOD.md](RESEARCH_METHOD.md) và [RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md).

`research-show --run <run>` đọc tiến độ và kiểm tra worker còn sống; `research-audit --run <run>` đối chiếu artifact đã hoàn tất; `research-resume --run <run>` chỉ tiếp tục full-reference còn thiếu sau khi decisions đã khóa, cùng protocol/runtime. Không gọi cấu hình tối ưu nếu thiếu đối chứng hoặc tự đổi epoch/dataset/máy sau đo. `auto` bên dưới vẫn là chế độ vận hành nhanh.

## Một lệnh chạy

```shell
python train_config.py --data "/path/to/dataset"
```

Ở Windows tại workspace này:

```powershell
.venv\Scripts\python.exe train_config.py --data "ViECAP-Image-Captioning-Vietnamese-master"
```

Dataset có thể là thư mục dự án, thư mục dữ liệu hoặc file nhãn. Pipeline tự nhận diện, chuẩn hóa sang manifest riêng, đo thử các cấu hình trong ngân sách và ghi cấu hình đã chọn. Không cần tự tạo YAML cho caption, ImageFolder, COCO, VOC hoặc LabelMe rectangles.

Lệnh tương đương sau khi cài package là `python -m yolo_tuner auto --data ...` hoặc `train-config auto --data ...`. Lệnh `tune --data <thư mục/JSON/CSV>` cũng chuyển qua pipeline tự động; nhánh YOLO YAML cũ vẫn chạy.

Giao diện CLI hiện trực tiếp trong terminal: bảng phần cứng và dataset, kế hoạch tìm kiếm, vòng epoch và tiến độ `[i/n]`, kết quả từng trial, bảng preset và cấu hình đã chọn. Caption v2 hiển thị BLEU-4/chrF của câu sinh trên validation và loss; không hiển thị mAP của detection. Màu bật khi terminal hỗ trợ TTY; `NO_COLOR=1` tắt màu, bảng và tiến độ vẫn hiện khi chuyển hướng output. Đây là giao diện terminal, không phải cửa sổ GUI.

Thêm `--train` để chạy trọn luồng search → full training → đánh giá test caption nếu có test. Full training nằm ngoài ngân sách search; giới hạn riêng bằng `--timeout-hours`. Không kết hợp `--train` với `--plan-only`.

Xem lại kết quả đã lưu bằng giao diện CLI mà không huấn luyện lại:

```powershell
.venv\Scripts\python.exe train_config.py show --report "runs\auto\RUN_ID"
```

`--report` cũng nhận đường dẫn tới `report.json`. Phần cứng trong màn hình xem lại là thông tin đã lưu tại thời điểm đo, không phải phép dò lại máy hiện tại.

## Cài trên máy khác

Yêu cầu Python >=3.10, môi trường PC/laptop có thể cài PyTorch; Windows/Linux/macOS. Cài bản CPU là đủ để chạy thử và chọn cấu hình:

```shell
python -m venv .venv
# Windows: .venv\Scripts\python.exe; Linux/macOS: .venv/bin/python
python -m pip install -e .
```

Trên Linux/Windows không muốn cài runtime GPU, cài PyTorch CPU theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/) trước khi cài project. Nếu dùng GPU NVIDIA, chọn bản CUDA tương thích từ cùng hướng dẫn; không bắt buộc có NVIDIA. Mac dùng PyTorch có MPS. Pipeline thử một tensor thật để chọn CUDA/MPS, rồi dùng CPU khi backend tự chọn không chạy được. GPU AMD với PyTorch ROCm cũng dùng API `torch.cuda`, nhưng chưa được kiểm thử trực tiếp trong tác vụ này.

Khi chưa có thư viện ML, phần lập kế hoạch cho các định dạng ảnh vẫn có thể ghi cấu hình nếu các thư viện đọc ảnh/YAML và psutil đã có; trạng thái là `recommended_unverified`. Việc huấn luyện cần cài runtime. Không tự thay driver hay cài CUDA lên hệ điều hành.

## Nhận diện và xử lý dataset

| Input | Task/model thực thi | Metric thử nghiệm |
|---|---|---|
| YOLO YAML hoặc ảnh + labels 5 cột | Ultralytics detection | Early validation mAP50 |
| COCO JSON có bbox/category_id | Chuyển layout riêng rồi detection | Early validation mAP50 |
| VOC XML / LabelMe rectangle JSON | Chuyển bounding box sang YOLO | Early validation mAP50 |
| COCO-style caption JSON, caption CSV | Encoder ảnh + GRU ngữ cảnh ảnh + cross-attention | BLEU-4/chrF của caption sinh trên validation |
| Thư mục lớp `train/class_a`, `val/class_a` hoặc `class_a`, `class_b` | CNN classifier | Validation accuracy |
| Chỉ có ảnh, không có nhãn | CNN autoencoder, reconstruction pretraining | Pixel MSE trên validation |

Caption CSV chấp nhận cột ảnh `image`, `image_path`, `file_name` hoặc `filename`; cột caption `caption` hoặc `text`. Nếu cạnh CSV đã có JSON cùng tên thì dùng JSON để tránh đọc dữ liệu hai lần.

Thư mục có nhiều nguồn caption được gộp sau khi kiểm tra. SHA256 ảnh loại bản sao chính xác để tránh leakage; caption trùng trong cùng ảnh được gộp. Tập test gốc được giữ riêng, không dùng để chọn cấu hình hay xây vocabulary. Nguồn train không có val được lấy một holdout có seed từ chính train của nguồn đó. Near-duplicates/video groups vẫn cần metadata hoặc kiểm tra riêng; hash không giải quyết được các ảnh gần giống.

Nếu một input chứa nhiều task thật, dùng `--task detection` hoặc `--task captioning`, hoặc trỏ vào thư mục cụ thể. JSON dự đoán `_pred`/`_gt` trong repo không được coi là dataset training. Nhãn sai tọa độ, ảnh thiếu hoặc mơ hồ, class map không thống nhất sẽ báo lỗi rõ. Các task segmentation/pose/OBB, audio/text/tabular và định dạng riêng chưa có adapter không được tự nhận là detection.

Ảnh không nhãn vẫn nhận được cấu hình có thể train reconstruction; pipeline không biến caption hoặc prediction YOLO thành bounding box ground truth. Cấu hình sinh caption dùng model đi kèm, không phải chỉnh ngầm checkpoint ClipCap/GPT của ViECAP gốc.

## Luồng tìm cấu hình

1. Đọc dataset, kiểm tra ảnh và nhãn, tạo split/dedup cần thiết, giữ source nguyên trạng.
2. Chọn backend thực sự hoạt động: CUDA/ROCm API, MPS, CPU. Đọc VRAM nếu có, RAM và số CPU. Precision GPU mặc định FP16; CPU/MPS mặc định FP32. workers=0; cache ảnh trong RAM tắt ở cả search và full training để không ngoại suy từ một subset đã cache toàn bộ sang dataset lớn.
3. Tạo các ứng viên và giới hạn theo thiết bị. Detection giữ Ultralytics với model local nếu có hoặc YAML học từ đầu. Caption mặc định thử encoder ResNet18 pretrained từ Torchvision (khoảng 45 MB, kiểm tra SHA256), frozen, crop/normalize đúng ImageNet, GRU đọc ảnh mỗi bước và spatial cross-attention; tìm hidden size, dropout và LR. Dùng `--offline` để cấm tải weights, `--caption-encoder cnn` để không cần pretrained, hoặc `--caption-encoder resnet18` để báo lỗi thay vì âm thầm fallback khi encoder không sẵn sàng. Auto fallback sang attention CNN học từ đầu và ghi rõ pretrained=False. Classification/reconstruction vẫn dùng model đi kèm.
4. Lấy tập mẫu cố định, cân bằng giữa nguồn/lớp. Vocabulary caption và giới hạn token fit trên train đầy đủ, dùng chung cho mọi trial; không đọc caption val/test để fit policy.
5. Mỗi worker là một subprocess. Tất cả cấu hình cùng rung có số epoch bằng nhau, cùng tập mẫu/seed/protocol. Đo forward/backward, validation, trạng thái optimizer, throughput, peak allocator GPU và RSS RAM. Hết thời gian thì dừng process; OOM giảm batch và chạy lại có giới hạn. Không loại model theo accuracy bịa trước khi đo.
6. Successive Halving giữ khoảng một nửa theo metric task. Caption v2 dùng `(BLEU4 + chrF) / 200`, đo bằng greedy generation không đưa câu đúng vào decoder. Đối chứng là caption phổ biến nhất từ train đầy đủ. Không cho trial lặp một câu hoặc không vượt đối chứng trên cả hai metric vào preset đạt chất lượng. Loss teacher-forcing chỉ là số đo phụ. Reconstruction dùng `1/(1+MSE)`; classification dùng accuracy thật. Báo cáo/checkpoint cũ giữ nguyên metric cũ, không đổi tên CE thành BLEU.
7. Giữ cohort ở rung hoàn tất gần nhất nếu hết giờ. Tạo Pareto về quality proxy/time/memory, Balanced bằng Tchebycheff chuẩn hóa. Thử tăng batch 2×/4× cho model thắng khi còn thời gian; batch chỉ được thay khi throughput tăng và chất lượng đo không giảm đáng kể. Ngoại suy thời gian tách compute cost và chi phí đọc ảnh, rồi cân chi phí đọc theo số example thật của từng nguồn để không đánh đồng ảnh lớn/chậm với ảnh nhỏ/nhanh.
8. Xuất cấu hình cùng chứng cứ. Không đủ thời gian/RAM/validation thì ghi `recommended_unverified`. Đã đo nhưng caption không qua đối chứng thì ghi `measured_quality_not_established`, lưu cấu hình để tiếp tục học, không gắn nhãn đạt chất lượng. Full training v2 chọn best.pt theo BLEU4/chrF validation, giảm LR khi plateau và early stop sau 5 epoch không cải thiện; test không dùng để chọn epoch hoặc cấu hình.

Gradient accumulation cộng loss dạng sum rồi chia theo tổng token/example/pixel thực, kể cả nhóm cuối. Encoder CNN dùng GroupNorm cho batch=1; ResNet frozen giữ BatchNorm ở eval. Caption v2 dùng label smoothing/dropout/AdamW. Checkpoint lưu cả encoder (inference không cần tải lại), optimizer, scaler, scheduler, RNG, vocabulary, config và fingerprint train/val để không resume subset như đã học full dataset.

ResNet frozen được trích đặc trưng một lần cho train/val, lưu float16 trên đĩa; DataLoader đọc qua mmap, không giả định toàn bộ ảnh nằm trong RAM. Cache xác định bởi hash ảnh, weights, crop và grid; không chứa ảnh test. Search/full dùng cùng policy cache và tính cả generation vào thời gian validation. Inference độc lập giữ cùng precision đặc trưng. Chuẩn bị encoder/cache nằm trong deadline search, có log và giảm batch nếu OOM.

Ngân sách mặc định 20 phút. Deadline worker tính cả khởi động và validation; thời gian chuẩn hóa/indexing/export file cũng được ghi trong tổng elapsed. Các thao tác đĩa/indexing không thể ngắt giữa một lần đọc file nên có thể vượt ngân sách rất nhỏ; báo cáo hiển thị elapsed thực, không che phần vượt. Không tuyên bố tối ưu toàn cục hay final accuracy chỉ từ thử nghiệm ngắn.

## Các lệnh thường dùng

```shell
# Tự chọn phần cứng và cấu hình
python train_config.py --data DATASET --budget 20

# Chạy trọn luồng thật; test caption tự đánh giá nếu dataset có test
python train_config.py --data DATASET --train

# Không tải weights; auto dùng cached ResNet hoặc fallback attention CNN
python train_config.py --data DATASET --offline

# Máy không GPU, chạy nhanh với ít ứng viên
python train_config.py --data DATASET --device cpu --budget 3 --max-candidates 3 --train-subset 32 --val-subset 12 --rungs 1 2

# Chỉ tạo cấu hình ban đầu, không huấn luyện thử
python train_config.py --data DATASET --plan-only

# Dataset detection, chọn checkpoint cụ thể
python train_config.py --data DATASET --task detection --models weights/yolo26n.pt

# Huấn luyện thật cấu hình đã chọn, tự dò lại phần cứng
python -m yolo_tuner train --selection runs/auto/RUN_ID/selected.json

# Caption: train thật, sau đó tự sinh caption và đánh giá test độc lập
# SacreBLEU đã là dependency từ phiên bản 0.3
python -m yolo_tuner train --selection SELECTED_JSON --evaluate-test

# Xem tiến độ/checkpoint/báo cáo đánh giá của một full-training run
python train_config.py show-training --run runs/full/FULL_RUN_ID

# Sinh caption cho ảnh thật, không dùng caption tham chiếu
python train_config.py predict --checkpoint BEST_PT --image IMAGE_PATH

# Đánh giá riêng tất cả ảnh test từ manifest đã chuẩn hóa
python train_config.py evaluate-caption --checkpoint BEST_PT --manifest DATASET_JSON --split test

# Tiếp tục checkpoint compact model
python -m yolo_tuner train --selection SELECTED_JSON --resume LAST_PT

# Chủ động kéo dài giới hạn epoch khi resume
python -m yolo_tuner train --selection SELECTED_JSON --resume LAST_PT --epochs 30

# Khi chuyển dữ liệu sang máy khác
python -m yolo_tuner train --selection SELECTED_JSON --data NEW_DATASET_PATH --device auto
```

Khi chuyển sang máy yếu hơn, nên chạy lại `auto` để đo batch/time phù hợp. Full training tự giảm batch khi OOM; CPU/MPS tự đổi precision sang FP32. Override dataset khi chuyển máy phải giữ class definitions/model/vocabulary phù hợp; không dùng config của một task cho task khác.

## Output

Mỗi lần chạy có thư mục timestamp riêng:

- `dataset.json`: manifest train/val/test đã xử lý, hash, nguồn và cảnh báo.
- `subset.json`: đúng danh sách mẫu dùng thử (nhánh compact).
- `report.json`: phần cứng, lịch sử, metrics, preset, Pareto, thời gian thực và status.
- `selected.json`: lựa chọn dùng cho lệnh train, task/backend và vocabulary caption.
- `train.yaml`: biểu diễn YAML của cấu hình lựa chọn; lệnh train hiện đọc selected.json.
- `REPORT.md`: cấu hình và lệnh chạy dễ đọc.
- `trials/`: request, result và worker.log. Full training thêm `last.pt`, `best.pt`.

`measured_best_observed` = cấu hình được chọn từ phép đo thật trong không gian đã thử. Caption v2 phải qua đối chứng trên validation mới dùng trạng thái này. `recommended_unverified` = chưa có trial hoàn tất. `measured_quality_not_established` = đã đo nhưng chưa qua đối chứng. `final_accuracy_verified` không tự thành true sau vài epoch hoặc sau BLEU/chrF; chưa đánh giá độ đúng ngữ nghĩa bằng con người.

## Chạy được không đồng nghĩa chất lượng tốt

Lệnh `train` đọc toàn bộ manifest train/val, không dùng subset search. Mỗi epoch lưu last.pt; caption v2 chọn best.pt theo caption validation và có early stopping. Model cũ giữ tiêu chí checkpoint cũ để tương thích. CLI hiện epoch/loss/metric khi checkpoint lưu; show-training kiểm tra PID và thời điểm tạo tiến trình ở run mới, phân biệt đã dừng và đang chạy. Timeout/Ctrl+C dừng cả cây worker, kể cả launcher Python trên Windows.

Lệnh `predict` dùng ảnh + checkpoint để sinh câu tự hồi quy, không đưa câu đúng vào decoder. Lệnh `evaluate-caption` dùng tất cả ảnh của split được yêu cầu, kiểm tra overlap với train/val và ghi `predictions.json`, `evaluation.json`, `EVALUATION.md`. Không tự thay test thiếu bằng train hoặc validation.

Metric được tính bằng [SacreBLEU](https://github.com/mjpost/sacrebleu): BLEU-4 và chrF multi-reference, thang 0–100; không phải phần trăm ảnh mô tả đúng. Tokenizer là regex tiếng Việt/lowercase của pipeline, không phải PTB tiếng Anh của COCO; không so trực tiếp với BLEU/CIDEr trong paper khác tokenizer/protocol. Chưa triển khai CIDEr hoặc đánh giá con người.

Đối chứng là một câu phổ biến nhất từ **train**, lặp cho tất cả ảnh đánh giá; không fit đối chứng trên val/test. Báo cáo cũng có tỷ lệ câu khác nhau và phép xoay caption để phá khớp ảnh. Nếu chỉ sinh một câu hoặc không vượt đối chứng trên cả BLEU-4/chrF, trạng thái là `quality_not_established`, lệnh đánh giá trả exit code 3. Vượt đối chứng cũng chỉ là kiểm tra tối thiểu, không tự chứng nhận model sẵn sàng dùng trong sản phẩm hay cấu hình tối ưu toàn cục. Mã checkpoint bất kỳ không cho biết nó từng học dữ liệu nào; báo cáo không tự xác nhận provenance chỉ từ weights.

Nguồn API: [PyTorch installation](https://pytorch.org/get-started/locally/), [Ultralytics detection formats](https://docs.ultralytics.com/datasets/detect/), [COCO conversion](https://docs.ultralytics.com/guides/coco-to-yolo/). Model compact trong repo là baseline mới, không sử dụng các metric caption sẵn có trong ViECAP để tuyên bố chất lượng.

Nguồn thiết kế v2: [Torchvision ResNet18 weights và preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [Show, Attend and Tell](https://proceedings.mlr.press/v37/xuc15.html). Implementation GRU + cross-attention ở đây không phải tái lập nguyên paper hoặc model ViECAP gốc.
