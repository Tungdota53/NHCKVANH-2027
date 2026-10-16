# Xác minh pipeline 0.3 — 2026-10-12

Nhánh nghiên cứu bản0.4 đã có 24 full-reference runs và 78/78 tests ở hai runtime: xem [RESEARCH_VALIDATION.md](RESEARCH_VALIDATION.md). Phần dưới là kiểm chứng model/pipeline vận hành bản0.3 trước đó, không dùng thay đối chứng nghiên cứu mới.

## Caption v2: đã huấn luyện và đánh giá dữ liệu thật

Model `attn_gru_captioner`: encoder ResNet18 ImageNet frozen, spatial cross-attention và GRU nhận context ảnh ở mọi token. Weights official đã kiểm tra SHA256; train/val features float16 được cache trên đĩa, không đưa test vào cache. Tất cả số dưới đây đo bằng cùng tokenizer/protocol, không phải % caption đúng.

| Đánh giá trên 331 ảnh test | CNN+GRU cũ | Caption v2 | Câu cố định từ train |
|---|---:|---:|---:|
| BLEU-4 | 23,2035 | 33,9476 | 28,0454 |
| chrF | 41,1446 | 46,0391 | 43,2004 |
| Caption khác nhau | 12 | 42 | 1 |

Phá alignment bằng cách xoay caption làm BLEU-4 v2 giảm xuống 22,3871 và chrF 36,1430. Đây là bằng chứng chẩn đoán model có dùng thông tin ảnh, không phải chứng nhận độ đúng ngữ nghĩa. Theo nguồn: Flickr test BLEU-4 21,4700; UIT-ViIC 38,9699.

Search thật với 512 ảnh train / 128 ảnh val, 3 ứng viên, rung 2/4 epoch. Cấu hình chọn: imgsz 224, batch 32, hidden 256, effective batch 64, FP16, LR 0.001, dropout 0.3, label smoothing 0.05. Caption validation của trial thắng: BLEU-4 42,1574 / chrF 53,2846 (xem report.json cho giá trị đầy đủ). Chỉ trial qua đối chứng mới vào preset đạt chất lượng.

Full training đọc đủ **13.858 caption train / 4.714 caption val**. Giới hạn 20 epoch; early stopping hợp lệ ở epoch 6 sau 5 epoch không tăng score generation, không phải timeout/lỗi. Best.pt chọn ở epoch 1 theo **caption validation**, không theo test. Val loss tiếp tục giảm không được dùng để ghi đè checkpoint có câu sinh tốt hơn. Worker full training 71,2 giây, chưa tính chuẩn bị cache; không so trực tiếp thời gian 6 epoch cached này với lượt cũ 20 epoch đọc ảnh.

Artifacts mới, lượt cũ được giữ nguyên:

- Search: `runs/viecap_v2/20261012_194206_062909/report.json` và `selected.json`.
- Full run: `runs/viecap_v2/20261012_194206_062909/full/20261012_194351_301967/training_report.json`.
- Checkpoint dùng được: `runs/viecap_v2/20261012_194206_062909/full/20261012_194351_301967/trials/trial_0001/best.pt`.
- Test sau kiểm tra precision cache/inference: `runs/viecap_v2/20261012_194206_062909/full/20261012_194351_301967/evaluation/20261012_195736_954586/EVALUATION.md`, `evaluation.json`, `predictions.json`.

Xem kết quả mới:

```powershell
.venv\Scripts\python.exe train_config.py show-training --run "runs\viecap_v2\20261012_194206_062909\full\20261012_194351_301967"
```

## Các kiểm tra v2 khác

- CPU-only + offline trên dữ liệu Flickr thật: search, full training 2 epoch và test 100 ảnh hoàn tất. Dataset nhỏ/2 epoch chỉ sinh một câu và không qua đối chứng; CLI báo chưa đạt và trả exit code 3, không ghi thành chất lượng tốt. Đây là kiểm tra thực thi CPU, không phải bằng chứng mọi dataset sẽ có chất lượng tốt.
- CPU resume: tiếp tục checkpoint epoch 2 → 3, optimizer/scaler/scheduler/RNG được tải, kiểm tra fingerprint dữ liệu/vocabulary; loss val 4,3588. Run: `runs/v2_cpu_resume/20261012_195509_177329/training_report.json`.
- GPU batch 64 đo lại trên cùng subset/rung 4: hoàn tất, peak allocator 162 MiB, BLEU-4 42,04 / chrF 52,22; giữ batch chỉ khi đủ cả throughput và quality. Không thay checkpoint full/test theo phép đo này.
- **61/61 tests đạt trong cả runtime CPU/CUDA**; kiểm tra gradient vào encoder/attention, frozen BatchNorm, generation/BOS/EOS, mmap/identity, fingerprint resume không phụ thuộc đường dẫn, quality gate, early stop vs timeout, cleanup toàn bộ cây worker Windows và tương thích checkpoint cũ.
- Package 0.3.0 đã cài editable vào hai venv, SacreBLEU là dependency chính; compileall và kiểm tra dependency tương thích đạt.

Kết luận: bản nâng cấp đã chạy thật và vượt baseline trên ViECAP này. Model vẫn có nhiều caption chung chung; chưa có CIDEr hoặc đánh giá con người, chưa bảo đảm đạt mục tiêu ứng dụng, mọi dataset/máy hoặc optimal toàn cục.

## Lịch sử: pipeline 0.2

Đã chạy thực tế cả môi trường CUDA và môi trường chỉ có CPU. Cấu hình người dùng đã có, không còn dừng ở bước kiểm tra ảnh/inference.

## ViECAP: cấu hình đã chọn

Input: `ViECAP-Image-Captioning-Vietnamese-master`. Tự nhận diện captioning, gộp UIT-ViIC và Flickr, loại 2 bản sao ảnh train, gộp caption lặp, giữ test gốc. Flickr không có validation riêng nên lấy 20 ảnh từ train làm holdout; không đưa test vào search.

- Train: 2.773 ảnh / 13.858 caption.
- Validation: 944 ảnh / 4.714 caption.
- Test giữ riêng: 331 ảnh / 1.654 caption sau khi gộp các caption nguyên văn trùng.

Tìm kiếm cuối: 4 kiến trúc/LR/image-size ứng viên ở rung đầu, 2 finalist ở rung thứ hai, thêm hai phép thử batch. Tổng 8 worker, 70,641 giây trong ngân sách 180 giây. Model được chọn từ phép đo thực tế:

| Tham số | Giá trị |
|---|---|
| Task | Captioning |
| Model | CNN + GRU đi kèm, học từ đầu |
| Encoder width / GRU hidden | 32 / 256 |
| Image size | 160 |
| Batch | 4 |
| Effective batch | 32, accumulation theo token thực |
| Precision | FP16 trên GPU này |
| Optimizer / LR | AdamW / 0.001 |
| Workers / Image cache | 0 / tắt |
| Epoch đề xuất | 20 |
| Token validation loss sau 2 epoch thử | 4,35214 |
| Peak reserved allocator CUDA trong thử | 76 MiB, không gồm CUDA context/driver |
| Ước tính 20 epoch sau khi sửa I/O weighting | khoảng 2.800 giây (46,7 phút); chưa phải cam kết thời gian |

Lượt full-training xác minh riêng đã hoàn tất 1 epoch trên toàn bộ train/val đã chuẩn hóa: 13.858 example train và 4.714 example val; token loss 1,44657, wall time khoảng 139,8 giây, lưu last.pt/best.pt. Đây là kiểm tra khả năng train thực, không phải báo cáo chất lượng caption sinh tự do. Chưa huấn luyện đủ 20 epoch hoặc tính BLEU/CIDEr trên test. Pipeline dùng model compact mới, không chạy lại checkpoint ViECAP ClipCap/GPT gốc.

Phép thử đầy đủ cho thấy cache trên subset làm ngoại suy quá lạc quan. Mã cuối đã tắt image cache cả search/full, tách compute cost khỏi image decoding và cân chi phí đọc ảnh theo tỷ lệ nguồn trong dataset đầy đủ. Phép đo mới gần chi phí epoch đầy đủ đã quan sát, nhưng chưa hiệu chỉnh khoảng dự đoán.

Artifacts cuối:

- `runs/viecap_pipeline_final/20261012_171552_978704/REPORT.md`
- `runs/viecap_pipeline_final/20261012_171552_978704/selected.json`
- `runs/viecap_pipeline_final/20261012_171552_978704/report.json`
- Full-training check: `runs/viecap_full_validation/20261012_171030_554341/training/20261012_171034_394646/training_report.json`

## Các nhánh khác đã chạy

| Kiểm tra thực | Kết quả |
|---|---|
| Captioning trên CUDA, ViECAP toàn bộ nguồn | Search và full epoch thành công |
| Captioning CPU chọn thủ công | Search hoàn tất; full training Flickr 3 epoch, loss 3,79393 |
| Captioning auto trong môi trường PyTorch chỉ CPU | Tự chọn cpu, search 19,375 giây, lưu cấu hình batch 8 / FP32 |
| Resume caption checkpoint CPU | Tiếp tục epoch 3 → 4, loss 3,60974; optimizer/scaler/model được tải lại |
| YOLO detection tự nhận dạng YAML | GPU search thành công trên fixture có box; selected.json và YAML tạo được |
| ImageFolder classification | GPU search hoàn tất và tạo cấu hình |
| Ảnh không nhãn | CPU reconstruction search hoàn tất và tạo cấu hình |
| Plan-only | Tạo cấu hình `recommended_unverified`, không gọi worker |
| Lệnh positional và console entry point | `train_config.py DATASET` và `train-config --data DATASET` hoạt động |

Unit/adapter/CLI/model tests: **32/32 đạt trong cả môi trường CUDA và môi trường PyTorch chỉ CPU**. Có kiểm tra vocabulary chỉ fit train, chống trùng nội dung giữa split, loại ảnh hỏng mà không xóa nguồn, chuyển COCO/LabelMe/YOLO, chia ImageFolder, task model có gradient ở encoder với batch=1, Pareto, finite-sample conformal, deadline và fallback GPU driver không chạy được.

Môi trường GPU: Python 3.12.14, torch 2.6.0+cu124, torchvision 0.21.0+cu124, Ultralytics 8.4.175. Môi trường CPU: torch 2.14.1+cpu, torchvision 0.29.1+cpu. `uv pip check` đều báo 42 package tương thích; compileall đạt. Backend MPS được kiểm tra qua mô phỏng chọn thiết bị, chưa chạy trên Mac thật. ROCm và việc huấn luyện trên mọi cấu hình hệ điều hành/phần cứng chưa được thực nghiệm trực tiếp.

Cấu hình là lựa chọn tốt nhất được quan sát trong ngân sách và không gian ứng viên đã thử. Không có đảm bảo rằng mọi dataset bất kỳ hoặc mọi máy vật lý đều train được, hay đây là global optimum. Phạm vi adapter và cách thêm dữ liệu/labels được công bố ở PIPELINE.md; input sai hoặc task chưa có adapter không được chuyển thành nhãn giả.
