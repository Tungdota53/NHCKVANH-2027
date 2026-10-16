# Phương pháp xác định cấu hình huấn luyện tối ưu

Nâng cấp0.5: thêm adaptive_verified, grid expanded, local verification và báo cáo cross-environment kiểm chứng từ raw measurements. Xem [PORTABLE_VERIFICATION.md](PORTABLE_VERIFICATION.md). Kết quả nghiên cứu0.4 bên dưới vẫn được giữ nguyên; không sửa history để biến thuật toán mới thành người thắng sau khi biết oracle.

## Câu hỏi được giải quyết

Với dataset D, phần cứng H, protocol huấn luyện P, không gian cấu hình hữu hạn C và ngân sách tìm kiếm B đã công bố, phương pháp có chọn được cấu hình gần nghiệm tốt nhất trong C với chi phí thấp không?

`auto` là pipeline vận hành: tìm cấu hình đo được, có thể OOM-recovery/early stopping. **`research` là protocol thực nghiệm riêng**: đóng băng C/P, so sánh các phương pháp, sau đó full-train mọi cấu hình để có đối chứng độc lập. Không gọi kết quả `auto` là tối ưu toàn cục. Không dùng loss sớm thay cho chất lượng sau huấn luyện.

## Định nghĩa tối ưu

Mỗi cấu hình được train trên toàn bộ train/validation theo cùng horizon E và các seed đã khai báo. Early stopping bị tắt trong nghiên cứu; scheduler giữ cùng full horizon, kể cả khi trial proxy chạy ít epoch.

- Captioning: Q = (BLEU-4 + chrF)/200 của checkpoint validation tốt nhất trong E epoch. Checkpoint tốt nhất có thể trước epoch E; thời gian vẫn tính đủ E epoch. Q không phải tỷ lệ caption đúng.
- Classification/reconstruction: Q là validation `quality_score` tại epoch cuối. Không trộn/so trực tiếp giá trị Q giữa các task.
- T: wall time worker, gồm startup, train, validation và checkpoint; không gồm setup dùng chung.
- M: CUDA peak **reserved allocator** PyTorch, CPU/MPS peak RSS process, đơn vị MiB. Không phải tổng bộ nhớ GPU/máy; không dùng M của hai backend như cùng một đại lượng vật lý.

Objective được khóa trước: `quality` tối đa Q trung bình; `time` tối thiểu T trung bình; `memory` tối thiểu M lớn nhất trên seed; hoặc `balanced`:

`U(c) = wQ * mean(Q) - wT * mean(T)/time_scale - wM * max(M)/memory_scale`.

Trọng số được chuẩn hóa tổng bằng 1; mặc định 0.6/0.25/0.15, anchors 600 giây/1024 MiB. Không chuẩn hóa theo min/max của reference sau khi đã biết kết quả. Anchors/trọng số là lựa chọn ưu tiên, không có một “tối ưu” duy nhất khi ưu tiên khác nhau.

Feasible khi đủ mọi seed/full horizon, min(Q) đạt ngưỡng và max(T), max(M) không vượt giới hạn. Đây là điều kiện trên các lần chạy quan sát được, **không phải** khoảng tin cậy cho máy/seed tương lai. Muốn tránh chọn mô hình quá yếu khi chỉ tối ưu tốc độ/bộ nhớ, cần đặt `--min-quality` phù hợp trước khi chạy.

Nghiệm đối chứng `c* = argmax U(c)` trên các cấu hình feasible **đã đo đủ toàn bộ**. Chỉ được ghi `empirical_optimal_in_declared_space`. Đây là tối ưu thực nghiệm trong C/D/H/P/seed, không phải bảo đảm tối ưu với mọi dataset, mọi model hay mọi máy.

## Phương pháp và đối chứng

1. **Default**: cấu hình c00 khóa trước, không gọi evaluator; chi phí tìm kiếm bằng 0, chi phí deployment vẫn có.
2. **Random full-fidelity**: thứ tự cấu hình ngẫu nhiên, mỗi trial dùng full data và E epoch, cùng wall-clock cap B. Timeout không được coi là trial hoàn tất.
3. **Successive Halving**: các ứng viên cùng subset/seed/fidelity; rungs mặc định 1/2/4 epoch, giữ nửa theo Q. Các rung chạy lại từ cùng initialization và cùng scheduler horizon, không resume một scheduler khác. Tính toàn bộ chi phí chạy lại vào B.
4. **Pareto racing**: giữ ứng viên theo Pareto Q/T/M ước tính và U cố định; giữ ít nhất một runner-up khi có, nhằm giảm loại sớm. Nếu còn đủ ngân sách ước tính + margin thì xác nhận full data/E epoch. Nếu chưa xác nhận, kết quả được ghi là proxy, không chứng nhận tính feasible cuối.

Pareto racing là heuristic đang được kiểm định, **không phải Hyperband/BOHB**, không có chứng minh regret bound hay superiority. Giới hạn VRAM đo được có thể loại ứng viên; Q thấp ở epoch đầu không được coi là chứng cứ cấu hình không thể đạt chất lượng cuối.

Đối chứng bổ sung `random_proxy`: thứ tự cấu hình ngẫu nhiên, tất cả trial dùng cùng subset và rung cao nhất (mặc định 4 epoch), cùng cap. Bật bằng `--methods default random random_proxy successive_halving pareto_racing`. Nó giúp phân biệt lợi ích subset với lợi ích lịch racing; không thay thế Random full-fidelity hoặc ablation nhiều ngân sách.

Khi rung bị gián đoạn, dùng cohort cùng fidelity hoàn tất gần nhất. Nếu rung đầu chưa hoàn tất, chỉ dùng những trial hoàn tất và giữ giới hạn này trong history. Quyết định chỉ biết phép đo online; không truy cập kết quả đối chứng full-reference.

Random full-fidelity có thể không hoàn tất được một trial khi B quá nhỏ. Kết quả khi đó trả `selected=null`, regret không xác định, feasibility bằng 0; **không** gán regret 0 hoặc kết luận phương pháp mới thắng phổ quát. Cần thêm random cùng fidelity, ablation và nhiều mức ngân sách trước khi tách được lợi ích của lịch phân bổ tài nguyên khỏi lợi ích dùng subset.

## Protocol kiểm chứng

1. Discover dataset, khóa train/val/test và subset; vocabulary chỉ fit train. `test` không tham gia tìm kiếm/reference chọn cấu hình.
2. Chuẩn bị chung runtime/weights/features một lần và báo chi phí setup riêng. Caption có pretrained encoder nguồn chính thức và checksum; không tải weights tùy ý.
3. Ghi `protocol.json` và SHA256 trước search: configs, seed, split, objective, anchors, ngân sách, hardware/runtime, metric và preprocessing. Các lượt mới cũng ghi checksum/source snapshot. `dataset_sha256` là checksum **manifest có per-image SHA256 do discover tính**, không phải nối bytes tất cả ảnh. Input được kiểm tra lại trước xuất kết luận/resume/audit; phải giữ nguyên dataset nguồn khi tái lập. Kiểm tra ảnh test phục vụ integrity/chống trùng split, không tính metric test để chọn cấu hình.
4. Chạy các phương pháp lần lượt, xoay thứ tự giữa seed để giảm systematic warmup bias. Không chạy CPU/GPU training đồng thời khi đo thời gian. Tác vụ người dùng/thermal/power vẫn có thể ảnh hưởng; không coi số đo laptop là latency bảo đảm.
5. Khóa `decisions.locked.json` trước khi chạy exhaustive reference; hash giúp phát hiện thay đổi artifact, không phải timestamp đăng ký độc lập với bên thứ ba.
6. Full-reference: mọi cấu hình × mọi seed, cùng full data/E epoch, thứ tự ngẫu nhiên. Chi phí này là **offline oracle cost**, không giấu trong ngân sách online.
7. So quyết định đã khóa với reference. Thiếu reference thì không chứng nhận nghiệm hoặc regret/HV cuối; không OOM-retry đổi batch ngầm trong nghiên cứu.
8. `research-audit` đối chiếu request/result gốc, hash protocol/decisions, tính lại proxy errors; không chạy thêm training và không sửa quyết định. Artifact audit riêng, giữ report ban đầu.

`selected.json` của nghiên cứu là **nghiệm exhaustive reference**, không phải nghiệm online có được miễn phí trong B. Muốn dùng lựa chọn của một phương pháp thì đọc `decisions.locked.json`. Khi chuyển dataset/máy/runtime/epoch hoặc OOM đổi batch, pipeline train hạ trạng thái tính tối ưu: cấu hình vẫn có thể dùng nhưng chưa được kiểm chứng lại.

## Số liệu phải báo cáo

- Q/T trung bình, độ lệch chuẩn, worst-seed feasibility, memory max; 3 seed chỉ mô tả biến thiên quan sát, không đủ để suy ra mọi phân phối.
- Regret objective `U(c*) - U(c_chosen)` trên reference đầy đủ; quality regret riêng. Không có lựa chọn/không feasible: null, không phải 0. Báo cả feasibility rate để tránh survivorship bias của mean regret trên những lượt thành công.
- Search wall time thật, budget overrun, search/full time, setup chung và exhaustive-reference cost.
- Pareto và recovered hypervolume của finalist với **full outcomes**, điểm tham chiếu cố định. Chuẩn hóa trục `(Q, max(0,1-T/time_scale), max(0,1-M/memory_scale))`; nếu reference HV bằng 0 thì tỷ lệ không xác định.
- Proxy quality MAE, time MAPE, memory relative error so với **full-reference cùng training seed**, chỉ trên proxy hoàn tất. Đây là sai số ngoại suy, không phải calibrated coverage.
- Cấu hình được chọn từng seed, độ ổn định, số trial/OOM/timeout, số run đối chứng thực sự hoàn tất.

## Chạy thật

```powershell
.venv\Scripts\python.exe train_config.py research --data "ViECAP-Image-Captioning-Vietnamese-master" --caption-encoder resnet18 --offline --full-epochs 8 --seeds 17 29 43 --search-budget 2 --objective balanced --output runs/research_viecap
.venv\Scripts\python.exe train_config.py research-show --run "runs/research_viecap/<run>"
.venv\Scripts\python.exe train_config.py research-audit --run "runs/research_viecap/<run>"
```

Nếu full-reference bị gián đoạn sau khi decisions đã khóa, dùng `research-resume --run <run> --timeout-hours 3`. Chỉ chạy lại các reference seed thiếu/thất bại từ initialization gốc (không nối checkpoint thiếu thời gian đo); giữ attempts cũ và tính chi phí thất bại đã ghi nhận. Phải cùng source/runtime/hardware/manifest, không có process đang sống. Search chưa khóa thì cần study mới: không tái chạy như thể ngân sách chưa bị gián đoạn. Downtime hoặc wall time chưa kịp ghi lúc process chết không được gán thành thời gian active; chi phí trường hợp đó là phần đã ghi, có thể thiếu chi phí trial bị mất.

Tạo dataset đối chứng phân loại CIFAR-10, giữ nhãn gốc và split chính thức (train-derived validation, test từ official test):

```powershell
.venv-cpu\Scripts\python.exe scripts/prepare_cifar_research.py
.venv-cpu\Scripts\python.exe train_config.py research --data runs/datasets/cifar10_research --device cpu --full-epochs 8 --seeds 17 29 43 --search-budget 2 --methods default random random_proxy successive_halving pareto_racing --output runs/research_cifar_cpu
```

Đổi objective/constraints bằng `--objective`, `--min-quality`, `--max-time`, `--max-memory`, `--weights`, `--time-scale`, `--memory-scale`; phải tạo study mới, không sửa protocol sau khi đã biết reference. Đổi grid bằng JSON list hyperparameter overrides `--grid`, chỉ trong cùng task/model/encoder đã hỗ trợ. Ví dụ `[{}, {"hidden":384}, {"lr0":0.0003}, {"hidden":128,"batch":16}]` cho captioning.

Runner nghiên cứu hiện hỗ trợ captioning/classification/reconstruction compact, CPU/CUDA/MPS khi runtime tương thích. Nhánh YOLO detection vẫn dùng `tune`/predictor offline cũ; **chưa có** exhaustive research adapter mới cho mọi detection model. Không giả định đọc được dataset bất kỳ chưa có adapter hoặc chạy được trên máy thiếu tài nguyên/phần mềm.

## Cơ sở và giới hạn nghiên cứu

Random Search là đối chứng dựa trên [Bergstra & Bengio, JMLR 2012](https://www.jmlr.org/papers/v13/bergstra12a.html). Lịch phân bổ tài nguyên liên quan Successive Halving/Hyperband trong [Li và cộng sự, JMLR 2018](https://www.jmlr.org/papers/v18/16-558.html); mã hiện tại không triển khai toàn bộ Hyperband. Chế độ deterministic dựa trên [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), không bảo đảm kết quả đồng nhất giữa platform/phiên bản khác nhau. Dữ liệu đối chứng phân loại từ [CIFAR-10 chính thức](https://www.cs.toronto.edu/~kriz/cifar.html).

Trước khi kết luận phương pháp có giá trị tổng quát: cần grid rộng hơn, nhiều ngân sách, random cùng fidelity, ablation runner-up/full confirmation, nhiều dataset thật và GPU/máy độc lập, kiểm định ngoài miền và test mới chưa từng dùng trong engineering. Hai backend CPU/CUDA trên cùng laptop **không** thay thế nhiều máy/GPU. Test ViECAP đã được xem trong các lần nâng cấp trước nên không công bố là untouched holdout cho nghiên cứu này.
