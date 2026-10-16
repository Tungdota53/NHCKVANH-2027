# CLI 0.10.0 — thuật toán tối ưu cấu hình và đối chứng

Mặc định mới là `adaptive_full`: chọn phép thử tiếp theo từ full confirmations đã đo, không prune theo quality sớm. Xem [phương pháp](ALGORITHM_010_METHOD.md) trước khi diễn giải “tối ưu”. `reserve` và các phương pháp cũ vẫn chọn được; không thay chứng nhận/source của các lượt lịch sử.

```powershell
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\images" --method adaptive_full
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\images" --method adaptive_no_cost
.venv\Scripts\python.exe -m research_pipeline.v2 algorithm-report --manifest studies.json --output runs/analysis_new
```

Ba ablation: `adaptive_no_cost` bỏ cost denominator của acquisition, không bỏ time term trong objective; `adaptive_no_uncertainty` bỏ optimism bonus; `adaptive_no_diversity` bỏ explicit novelty bonus, không bỏ feature geometry. Chúng dùng cùng bootstrap, constraints và full-confirm policy để so sánh có kiểm soát.

Acquisition được khóa qua `--acquisition-ridge` (0.05), `--acquisition-length-scale` (1), `--acquisition-exploration` (0.25), `--acquisition-novelty` (0.05). Đây là design choices chưa chứng minh phổ quát; không tune bằng test/reference. Adaptive hiện chỉ hỗ trợ `--selection-rule weighted`; dùng reserve/random/SH cho các rules khác.

`algorithm-report` đọc manifest như dưới, yêu cầu study có reference full đã audit và budget online nhỏ hơn measured exhaustive reference cost. Nó không huấn luyện, không sửa study, chỉ ghi thư mục output **mới**. Không-selection giữ riêng, CI chỉ xuất khi đủ5 paired usable repeats trong cùng study; không gọi nhiều training seeds là nhiều dataset/GPU.

```json
{
  "schema_version": 1,
  "studies": [{"path": "runs/benchmark/<study>"}],
  "required_methods": ["adaptive_full", "random", "reserve", "default", "successive_halving"],
  "bootstrap_seed": 2027,
  "bootstrap_samples": 2000,
  "confidence": 0.95
}
```

Mỗi search/ref vẫn có chi phí thật. Khuyên xem preparation/startup/full cost local trước khi chọn budget cho cả nghiên cứu; không giả định60s đủ chỉ vì dataset nhỏ. Tham số mặc định vận hành1200s không bảo đảm đủ mọi grid/horizon/seeds.

Các đường dẫn study/reference tương đối trong manifest được tính từ **thư mục chứa manifest**, không phải cwd. `required_methods` áp dụng cho mọi study: khi ghép study8methods với study chỉ adaptive/random, đặt hai methods chung hoặc bỏ yêu cầu này; các baseline thiếu vẫn được ghi rõ. [ALGORITHM_010_METHOD.md](ALGORITHM_010_METHOD.md) là mô tả thuật toán hiện tại; các tài liệu0.7/0.9 phía sau là lịch sử/nền thực thi.

## Mở giao diện

Tại PowerShell trong thư mục dự án:

```powershell
.venv\Scripts\python.exe train_config.py
```

Hoặc dùng launcher tự tìm virtual environment của dự án:

```powershell
.\cli.ps1
```

Chưa có môi trường: xem `setup-v2.ps1` và `PIPELINE_V2.md`. Launcher không tự cài package, sửa driver hay mở thêm cửa sổ. Không cần thêm thư viện giao diện. Console dùng UTF-8; các thành phần hiển thị có fallback ASCII và không phụ thuộc màu/ANSI.

Menu chỉ mở khi cả đầu vào và đầu ra là terminal tương tác. Nếu chạy không tham số qua pipe/redirect hoặc từ job tự động, chương trình in help và thoát; không chờ nhập hoặc tự train. Dùng lệnh đầy đủ khi chạy script. `menu`/`wizard` sẽ báo lỗi rõ ràng nếu không có terminal.

## Tìm cấu hình mới

Chọn **1** trong menu, hoặc:

```powershell
.venv\Scripts\python.exe train_config.py wizard
```

1. Dán đường dẫn **dataset ảnh**: thư mục, YAML, JSON hoặc CSV caption tham chiếu ảnh. Có thể có khoảng trắng và dấu nháy bao quanh. Chọn `auto`, `cpu` hoặc chỉ số GPU. Chương trình đọc nhãn/splits và phần cứng; chưa training. CSV bảng đặc trưng không được coi là dataset ảnh. Nếu phát hiện bảng/file nén/thiếu ảnh, CLI giải thích nguyên nhân và cho nhập dataset khác ngay tại bước này.
2. Nhập **budget theo phút**, epoch cho **mỗi lượt full training**, tập cấu hình và seeds. Core có 12 ứng viên; compact có 3 ứng viên LR; hoặc nhập file grid JSON của bạn. Khuyên dùng 3 seeds để đo biến thiên. Một seed ít chi phí hơn nhưng không có ước lượng biến thiên giữa seeds.
3. Nếu cần, đặt weights accuracy/time/memory, trần bộ nhớ, trần thời gian mỗi full lượt, thang chuẩn hóa và thư mục kết quả. Xem lại toàn bộ kế hoạch và lệnh tương đương.
4. Chọn **chạy thật**, **chỉ lưu protocol**, hoặc **hủy**. Chỉ thực hiện sau xác nhận `y`; mặc định là **không**.

Nhập `b` ở câu hỏi bất kỳ để quay lại; `Ctrl+C` hủy hoặc dừng worker an toàn. Đóng đầu vào/EOF khi đang nhập không khởi động job. Thời gian bạn suy nghĩ/nhập menu không tính vào budget search. Kiểm tra trước trong wizard là thao tác đọc riêng; preparation/hash/export khi bắt đầu optimize vẫn được tính vào budget theo protocol.

**Budget mặc định menu là20phút; optimize là1200giây.** Đây là giá trị khởi đầu, không phải dự báo đủ cho dataset. Budget dùng chung giữa cấu hình và seeds. Mỗi worker còn có timeout riêng; hết budget có thể chỉ có best observed hoặc chưa có cấu hình đủ điều kiện. Thuật toán không tự giảm epoch/seeds để giả thành công. Đây là soft end-to-end deadline có dự phòng final audit theo chi phí quan sát; không bảo đảm deadline tuyệt đối trên mọi filesystem.

## Điều khiển hiệu năng và đóng gói

```powershell
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\images" --workers auto --image-cache-mb 64 --pin-memory auto
.venv\Scripts\python.exe -m research_pipeline.v2 optimize --data "D:\datasets\images" --method successive_halving --probe-fraction 0.1 --rungs 1 2 4 --selection-rule topsis
.venv\Scripts\python.exe -m research_pipeline.v2 benchmark-report --run "runs/benchmark/<study>"
```

- `--workers`: auto hoặc0..8. Auto bảo thủ:0 cho CPU/dữ liệu nhỏ, tối đa2 với CUDA và≥8192ảnh; không gọi đây là workers tối ưu. Spawn Windows có chi phí lớn, nên đo trước khi tăng.
- `--image-cache-mb`: tổng trần decoded-image caches train/val và workers, mặc định64MiB, còn chặn ở10% RAM khả dụng.0=tắt. Caption dùng chung ảnh decoded giữa các câu; không cache feature của encoder đang train. Detection vẫn tắt RAM cache YOLO không giới hạn. Model/tensors/prefetch/pinned queues không nằm trong trần này.
- `--pin-memory auto/on/off`: chỉ bật pin/truyền nonblocking trên CUDA; request và thực tế lưu trong kết quả.
- SH policy mới giữ quota probe và ngân sách full-confirmation. Nếu chưa có forecast, ưu tiên bootstrap một cấu hình đầy đủ seeds; rung không đủ ngân sách được ghi deferred, không giả nói đã loại ứng viên bằng SH.
- `--selection-rule weighted/tchebycheff/topsis/knee` khóa trước search. Knee không dùng weights trừ fallback Tchebycheff khi mặt phẳng suy biến. Preset chọn trong observed feasible Pareto front.
- `benchmark-report` chỉ audit/xuất lại study với reference đủ mọi cấu hình/seeds, không training. Không thể hợp thức hóa hard crash thiếu worker cost. Resume hiện tiếp tục search bookkeeping, **chưa** nối epoch/model/optimizer/RNG của trial dang dở.

Package cài thường dùng `python -m research_pipeline.v2` hoặc `train-config` ngoài source checkout. Sau setup thành công, launcher ưu tiên interpreter trong `.train-config-runtime.json`, hiển thị đường dẫn; ghi đè bằng `.\cli.ps1 -Python .venv-cpu\Scripts\python.exe`.

Dependency floor0.9.1 là PyTorch≥2.6/torchvision≥0.21, khớp setup/runtime CUDA đã kiểm chứng; không còn khai báo hỗ trợ Torch2.3 với AMP API cũ. Đây không có nghĩa mọi phiên bản/platform mới đều đã qua nghiệm thu; xem matrix thực tế trong validation. Không nâng driver/dependencies ML đang cài khi chỉ cập nhật editable `--no-deps`.

`report.json` ghi execution policy, integrity cache và finalization. Full hash bắt buộc đầu/cuối execution; giữa trials dùng metadata cache, không chống giả mạo. `online_session_seconds` protocol mới tính đến primary report/CLI export, không gồm lần ghi cuối timing receipt; protocol cũ giữ cách đo cũ. Exit3 nghĩa là train-v2 không đủ bằng chứng thực thi; exit0 optimize vẫn phải kiểm tra verification_status, không đồng nghĩa có nghiệm tối ưu.

Detection mặc định dùng YOLO11 YAML, không pretrained. Dùng grid JSON và model/checkpoint local khi muốn kiểm soát pretrained, image/batch/LR… CLI không tự quyết định thay mô hình bên ngoài phạm vi đã chọn. V2 hiện hỗ trợ detection, classification và captioning có nhãn; không phải mọi dạng dataset hoặc mọi framework AI.

Ví dụ `archive/marketing_campaign.csv` có các cột `Income`, `Education`, `Response` là **dữ liệu bảng**, không phải caption CSV. Chọn trực tiếp file CSV cũng không khắc phục thiếu adapter. Cần thêm nhánh tabular và xác định cột mục tiêu, bài toán/metric, cách chia dữ liệu và preprocessing train-only trước khi nghiên cứu cấu hình. CLI không tự đoán target, tạo ảnh/nhãn giả hay trả cấu hình YOLO cho bảng này. Chẩn đoán chỉ đọc header có giới hạn, không hiển thị các bản ghi khách hàng.

Nếu lệnh `Set-ExecutionPolicy ... Activate.ps1` xuất hiện ở câu hỏi menu, nó đã được gửi vào terminal khi CLI đang chờ nhập, không phải lựa chọn menu. CLI không thực thi chuỗi này. Hãy hoàn tất kích hoạt môi trường trước khi mở CLI, hoặc gọi trực tiếp `.venv\Scripts\python.exe train_config.py` mà không cần activate. Nếu terminal/editor tự gửi lệnh kích hoạt, chờ nó hoàn tất rồi chạy lại CLI; khi menu xuất hiện chỉ nhập lựa chọn được hỏi.

## Khi đang chạy

Terminal hiển thị giai đoạn chuẩn bị, thư mục mới, budget còn lại, cấu hình/seed, **PROBE** hay **FULL**, epoch hoàn tất, thời gian trôi qua và đường dẫn worker log. Cập nhật khi đổi epoch, hoặc heartbeat khoảng 20 giây khi epoch chưa xong. Đây là tiến độ đã ghi, không phải % tổng search hoặc ETA được bảo đảm. Output dạng dòng mới nên vẫn đọc được khi lưu log.

Mở PowerShell khác để đọc trạng thái:

```powershell
.venv\Scripts\python.exe train_config.py status --run "runs/v2/<luot>"
```

`status` kiểm tra PID và thời điểm tạo process để tránh nhầm PID đã tái sử dụng. Nó không khởi động training và không audit toàn bộ chứng cứ. Trạng thái lưu `search` không chứng minh worker còn sống. Dùng `audit` để xác minh chứng cứ.

## Quản lý kết quả

Menu **4** liệt kê 20 lượt mới nhất, gồm v2, nhánh research detection cũ và training cũ. Có thể nhập đường dẫn trực tiếp để mở lượt khác. Với v2 optimization:

- **Xem**: trạng thái, cấu hình, validation mean/SD, full training coverage, thời gian và loại bộ nhớ, test đã lưu nếu có.
- **Audit**: kiểm tra request/result, checkpoint, source snapshots, event ledger và tính lại lựa chọn; không training.
- **Bắt đầu / Resume**: protocol chưa chạy được thực thi nguyên trạng; resume yêu cầu budget **bổ sung** và xác nhận.
- **Train lại**: chạy cấu hình đã chọn trong thư mục mới, tách khỏi search. Đổi GPU/dataset/horizon không kế thừa chứng nhận tối ưu.
- **Test**: chỉ sau lựa chọn đã khóa, có xác nhận; nếu đã mở test, cần đồng ý rõ ràng cho retest. Không tự dùng test để chọn lại cấu hình.

Lượt deployment chỉ cho xem chứng cứ thực thi; không gán thành lượt optimization. Lượt training/research cũ được mở bằng lệnh tương thích cũ.

Kết quả phân biệt:

- **Tối ưu thực nghiệm trong tập đã đo đủ**: đủ full-data/full-horizon/all-required-seeds cho toàn bộ tập khai báo và có cấu hình thỏa objective/constraints.
- **Tốt nhất đã đo**: cấu hình được chọn đủ full training/seeds, nhưng tập ứng viên còn chưa đo hết.
- **Chưa có cấu hình đủ điều kiện**: không thay bằng probe, metric bằng 0 giả hoặc cấu hình chưa xác minh.

Chứng nhận chỉ áp dụng cho dữ liệu/máy/objective/grid/horizon đã khóa. SD không phải confidence interval. RAM RSS trên CPU không phải VRAM; VRAM reserved của allocator không phải bộ nhớ toàn thiết bị. Thời gian mỗi full lượt không phải toàn bộ chi phí search. Test một seed không được trình bày thành trung bình nhiều seed.

## Dùng trong script

```powershell
.venv\Scripts\python.exe train_config.py --help
.venv\Scripts\python.exe train_config.py optimize --help
.venv\Scripts\python.exe train_config.py optimize --data "D:\datasets\my dataset" --device auto --space core --full-epochs 20 --seeds 17 29 43 --budget-seconds 3600
.venv\Scripts\python.exe train_config.py runs --root runs --limit 30
.venv\Scripts\python.exe train_config.py inspect --data "D:\datasets\my dataset" --json
.venv\Scripts\python.exe train_config.py doctor --json
.venv\Scripts\python.exe train_config.py status --run "runs/v2/<luot>" --json
.venv\Scripts\python.exe train_config.py audit --run "runs/v2/<luot>"
.venv\Scripts\python.exe train_config.py resume --run "runs/v2/<luot>" --additional-budget-seconds 3600
```

`inspect`/`doctor` mặc định hiển thị dễ đọc; dùng `--json` để lấy dạng máy đọc. `status --json` trả `saved_status`/`saved_report` và ghi rõ không kiểm tra PID hoặc audit. `runs --json` trả danh sách. Không hỗ trợ JSON stdout cho optimize có tiến độ; hãy đọc artifact JSON sau khi chạy.

Exit codes: `0` lệnh thành công/thoát menu/hủy trước chạy; `2` tham số, dữ liệu, môi trường hoặc chứng cứ không hợp lệ; `130` Ctrl+C. Optimize có thể kết thúc hợp lệ với quyết định **chưa có cấu hình đủ điều kiện** và exit `0`; phải kiểm tra `report.json.decision.verification_status`, không suy ra optimum từ exit code. Trong menu, lỗi thao tác được hiển thị rồi quay lại để bạn sửa; exit menu không có nghĩa các thao tác trước đã thành công. Dùng `--debug` để giữ traceback chẩn đoán.

Nếu audit sau Ctrl+C báo `Completed worker omitted from state; reconcile before resume`, giữ nguyên thư mục: outcome đã có trong journal nhưng chưa lưu vào scheduler state. Lệnh resume chính thức sẽ reconcile trước audit; cần budget bổ sung và source/runtime còn tương thích. Không tự sửa state/hashes. Nếu mất điện/hard crash khiến chi phí chưa xác định, backend không tự coi chi phí đó bằng 0.

## Lượt cũ và nâng cấp source

Không đổi các checkpoint, dataset, metric hoặc ledger đã lưu. Những lệnh `--data ...`, `show-training`, `research...` vẫn giữ routing cũ; `legacy-help` in trợ giúp riêng. `--data` không có tên lệnh vẫn là pipeline cũ, không tự chuyển thành optimize v2.

Protocol khóa hash source: nâng cấp CLI cũng làm source khác. `status` sẽ cảnh báo; **resume và evaluate-v2 của protocol cũ bị từ chối nếu không tương thích source/runtime**. Không sửa hash hoặc bỏ qua guard. Vẫn xem và audit bằng snapshot gốc được; tạo lượt mới cho lần optimize dùng mã mới. Đây là bảo vệ tính tái lập, không phải lỗi giao diện. UI không xóa dữ liệu, tự giết job khác hay sửa protocol để tiếp tục.

Chi tiết thuật toán và giới hạn khoa học vẫn ở `PROTOCOL_V2.md`, `PIPELINE_V2.md` và `UPGRADE_VALIDATION.md`. Nâng cấp UX không tạo thêm chứng cứ tối ưu trên GPU/dataset chưa kiểm chứng.

Kiểm chứng phiên bản giao diện này: [CLI_VALIDATION.md](CLI_VALIDATION.md).
