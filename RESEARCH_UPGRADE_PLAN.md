# Kế hoạch nâng cấp nghiên cứu: tìm cấu hình huấn luyện theo ngân sách và bằng chứng

Ngày lập: 13/10/2026. Trạng thái: kế hoạch mới, chưa triển khai các hạng mục dưới đây.

Cập nhật triển khai: pipeline 0.7.1 đã thực hiện các hạng mục lõi và thực nghiệm local; trạng thái từng nhóm ở [UPGRADE_PROGRESS.md](UPGRADE_PROGRESS.md), raw findings ở [UPGRADE_VALIDATION.md](UPGRADE_VALIDATION.md). Nội dung dưới đây giữ nguyên như thiết kế ban đầu; không có nghĩa tất cả G0–G8 đã hoàn thành.

Kế hoạch này tiếp nối [RESEARCH_PLAN.md](RESEARCH_PLAN.md), không thay thế hoặc sửa kết quả lịch sử. Bằng chứng hiện tại nằm trong [DETECTION_VALIDATION.md](DETECTION_VALIDATION.md); phương pháp hiện tại nằm trong [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## 1. Mục tiêu và những điều không được hứa

Câu hỏi chính của đề tài:

> Với dataset, phần cứng, điều kiện huấn luyện và ngân sách tìm kiếm đã cho, làm thế nào tìm được cấu hình chạy thật, đạt các giới hạn yêu cầu, và cân bằng tốt giữa chất lượng, bộ nhớ và thời gian?

Sản phẩm phải trả lời được bốn câu hỏi: cấu hình nào đã chạy được; cấu hình nào chưa đủ bằng chứng; vì sao chọn cấu hình này; mức độ tin cậy và phạm vi của kết luận là gì.

Không hứa tìm tối ưu toàn cục trên mọi máy hoặc mọi dataset. Chương trình sẽ hỗ trợ các task/định dạng/backend công bố rõ; đầu vào không được hỗ trợ phải báo lỗi có hướng dẫn, không tự tạo nhãn hoặc đoán một cấu hình rồi gọi là tối ưu. Máy không đủ tài nguyên có thể nhận kết quả `insufficient_resources`, thay vì một chứng nhận giả.

Ba mức kết luận bắt buộc:

| Mức | Bằng chứng cần có | Được phép nói |
|---|---|---|
| Dự kiến | Metadata/probe/predictor, chưa đủ full confirmations | Cấu hình ứng viên; chưa xác minh |
| Tốt nhất đã xác minh | Full data, đủ horizon/seeds, actual settings khớp, constraints đạt | Tốt nhất trong các cấu hình đã xác minh |
| Tối ưu thực nghiệm trong tập khai báo | Mọi ứng viên trong tập hữu hạn đã có kết quả quyết định được theo cùng protocol | Tối ưu trong đúng tập, dataset, hardware, horizon, seeds và objective đó |

Không đổi certificate cũ thành certificate cho GPU/dataset/horizon mới. Prediction interval, confidence interval của mean, và SD giữa seeds là ba đại lượng khác nhau.

## 2. Vì sao phải nâng cấp tiếp

Các số sau là kết quả đã đo, không phải mục tiêu tương lai:

| Vấn đề | Bằng chứng hiện có | Hướng sửa |
|---|---|---|
| Profile tiêu ngân sách trước khi có đáp án | Pilot detection 60 giây/lượt: adaptive có lựa chọn đủ bằng chứng 1/3; Random full 3/3 | Có incumbent sớm, profile có chọn lọc, dự trữ ngân sách xác minh |
| Ngoại suy tài nguyên chưa đáng tin | Production c01: dự báo 260 MiB/14.64 giây; đo 318 MiB/23.05 giây | Đo đủ các phase; hiệu chỉnh trên dữ liệu độc lập |
| Không gian quá nhỏ | Detection mới 3 cấu hình, 4 epoch; nhiều yếu tố thay đổi đồng thời | Grid có cấu trúc, kiểm soát yếu tố, horizon dài hơn |
| Chưa chứng minh chuyển GPU | Registry có 4 dataset content/splits nhưng chỉ 1 GPU CUDA vật lý | Holdout theo dataset và GPU vật lý, không gọi CPU là GPU thứ hai |
| Đối chứng SH chưa tách hết nguyên nhân | Promotion theo quality nhưng lựa chọn cuối theo utility | Bổ sung scalarized-utility SH và đối chứng cùng fidelity |
| Đếm thời gian chưa hết từ lúc gọi CLI | Setup hiện tại bắt đầu sau discovery/runtime probe | Ledger end-to-end, tính cả overhead và thất bại |
| Test Penn-Fudan đã xem | Seed 17 test mAP50-95 = 0.622469 | Dùng như lịch sử; kiểm định mới cần holdout mới |

Các finding bất lợi cho phương pháp hiện tại phải giữ nguyên. Nâng cấp chỉ được công bố tốt hơn nếu nghiên cứu mới chứng minh điều đó.

## 3. Đóng góp nghiên cứu dự kiến và giả thuyết kiểm định

Tên làm việc: **tìm kiếm có xét chi phí và dự trữ xác minh**. Đây là tên mô tả, chưa phải tuyên bố một thuật toán mới đã có bảo đảm lý thuyết.

Đóng góp cần kiểm định là cơ chế phối hợp: chọn probe khi có ích, bảo vệ ngân sách full confirmation, dự báo tài nguyên có hiệu chỉnh, và hạ mức kết luận khi chuyển miền. Không coi việc ghép Random Search, Pareto và Bayesian optimization là đủ để chứng minh tính mới.

| Giả thuyết | Đối chứng trực tiếp | Đại lượng chính |
|---|---|---|
| H1: reserve + profile gating giảm thất bại ở ngân sách nhỏ | Policy hiện tại; bỏ reserve; profile mọi config | Tỷ lệ có lựa chọn feasible theo ngân sách |
| H2: predictor hiệu chỉnh giúp scheduling an toàn hơn | Ngoại suy hiện tại; point predictor không calibration | Underprediction, interval coverage/width, OOM, overrun |
| H3: cost-aware ordering cải thiện chất lượng lựa chọn khi ngân sách đủ | Random full; Random proxy; utility-SH | Regret cùng objective, với feasibility được báo riêng |
| H4: cập nhật từ probe tại máy mới tốt hơn dùng predictor cũ nguyên trạng | Cold start; transfer không cập nhật | Feasibility, chi phí thích nghi, sai số theo miền chưa thấy |

Thứ tự ưu tiên là H1 → H2 → H3 → H4. Nếu phương pháp chỉ tốt hơn ở một số miền/ngân sách, công bố đúng điều kiện đó. Nếu Random vẫn tốt hơn, giữ Random làm mặc định tại miền ấy.

## 4. Mô hình toán và protocol phải khóa trước

### 4.1. Đầu vào

- D: dataset, labels, groups, split và content identity.
- H: backend, GPU vật lý nếu có, CPU/RAM, runtime/driver, trạng thái máy.
- P: metric, horizon E, checkpoint rule, augmentation, optimizer/scheduler semantics và các seed.
- C: danh sách cấu hình hữu hạn, tạo và hash trước search.
- B: ngân sách wall-clock online cho một lần tạo recommendation, dùng chung giữa seeds.
- Các giới hạn: chất lượng tối thiểu, thời gian full training tối đa, bộ nhớ, disk và deadline.

Tách `split_seed`, `training_seeds` và `search_seed`. Số lần lặp search không được nhầm với số seed full training. Bộ seeds xác minh là một phần protocol; mỗi phương pháp phải đáp ứng cùng yêu cầu.

### 4.2. Hàm mục tiêu

Giữ một objective chính dễ giải thích:

```text
U(c) = wA * mean(A_c)
     - wT * mean(T_c) / T_ref
     - wM * max(M_c) / M_ref

wA, wT, wM >= 0; tổng weights = 1; T_ref, M_ref > 0.
```

A là metric validation theo task; T là full-worker wall time; M có counter/unit cố định trong protocol. Anchors là mốc ưu tiên được khai báo trước, không chuẩn hóa bằng min/max của oracle sau thực nghiệm. Search cost không lẫn với T; B là ràng buộc riêng, chi phí tìm kiếm được báo riêng.

Feasible yêu cầu đủ full data/E/seeds, loss hữu hạn, optimizer thực sự cập nhật, actual configuration đúng, checkpoint hợp lệ và constraints đạt. Khóa rõ rule constraints, chẳng hạn min(A) ≥ A_min, max(T) ≤ T_max, max(M) ≤ M_max qua required seeds; không tráo mean và worst-case sau khi đo. Nếu A_min = 0 chỉ để kiểm workflow, không gọi đó là chứng nhận chất lượng ứng dụng. Proxy chưa đủ; timeout/chưa đo là unresolved, không gán quality = 0. OOM quan sát ở seed bắt buộc làm cấu hình không đạt protocol đó, không chứng minh cấu hình luôn OOM ở mọi máy/seed.

Với weights dương, tối đa U không bị trội theo mean(A), mean(T), max(M): một cấu hình trội nghiêm ở ít nhất một trục sẽ có U lớn hơn. Tính chất này không chứng minh policy tìm kiếm sẽ tìm được nghiệm đó. Weighted sum cũng không nhất thiết quét được toàn bộ Pareto front không lồi; vì vậy luôn xuất cả Pareto set.

### 4.3. Bảo vệ đánh giá khỏi việc chọn quá mức trên validation

- Train: cập nhật weights.
- Validation-search: chọn checkpoint/config và tạo reference của cùng objective.
- Validation-audit nếu đủ dữ liệu: đánh giá lựa chọn đã khóa, không điều khiển search.
- Test: chỉ mở sau khi protocol/lựa chọn đã khóa; mọi lần mở đều ghi exposure.

Nếu dữ liệu quá ít, dùng nested/grouped splits ở nghiên cứu, không ép bốn split nhỏ đến mức metric vô nghĩa. Đơn vị group là scene/video/subject khi có metadata thật; filename grouping chỉ là approximation và phải ghi rõ. Báo exact/near-duplicate và nguồn pretrained/overlap đã biết. Reference regret trên validation-search không thay thế chất lượng generalization trên outer holdout. Khóa cả rule chọn checkpoint/seed để deploy hoặc evaluate; không chọn seed đẹp nhất bằng test.

### 4.4. Minh chứng toán học dự kiến cho bộ điều phối ngân sách

Gọi b là ngân sách còn lại, r là tổng upper costs của những confirmations đã reserve cộng cleanup, và k là upper cost của action khám phá mới. Chỉ mở action nếu k + r ≤ b. Nếu actual action cost ≤ k và các confirmations thực sự có cost không vượt reserve, sau action còn ít nhất r để hoàn tất công việc đã reserve. Lặp lại bất đẳng thức cho ta invariant bằng quy nạp; thêm finalist phải kiểm lại tổng reserve mới.

Đây là tính chất **có điều kiện về cost bounds**, không bảo đảm thực tế khi bounds chỉ là forecast, phần cứng đổi trạng thái hoặc training thất bại. Nó bảo vệ khả năng cấp thời gian, không chứng minh quality đạt ngưỡng hoặc regret nhỏ. Cần property tests với exact bounds, tests cố ý underprediction, và số đo violation ngoài thực tế. Không dùng prediction interval marginal để tuyên bố mọi action đều thỏa giả thiết.

## 5. Kiến trúc nâng cấp và ranh giới triển khai

Chỉ phát triển phiên bản mới khi triển khai được yêu cầu; không sửa snapshots/runs lịch sử, không sửa source/runtime của job đang chạy. Thiết kế module mới trong `research_pipeline`, tách schema/protocol khỏi policy.

```text
Dataset adapter + hardware probe
              ↓
Protocol / candidate space / content-addressed manifests
              ↓
Budget controller → action scheduler → isolated real worker
                         ↑                 ↓
                   predictor ← event ledger / phase telemetry
                                           ↓
                               full-confirmation registry
                                           ↓
                             Pareto + selection + audit report
                                           ↓
                             locked test / deployment reverify
```

Các file dưới đây là **dự kiến**, chưa có nghĩa đã triển khai:

| Thành phần | File dự kiến | Trách nhiệm |
|---|---|---|
| Schema/protocol | `research_pipeline/schema.py`, `protocol.py` | Validate, version, hash; phân biệt predicted/observed/certified |
| Candidate space | `research_pipeline/space.py` | Task-aware fields, compatibility, IDs, effective batch |
| Ledger/telemetry | `research_pipeline/ledger.py`, `telemetry.py` | Event journal, timing, counters/phase/unit, provenance |
| Predictor | `research_pipeline/predictors.py`, `calibration.py` | T/M trước; A tùy chọn; uncertainty/OOD |
| Budget/controller | `research_pipeline/budget.py`, `scheduler_v2.py` | Incumbent, reserve, action ordering, deadline |
| Task contract | `research_pipeline/adapters/` | Worker capability và metric semantics theo task |
| Analysis | `research_pipeline/statistics.py` | Paired evaluation, missingness, CI, finite-reference scope |
| Benchmark manifest | `configs/benchmark_protocol_v2.json` | Domains, grids, seeds, budgets, hypotheses đã khóa |

Nâng cấp có kiểm soát `runner.py`, `worker.py`, `integrity.py`, `study.py`, `report.py`, `cli.py`, `evaluation.py`. Caption/classification được tích hợp qua contract rồi kiểm tra tương đương, không viết lại model chỉ để đồng nhất giao diện. Segmentation/LLM/distributed training chưa nằm trong phạm vi đầu tiên.

## 6. G0–G3: nền tảng đúng, đo đúng, tìm kiếm không tự hết ngân sách

### G0 — Đặc tả, dữ liệu và chuẩn tái lập [P0]

1. Audit các run cũ; ghi bản đồ lỗi/rủi ro, giữ kết quả 0.5/0.6 và checkpoint gốc.
2. Tạo protocol schema v2 và schema migration đọc lịch sử, không ghi đè lịch sử.
3. Tách task detection/classification/captioning; supported formats, label validation, corruption/group/duplicate checks và cảnh báo split nhỏ.
4. Khóa objective, candidate IDs, seeds, fidelity, timeout, test exposure, source/runtime/model/dataset hashes.
5. Chuẩn bị manifest nghiên cứu mới: development/calibration/heldout domains và claims tương ứng; Penn-Fudan cũ chỉ là development/historical evidence.

Đầu ra: `PROTOCOL_V2.md`, schema, manifest, audit tests. Nghiệm thu: cùng input tạo cùng IDs; sửa nhãn/model/source làm identity đổi; unsupported task không nhận recommendation giả; không có code path đọc oracle/test vào search.

### G1 — End-to-end ledger, telemetry và độ bền [P0]

1. Bắt đầu monotonic clock từ lúc CLI vào workflow. Charge discovery/hash, probe, worker startup, dataloader, train, val, checkpoint, predictor fitting/inference, controller, audit trước commit và failed/retry attempts. B online kết thúc tại selection commit; post-commit report/export và test có cost riêng, elapsed toàn workflow vẫn công bố.
2. Báo riêng: cold setup; online search + confirmations; offline reference; test; report/export; elapsed toàn workflow. Downloads/cài môi trường bên ngoài session có ledger riêng, không giấu như chi phí miễn phí. Không so cold-start method với warm-cache method mà không kiểm soát.
3. Đo theo phase: initialization, optimizer first step, steady-state train, validation, best/last saving. Ghi CUDA allocated/reserved và NVML/device counter khi khả dụng; CPU RSS theo process tree với định nghĩa tránh nhầm shared pages.
4. Chọn primary memory counter cố định cho từng protocol. Nếu process VRAM không đọc được trên nền tảng đó, dùng allocator với nhãn rõ; tổng GPU usage là diagnostic, không tự đổi thành process VRAM. Không so RSS với VRAM trong một ranking chung.
5. Ghi sampling interval, peaks có thể bị bỏ sót, GPU free memory trước trial, power/thermal/clock nếu backend hỗ trợ; contention detection không được gọi là cô lập phần cứng tuyệt đối.
6. Atomic event journal + selection commit; request ID/attempt ID; audit trước khi skip trial ở resume. Resume giữ failed cost, kiểm PID/create-time/source/data/checkpoint; không tự tiếp tục nếu identity không khớp.
7. Tách workflow resume với checkpoint-training continuation. Continuation chỉ hỗ trợ khi restore đủ optimizer/scheduler/scaler/RNG/sampler và có equivalence test; nếu chưa có thì restart trial, ghi đúng chi phí.
8. Deadline có cleanup grace được khai báo; overrun đo thật. Không mở trial khi không đủ thời gian shutdown an toàn; không tuyên bố upper predictor luôn chặn được deadline.

Đầu ra: ledger JSONL, phase telemetry, recovery tests, report reconciler. Nghiệm thu: chi phí không biến mất sau failure/resume; worker mồ côi bị phát hiện; raw artifacts bị sửa không được dùng như full confirmation; mọi counter có unit/source/availability; các khoảng exclusive cộng được và không double-count thời gian lồng nhau.

### G2 — Không gian tham số có cấu trúc [P1]

1. Tạo capability matrix: model/task/backend/precision/optimizer hợp lệ, initialization riêng với architecture, field không hoạt động được loại trước khi hash.
2. Batch và accumulation phải tách: `nominal_effective_batch = batch × accumulation` cho single-device; minibatch cuối và optimizer-step count được ghi. Không coi mọi batch có cùng effective batch là tương đương về chất lượng/thời gian.
3. Stage đầu khóa augmentation/scheduler/workers/cache để nghiên cứu model/image/batch/precision/LR/optimizer/decay. Stage sau mới thêm workers/cache/augmentation; không mở mọi chiều cùng lúc.
4. Tạo `C_core` khoảng 12–24 cấu hình từ thiết kế cân bằng/đối chứng một yếu tố, không cần full factorial khổng lồ. Có cặp chỉ đổi batch/accum, chỉ đổi image, chỉ đổi LR, và chỉ đổi model khi khả thi.
5. Tạo `C_large` hữu hạn có giới hạn, ví dụ tối đa 512 ứng viên, bằng generator + seed khóa trước. Mọi phương pháp nhận cùng C; canonical dedup tránh đánh giá config tương đương hai lần.
6. VRAM prediction chỉ sàng lọc rủi ro/scheduling, không tự chứng minh toàn bộ config bị dự đoán OOM là không khả thi. Cấu hình không thử giữ trạng thái unresolved.

Đầu ra: space schema/generator, core/large manifests, comparison pairs. Nghiệm thu: actual model/imgsz/batch/AMP/accum/LR/decay/scheduler khớp; không silent fallback; bảng cấu hình giải thích được yếu tố thay đổi. “512” là giới hạn thiết kế, không phải 512 lượt full training sẽ tự động chạy.

### G3 — Scheduler v2 không cần quality surrogate [P0/P1]

1. Bỏ rule mặc định profile mọi config. Dùng metadata + probe ngắn, gồm optimizer/validation phase cần thiết; hạn mức probe là policy được khóa.
2. Xây incumbent bằng cách full-confirm một ứng viên nhỏ/ít rủi ro sớm, nếu B đủ. Chỉ gọi feasible sau khi đạt đủ seed/constraints; “nhỏ” không đồng nghĩa chắc chắn đạt chất lượng.
3. B là shared budget cho toàn bộ session, không ba budget riêng khiến mỗi seed chọn config khác rồi không ghép được recommendation.
4. Dự trữ chi phí các seed còn thiếu của finalist, setup/cleanup và uncertainty margin; cập nhật khi có số đo. Không dùng reserve để hứa chắc completion khi forecast chưa đáng tin.
5. Khi budget nhỏ, ưu tiên có một config full-confirmed trước mở rộng. Khi budget lớn, mở rộng có kiểm soát theo model/LR/image strata, không chỉ LR.
6. Profile một ứng viên khi cần kiểm tra memory risk/cost uncertainty và còn đủ reserve; tránh profile lặp đã có evidence tương thích. Cache hit có cost thật, và fair baseline được cùng quyền truy cập cache/prior.
7. Dừng khám phá trước khi reserve bị tiêu; final selection chỉ từ full-confirmed set. Nếu không có ứng viên đạt, xuất lý do `insufficient_budget`/`no_verified_selection`, không nâng proxy thành full training.

Tham số development sweep dự kiến: probe cap 5%/10%/20% B; ít nhất một policy không probe; số finalists 1/2/3. Chọn policy bằng development data rồi khóa trước heldout benchmark, không đổi theo kết quả test. Reserve ưu tiên công việc đã thiếu seed hơn việc mở hàng loạt finalist không hoàn tất.

Đầu ra: `scheduler_v2.py`, budget tests và real low-budget pilot. Nghiệm thu: không repeat all-profile starvation; xác minh từ raw worker; benchmark đối chiếu Random phải tính cả confirmations của mọi phương pháp. Gate này kiểm tra feasibility và cost, chưa tuyên bố thắng về accuracy/regret.

## 7. G4 — Dự báo có hiệu chỉnh, chuyển máy có kiểm soát [P1]

### 7.1. Thứ tự xây predictor

Xây T/M trước, A sau nếu dữ liệu đủ. Không suy chất lượng chỉ từ GPU name, batch hoặc VRAM.

- T: mô hình thành phần startup + train steps + validation + checkpoint/I/O; dùng features N ảnh/tokens, E, resolution, batch, accumulation, model size, optimizer, backend, workers/cache và probe throughput.
- M: parameters/optimizer state/precision + activation features và phase peaks; so mechanistic baseline với regressor residual. Không cố áp một công thức linear resolution cho mọi kiến trúc/backend.
- A tùy chọn: surrogate theo task/metric/dataset và learning curves. Không trộn BLEU, accuracy và mAP thành cùng target; không dùng early loss thay mAP mà không kiểm định correlation/ranking.

Các observations có fidelity và censoring fields. Timeout không cho biết full T chính xác; OOM là failure event, không phải quality = 0 hoặc memory bằng GPU capacity. Đánh giá predictor success-only phải ghi selection bias; theo dõi feasibility model riêng.

### 7.2. Chia dữ liệu cho predictor

- Development-fit: fit và chọn features/models/hyperparameters.
- Calibration: hiệu chỉnh residual/interval sau khi predictor đã khóa.
- Heldout evaluation: chỉ đánh giá, không dùng để chỉnh predictor/policy.

Không random-split trial rows khiến cùng dataset/GPU/config family xuất hiện cả fit và heldout rồi gọi là transfer. Dùng grouped domain split: leave-one-dataset-out; leave-one-physical-GPU-out khi có đủ GPU; giữ out-domain cuối chưa thấy. Các thử nghiệm leave-one-out dùng để development không được đồng thời gọi là final blind test.

Báo MAE/median error/p90 relative error, underprediction, interval coverage/width, sample count và stratification theo task/GPU/model/phase. MAPE không dùng khi target gần zero. Dùng one-sided upper prediction interval cho scheduling nếu calibration đủ; nếu không thì gắn `uncalibrated`.

Split conformal là một lựa chọn hiệu chỉnh, không phải bảo đảm trên GPU chưa thấy: marginal coverage cần điều kiện exchangeability; không tự suy ra conditional coverage hoặc an toàn cho chuỗi chọn config thích nghi. Nguồn: [Angelopoulos & Bates, conformal prediction](https://arxiv.org/abs/2107.07511). Nếu muốn pruning dựa vào bounds đồng thời, phải có thiết kế/bảo đảm riêng; phiên bản đầu không hard-prune bằng bounds này.

### 7.3. Hành vi trên máy/dataset mới

1. Kiểm capability/runtime và scope của predictor.
2. Gắn nhãn in-domain/OOD/insufficient-calibration; GPU UUID mới không tự có calibration.
3. Probe tại máy mới, tính chi phí thích nghi online; update T/M với rule khóa trước.
4. Khi interval rộng hoặc OOD: fallback scheduler v2 đo thật; không xuất “95% an toàn” từ lịch sử máy khác.
5. Nếu không có adapter/nhãn/validation đủ: dừng có hướng dẫn. CPU fallback tạo protocol mới và đo lại, không giữ certificate GPU.

Đầu ra: predictor cards + calibration manifests + coverage report. Nghiệm thu: model fit không đọc heldout/test; uncertainty đúng target/phạm vi; OOD fallback chạy thật. Nếu predictor không vượt extrapolation baseline hoặc interval vô dụng, không bật làm mặc định.

## 8. G5 — Thuật toán có xét lợi ích/chi phí và ablation [P1/P2]

Chỉ triển khai sau G3/G4. Không bắt đầu bằng một hệ Bayesian lớn khi evidence còn ít.

### 8.1. Hai chế độ thử nghiệm

- Chế độ cơ sở: scheduler v2, empirical T/M, đa dạng ứng viên, không dùng quality surrogate. Đây là ứng viên mặc định đầu tiên.
- Chế độ nâng cao: constrained utility improvement per cost nếu joint surrogate đủ tin cậy. Pareto-oriented acquisition là nhánh nghiên cứu riêng, không tráo objective chính sau khi biết kết quả.

Với một full-confirmation action a, điểm acquisition dự kiến:

```text
score(a) = E[(U_after(a) - U_incumbent)_+ * I(feasible_after(a))]
           / E[total_action_cost(a)]
```

Expectation là dự báo từ mô hình, không guarantee. Action cost gồm số seed thiếu và overhead; uncertainty joint A/T/M không được mặc định độc lập. Nếu chưa có incumbent thì ưu tiên tạo một ứng viên đầy đủ. Probe không tự có U full: chỉ dùng gate giảm bất định; nhánh value-of-information thực sự phải mô phỏng ảnh hưởng probe đến quyết định và tính cả overhead, triển khai sau.

Không hard-prune chỉ vì proxy quality thấp hoặc upper time forecast cao. Giữ budgeted exploration rule khóa trước, báo mọi ứng viên bỏ qua là unresolved. Chỉ exact full outcomes đủ scope mới vào recommendation; acquisition không biến forecast thành số đo.

### 8.2. Pseudocode dự kiến

```text
start_ledger_and_deadline()
validate_and_lock(D, H, P, C, objective, B)
load_compatible_prior_or_mark_ood()
probe_minimally_if_budget_allows()
try_full_confirming_low_risk_incumbent(required_seeds)
while remaining_budget > reserved_confirmation_and_cleanup_cost:
    update_cost_models_from_online_evidence_only()
    a = choose_full_candidate_or_targeted_probe(frozen_policy)
    if a does not fit budget/risk gate:
        mark_unresolved_and_choose_another_or_stop()
    execute_real_worker(a); charge_all_costs_and_failures()
    update_registry_and_confirmation_reserve()
complete_missing_seeds_of_affordable_finalists()
audit_raw_evidence()
select_feasible_argmax_U_and_export_scope()
lock_decision_before_any_outer_test_or_reference_analysis()
```

### 8.3. Đối chứng và ablation khóa trước

Core baselines: declared default; Random full; Random proxy cùng fidelity/finalizer; quality-SH; utility-SH; adaptive hiện tại; scheduler v2. Utility-SH dùng promotion score ở rung với mapping/extrapolation tới full objective được khóa, không đọc full oracle; quality-SH dùng cùng rungs, restart/continuation semantics và finalizer. Các partial T/M không được giả là full T/M. Default cũng phải full-confirm và tính cost, không được đáp án đủ seeds miễn phí. Random Search và phân bổ fidelity là cơ sở đối chứng, không bằng chứng cho policy mới: [Random Search](https://www.jmlr.org/papers/v13/bergstra12a.html), [Hyperband](https://www.jmlr.org/papers/v18/16-558.html).

Ablation từng thành phần: không reserve; profile mọi ứng viên; không profile; point estimate không calibration; không transfer prior; không diversification; không cost-aware acquisition. Không chạy full Cartesian product mọi ablation; dùng nhánh bật/tắt từng thành phần và ghi các interaction chưa tách.

Sau core, thêm baseline mạnh nếu đủ ngân sách: [BOHB](https://proceedings.mlr.press/v80/falkner18a.html) cho model-guided multi-fidelity; [NEHVI](https://proceedings.neurips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html) cho noisy multi-objective search. Phải khóa implementation/version/adapter, tính model-fitting overhead và đối chiếu mục tiêu/fidelity; không gắn tên bài báo cho một bản triển khai khác rồi mượn bảo đảm của nó.

Đầu ra: policy spec, parameter manifest, implementation tests, preregistered comparison sheet. Nghiệm thu: mọi baseline cùng C/P/B/seed requirements và cùng quyền sử dụng prior; không có full oracle trong online search; mock/replay chỉ kiểm logic, không thay real benchmark.

## 9. G6 — Thực nghiệm mở rộng, thống kê và tiêu chí kết luận [P2]

### 9.1. Ma trận theo giai đoạn

| Giai đoạn | Dataset/hardware | Không gian/horizon | Mục đích |
|---|---|---|---|
| Development | Penn-Fudan/Fashion/CIFAR/ViECAP lịch sử, RTX 3050 và CPU hiện có | Grid nhỏ; pilot rồi horizon dài hơn | Sửa pipeline/policy, ước lượng variance/cost |
| Core reference mới | Ít nhất 2 detection datasets mới có nhãn thật, một nhỏ và một khó hơn; RTX 3050 | C_core 12–24; E khóa sau development learning curves | So regret với reference đầy đủ, kiểm quality/resource trade-off |
| Generalization | Thêm classification và captioning domains chưa dùng chỉnh policy | Grid task-specific, metric riêng | Không chỉ tối ưu YOLO/Penn-Fudan |
| Hardware transfer | Tối thiểu 3 GPU vật lý tổng cộng, VRAM/architecture khác nhau nếu tiếp cận được | Same-domain và unseen-domain transfer | Leave-one-GPU-out, adaptation cost |
| Large-space stress | Một số domains đã khóa, C_large | B hữu hạn, không giả có oracle đầy đủ | Best observed, uncertainty, coverage/failure; không exact regret |

Chỉ có RTX 3050 hiện tại: phần nhiều GPU vẫn `not_established`. GPU model khác nhưng cùng card/runtime giả lập, CPU, hoặc mock OOM không thay thế GPU vật lý khác. Không tự mua/thuê GPU trong kế hoạch này. Dataset mới phải chọn theo nguồn/quyền sử dụng, true grouping, nhãn, quy mô và chi phí; chưa download hoặc chạy trong lượt lập kế hoạch.

Chọn budgets theo cold full-confirmation cost đo ở development, chẳng hạn các mức tương ứng khoảng 1/2/4/8 config-confirmations. Sau đó khóa wall-second budgets cho từng domain trước final benchmark. Không dùng future oracle time để đặt B. Các mức 60/120/300/600 giây có thể giữ làm historical stress test, không mặc định đủ cho mọi dataset.

Horizon tham khảo development: 4 → 12 → 20 epoch với detection nhỏ; dataset/model khác dùng learning curves và protocol riêng. Không khẳng định 20 epoch luôn hội tụ, cũng không suy ranking 4 epoch là ranking 20 epoch. Nếu nghiên cứu early stopping, tách thành protocol khác và tính chính sách dừng vào cấu hình.

### 9.2. Reference và giới hạn regret

- Với C_core: exhaustive full reference độc lập, cùng D/H/P/objective/seeds. Khóa online decisions trước oracle; oracle không cấp prior hoặc warmstart miễn phí cho search.
- Exact empirical regret chỉ khi reference trong toàn C đã giải quyết đủ và selected feasible: `r = U_ref(reference_best) - U_ref(selected_config)`. Cả hai utility lấy từ cùng independent reference, không trừ online time/quality của một worker khác rồi ép regret âm thành 0. Nếu online/reference bất đồng feasibility, báo disagreement và regret undefined theo rule đã khóa.
- Thiếu reference: regret `null`. Reference subset không được gọi là optimum của C_large; nếu selected ngoài subset, chỉ báo comparator gap với nhãn có thể âm, không gọi global regret.
- Reference chạy sau development vẫn phải fresh run/raw evidence; chi phí oracle công bố riêng. Cache/shared trials giữa methods chỉ dùng cho simulation riêng, không chứng minh online wall cost.

### 9.3. Thiết kế thống kê

1. Đơn vị độc lập cấp cao là dataset/domain × hardware; paired runs dùng cùng splits/training seeds/search seeds giữa methods. Epoch/batch/image không phải số mẫu độc lập để chứng minh thuật toán thắng.
2. Development: ít nhất 3 training seeds để đo biến thiên và 5–10 search repeats khi đủ chi phí. Đây là số khởi đầu dự kiến, không khẳng định đủ power.
3. Chọn final repeat count từ variance, minimum practically relevant effect và ngân sách; khóa trước heldout. Nếu power/miền quá ít, báo exploratory, không xuất p-value để tạo cảm giác chắc chắn.
4. Primary endpoint theo thứ tự: feasible-selection rate tại B; sau đó regret/utility trên outcomes có đủ reference. Báo regret có điều kiện cùng failure rate; không giấu failure hoặc gán regret 0 khi không có selection.
5. Báo paired effect + interval; bootstrap có group/domain structure khi đủ domains, hoặc phân tích paired theo domain. Interval từ một domain không là uncertainty của mọi dataset/GPU.
6. Chọn một primary policy/baseline/endpoint; các weights/budget/ablation phụ được gắn exploratory hoặc dùng correction đã khai báo. Không chọn policy thắng sau test rồi báo nó như primary.
7. HPO seed SD, image-resampling uncertainty của mAP, và time variability do hardware được báo riêng. Không lấy mean validation 3 seeds so với test một seed như paired estimate 3 seeds.
8. Khóa metadata/prior trước heldout; online adaptation được phép theo rule và tính cost, nhưng không sửa feature/policy bằng outer test. Nếu đã xem holdout để sửa, đổi nhãn development và cần holdout mới.

Secondary endpoints: OOM/timeout/NaN rate; budget overrun; time-to-first-verified-config; Pareto coverage/hypervolume với normalization/reference point khóa; search cost; worst-domain behavior; calibration coverage và width; test quality sau selection. Không cộng raw BLEU/mAP/accuracy thành “accuracy trung bình” giữa tasks.

### 9.4. Gates nghiên cứu, không phải cam kết kết quả

- G3 gate: không làm feasibility kém hơn baseline qua lỗi scheduling/ledger; nếu không có lợi thế, giữ baseline và sửa tiếp trên development.
- G4 gate: heldout prediction diagnostics đủ để bật predictor, đồng thời interval có độ rộng còn hữu ích. Coverage thấp/không đủ sample ⇒ hạ nhãn/fallback.
- G5/G6 claim gate: chứng minh improvement có ý nghĩa thực tiễn với uncertainty và budget trade-off; không chỉ một run có regret 0.
- Không cố định mục tiêu “95% mọi máy” từ vài seed. Nếu đặt tỷ lệ feasible/OOM mục tiêu, phải định nghĩa population, số domain/lặp và interval trước final experiment; tỷ lệ quan sát không phải bảo đảm tương lai.

Đầu ra: `BENCHMARK_PROTOCOL_V2.md`, raw runs, locked decisions, reference ledger, statistical report, hypothesis outcomes kể cả negative findings.

## 10. G7 — CLI dễ dùng, report đủ minh chứng, portability thật [P1/P2]

CLI dự kiến, **chưa phải lệnh đã hỗ trợ**:

```text
train_config.py inspect --data <path> --device auto
train_config.py plan-search --data <path> --objective <file> --budget <seconds>
train_config.py optimize --protocol <locked.json>
train_config.py status --run <path> --live
train_config.py audit --run <path>
train_config.py resume --run <path>
train_config.py train --selection <path>
train_config.py evaluate --selection <path> --split test
```

Giữ aliases/lệnh cũ có thông báo rõ. Flag không hỗ trợ phải reject, không silently ignore; phân biệt inspect/plan/proxy/full-training/test. Interactive menu là lớp giao diện, không logic nghiên cứu; luôn có non-interactive CLI để tái lập.

Report phải có: dataset/split/groups; GPU thật/backend/runtime; mọi ứng viên và trạng thái; actual configuration; full seed coverage/E/checkpoint; A/T/M mean/SD/max với counter/unit; Pareto; utility/margin; predicted vs measured và calibration scope; tất cả online/oracle/test costs; OOM/overrun; audit; source/raw artifact links; lý do và phạm vi optimality.

Khi số seed ít hoặc margin nhỏ, hiển thị “chưa phân biệt chắc thứ hạng” bằng uncertainty analysis đã định nghĩa, không biến point ranking thành chắc chắn. Xuất JSON machine-readable và Markdown; timeline CLI lấy event ledger, trạng thái process sống khác trạng thái đã lưu.

Portability gồm: kiểm Python/driver/backend/capabilities; dependency constraints + bootstrap theo nền tảng; CPU fallback protocol mới; local model manifests/relocatable bundles; không tải ngầm weights trong trial; kiểm disk/RAM; không yêu cầu mọi máy có CUDA. Giữ CPU/CUDA supported-runtime matrix; MPS/ROCm chỉ ghi supported sau khi có runner và máy kiểm chứng thực tế. Offline mode cần artifacts đã có, không tạo lời hứa chạy thiếu dependencies/data/model.

Đầu ra: command reference, report template, bootstrap/doctor, packaged entry points. Nghiệm thu: máy/runtime mới nhận workflow đúng hoặc diagnostic rõ, không “Verification Passed” giả; execution khác verification; deploy đổi scope phải hạ certificate và đo lại.

## 11. G8 — Kiểm thử, đóng gói và hồ sơ bảo vệ đề tài [P0/P2]

- Unit/property tests: objective/Pareto/ties/zero weights, config dedup, counter units, budget reserve, failures/censoring, domain splits và source hashes.
- Integration tests: actual full training, all-seed confirmations, optimizer/AMP/gradient accumulation, full val/save peaks, test lock/exposure, portability và unsupported flags.
- Failure tests: safe simulated OOM/NaN/timeouts, interrupt parent/worker, corrupted JSON/checkpoint, stale lock, PID reuse, identity change, interrupted journal/selection commit. Mock failure không thay benchmark physical OOM; không cố làm máy hết RAM.
- Resume tests: audit trước skip, giữ attempts/cost, chỉ jobs thiếu được chạy, job đang sống không bị chạy trùng; checkpoint continuation có equivalence test riêng nếu triển khai.
- Mỗi supported runtime phải qua regression suite + smoke thật; migration đọc old runs vẫn qua. Mốc hiện tại 126 tests không là giới hạn đủ, không thay correctness bằng mục tiêu đếm tests.
- Release nguồn/package/docs/protocol cùng version; reproducible manifests, dependency checks; không làm việc trong runtime job đang chạy. Không ghi đè raw evidence hoặc checkpoint người dùng.

Hồ sơ bảo vệ: problem/formal objective; thuật toán/pseudocode/complexity; proof Pareto đúng phạm vi; hypotheses + preregistration; baselines/ablations; predictor calibration; paired statistics; failure cases; recommendation có raw evidence; negative findings và limits. Hash/audit là reproducibility/integrity, không phải independent scientific registration hoặc hardware attestation.

Nghiệm thu cuối tách ba mức: phần mềm chạy đúng; nghiên cứu đo đủ phạm vi đã khóa; thuật toán có/không có ưu thế theo dữ liệu. Không đánh dấu “nghiên cứu hoàn thành” chỉ vì CLI chạy hoặc unit tests xanh.

## 12. Thứ tự thực hiện, ngân sách và điểm phải dừng

### 12.1. Roadmap theo phụ thuộc

| Mốc | Điều kiện bắt đầu | Kết quả phải bàn giao | Gate trước đi tiếp |
|---|---|---|---|
| M1: G0–G1 | Khi được yêu cầu triển khai | Schema + ledger + recovery/audit | Không mất cost/identity, hồi quy lịch sử qua |
| M2: G2–G3 | M1 qua | Grid có cấu trúc + reserve/gated profiling | Real low-budget runs, đối chiếu feasibility |
| M3: G4 | M1/M2 có observations độc lập | T/M predictor + calibration/OOD cards | Không leakage, coverage/error/width được báo |
| M4: G5 | Predictor hữu ích hoặc core scheduler ổn | Cost-aware variant + baselines + ablations | Khóa primary policy trên development |
| M5: G6 local | M1–M4 khóa, máy nhàn, đủ budget | Fresh core reference + heldout domains | Statistics và costs tái tạo được |
| M6: G6 transfer | Có thêm GPU vật lý/tài nguyên | Unseen-GPU/domain validation | Scope phản ánh đúng hardware đã đo |
| M7: G7–G8 release | Correctness + evidence gates qua | CLI/report/package/hồ sơ nghiên cứu | Phân biệt supported, observed và not established |

G7 UI có thể làm sau M2; không trì hoãn ledger/thuật toán để trang trí giao diện. Thứ tự đầu tiên khi triển khai là **G0 → G1 → G3**, với G2 core đủ cho thử nghiệm; chưa cần quality surrogate/Bayesian optimization.

### 12.2. Tính ngân sách trước khi chạy

```text
ReferenceCostEstimate ≈ Σ_domain Σ_config Σ_required_seed E[job_cost_including_retry_policy]
OnlineBudgetCap       = Σ_domain Σ_method Σ_budget Σ_search_repeat B
TotalCostEstimate     ≈ external_shared_preparation + calibration_collection
                      + ReferenceCostEstimate + Σ expected_actual_online_session_cost
                      + outer_evaluation + post_commit_report_export
```

Online session cost đã gồm setup trong workflow, probes, failed attempts, model fitting và full confirmations; không cộng các khoản ấy lần hai. OnlineBudgetCap là tổng ngân sách cấp, không thời gian thực chạy; cleanup overrun được báo riêng, không giấu. Shared preparation/calibration/reference/test không được charge trùng nếu đã thuộc một session; ledger có ownership cho từng event.

Dùng phase measurements mới và khoảng dự báo, không ngoại suy 23 giây của Penn-Fudan 4 epoch thành giờ train cho dataset khác. Ví dụ C_core 12 configs × 3 seeds = 36 full jobs mỗi domain; 24 configs = 72 jobs. Chưa nhân hàng loạt methods/domains khi chưa có ledger cost estimate.

Chạy campaign tuần tự theo gate và budget manifest, không khởi động hàng trăm full jobs mặc định. Lần chạy đầu chỉ development/cost pilot; cân nhắc giảm grid/method/repeats có khai báo thay vì cắt epoch rồi vẫn nói same protocol. Không benchmark tài nguyên đồng thời training của người dùng. Nếu reference quá đắt, hạ claim xuống best observed/limited comparison, không điền regret 0.

Không có ETA cố định trong kế hoạch: horizon/dataset mới và tài nguyên nhiều GPU chưa đo/chưa sẵn có. Trước campaign phải xuất estimates, disk/checkpoint budget, giới hạn tổng compute và điều kiện dừng; lịch kỹ thuật không được trình bày như training time đo thật.

### 12.3. Điểm dừng và phương án thay thế

- Không có GPU thứ hai/thứ ba: hoàn thành local code/calibration/benchmark, giữ transfer claim `not_established`; cần quyền truy cập máy khác mới kiểm định được.
- Dataset thiếu labels/groups/validation: báo thiếu metadata hoặc protocol leakage risk; không chế nhãn/split “chuẩn” để tiếp tục.
- B không đủ một full-confirmed config: trả insufficient budget với estimate, không tự tăng B ngoài giới hạn.
- Predictor không hữu ích: dùng measurements + scheduler/Random, công bố thất bại dự báo.
- Policy mới không tốt hơn Random: không bật mặc định/không sửa test để thắng; nêu kết quả âm và miền phù hợp.
- Có job đang chạy hoặc integrity mismatch: giữ nguyên job/artifacts; triển khai qua version/runtime mới hoặc chờ cửa sổ máy nhàn khi được yêu cầu chạy.

## 13. Trạng thái sau lượt lập kế hoạch

Đã viết kế hoạch nâng cấp này dựa trên findings hiện tại và đối chiếu tài liệu phương pháp gốc. **Chưa thay thuật toán, chưa cài dependencies, chưa tạo benchmark mới, chưa mở thêm test, chưa chạy training.** Tất cả G0–G8 ở trên là công việc tương lai; các số ở mục 2 là lịch sử đã đo, không phải bằng chứng hoàn thành kế hoạch mới.
