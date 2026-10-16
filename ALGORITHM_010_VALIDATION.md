# Nghiệm thu thuật toán 0.10.0 — 2026-10-16

## Đã triển khai và khóa source

- `adaptive_full`: online kernel-ridge acquisition trên full all-seed outcomes; cost/uncertainty/novelty; partial confirmation priority; timeout lowerbounds; không subset-quality pruning.
- Ba ablation cùng coordinator; weighted-only guard; tham số acquisition khóa trong protocol; trace decision trong events/state/report và kiểm tra khi audit report.
- `algorithm-report`: raw-audited matched studies, independent full reference boundary, equal budgets, kiểm tra budget nhỏ hơn measured exhaustive cost, paired repeats/conditional CI/no-selection, phân biệt dataset và GPU vật lý.
- Code/test/mock fixtures không được tính thành bằng chứng ưu thế khoa học.

## Kiểm thử và đóng gói

Sau source freeze: CUDA **329 tests pass**; CPU **328 pass + 1 CUDA-only skip** (329 tổng). Có 17 acquisition tests, 7 safety tests và 22 study analysis tests mới. Chạy lại toàn bộ sau khi tất cả ML jobs kết thúc cũng đạt: CUDA 329/329 (16.396s), CPU 328 pass + 1 skip (13.858s). Đây là kết quả kiểm thử phần mềm, không phải số lần training hay bằng chứng superiority.

Hai môi trường editable đã cập nhật package0.10.0 bằng uv `--no-deps --no-build-isolation`, không nâng ML libraries. CPU Torch2.14.1+cpu; CUDA Torch2.6.0+cu124/torchvision0.21.0+cu124. NumPy được khai báo trực tiếp `>=1.26,<3`.

Wheel `runs/algorithm_010_packaging/wheels/yolo_budget_tuner-0.10.0-py3-none-any.whl`, SHA256 `bb848e501be08bd22179aa464a5b46bd07be88ebaa9265a58e6f3aaa91f26306`. Cài target mới, isolated import từ ngoài checkout; 59 packaged source hashes trùng frozen workspace. Module version/help/CPU doctor thành công, không cần root launcher. Chi tiết `runs/algorithm_010_packaging/packaging_verification.md`. Không chứng nhận clean OS hoặc mọi dependency/platform mới.

## Pilot B60 không đủ nguồn lực

Giữ `runs/algorithm_010_mnist/20261016_150413_175270` nguyên trạng. Hai children đầu complete nhưng không có full outcomes; child thứ ba có các timeout và trạng thái đã lưu vẫn là `phase="search"` khi dừng study. Đây là trạng thái lưu dở dang, không xác nhận process còn sống; kiểm tra cuối không còn worker ML đang chạy. Raw audit của pilot đạt không có nghĩa benchmark hoàn tất. Không có reference/benchmark complete; không hợp thức hóa thành matched result.

Observed prep27.609/23.828s; finalization reserve13.048/24.218s, actual finalize6.516/7.625s. Cachehit preflight~3s với1860files, tức vẫn có metadata traversal cost; total hashed bytes nhỏ không có nghĩa overhead kiểm chứng nhỏ. Diagnostic import TorchCPU8.5s. Không tắt integrity hoặc giảm horizon/seeds để giả thành công.

Revision resource B180/R1/C12 đã khóa trước khi có quality/reference; xem `ALGORITHM_010_EXPERIMENTS.md`. Đây là adaptation theo chi phí nền, không tune acquisition từ reference. R1 không đủ CI/superiority; không báo R3 original như đã chạy xong.

## Training/benchmark thực tế

Các operational studies chạy tuần tự, không concurrent ML workers. Hai matched studies và hai integration runs dưới đây đã hoàn tất; pilot B60 riêng vẫn được ghi là incomplete. Không mở test.

### MNIST CPU — online decisions và independent reference hoàn tất

Study `runs/algorithm_010_mnist_operational/20261016_151018_583510`, C12/E6/S2/R1/B180. Tất cả8 online children hoàn tất, zero overrun. Reference initial900s có21/24full, cấp thêm300s riêng, sau đó đủ24/24full outcomes/25attempts (1timeout). Các online decisions và budgets không đổi; `benchmark-report` audit/dựng lại kết quả thành công sau khi reference hoàn tất.

| Method | Full configs/12 | Online(s) | Selected width/LR | Val accuracy mean |
|---|---:|---:|---|---:|
| adaptive_full | 1 | 116.172 | 16 / 0.003 | 0.716667 |
| adaptive_no_cost | 1 | 108.344 | 16 / 0.003 | 0.716667 |
| adaptive_no_uncertainty | 1 | 94.875 | 16 / 0.003 | 0.716667 |
| adaptive_no_diversity | 2 | 158.485 | 16 / 0.003 | 0.716667 |
| random | 1 | 175.016 | 32 / 0.003 | 0.563333 |
| reserve | 1 | 126.188 | 16 / 0.003 | 0.716667 |
| successive_halving | 1 | 109.344 | 16 / 0.003 | 0.716667 |
| default | 1 | 111.078 | 16 / 0.001 | 0.666667 |

**Mọi adaptive variant có0learned-fit acquisition steps trong study này.** Phương pháp chính chỉ bootstrap một config đủseeds; no_diversity có thêm novelty/cost fallback trước khi đủ2config, không có action thứ3 dùng kernel đãfit. Vì vậy bảng trên không chứng minh quality surrogate hoặc ablations hiệu quả; chênh thời gian chịu startup/runtimevariation. Hạn chế chi phí thận trọng làm coverage thấp được giữ nguyên, không sửa source giữa study.

Random có partial config thứ2: seed17 complete nhưng seed29 timeout; không được dùng chọnconfig. Reserve probe timeout14.860s được tính chi phí. SH bootstrap/fallback, không có rung thực sự để tuyên bố earlystoppingwin. Sốquality trên là validation, không phải test hoặc application accuracy guarantee.

Reference cumulative online cost **1043.235s** (worker attempts900.466s), tổng grants1200s, không overrun. **B180 < 1043.235s**: đây thực sự là budget-limited study, không phải quét hết grid trong online budget. Reference chọn width16/LR0.003 theo cân bằng U; Accuracy preset riêng là width32/LR0.003 Q=0.766667, không phải cùng mục tiêu.

Conditional regret trên **cùng reference**: adaptive_full và cả3ablations/reserve/SH đều0; Random **0.336102**; Default **0.050955**. Một repeat, không có statistical CI/superiority. Adaptive/regret0 ở đây do bootstrap chọn đúng config reference, không chứng minh kernel quality hoặc acquisition học hiệu quả; không gán chênh runtime giữa các variants thành causal speedup.

### CIFAR-10 CUDA — hoàn tất matched study

Study `runs/algorithm_010_cifar_gpu/20261016_154627_339104`, cùng C12/E6/S2/R1/B180, FP32, RTX3050. Hai online methods khóa trước reference riêng, không dùng prior/quality từ MNIST. Adaptive chỉ full1config/12 và0fittedsteps; Random full1config và1timeout (dù epoch6đã ghi, worker chưa hoàn tất bằng chứng đúng deadline nên vẫn timeout).

| Method | Selected | Val Q mean | Online(s) | Conditional regret | Overrun |
|---|---|---:|---:|---:|---:|
| adaptive_full | width16/img32/LR0.003 | 0.245000 | 126.203 | 0.111208 | 0 |
| random | width32/img48/LR0.003 | 0.250000 | 172.219 | 0.170715 | 0 |

Reference initial900s đạt19/24full; thêm600s riêng, cuối đủ24/24full/25attempts. Cumulative reference cost **1084.265s**, tổnggrants1500s, khôngoverrun. B180nhỏhơn1084.265thật. Reference Balanced chọn width16/img32/LR0.0001, Q=0.218333, Tmean29.4s, M68MiB; Accuracy preset riêng Q=0.321667, width32/LR0.0003. Với objective đã khóa, tốc độ quan sát có thể khiến Balanced chọn quality thấp hơn.

**Cả hai online methods không tìm đúng reference winner.** R1và worker startup/time biến thiên lớn: không suy từ regret chênh lệch thành superiority hoặc thời gian kỳ vọng ổn định. Full E6 classifier này không có application-quality claim. GPU allocator reserved không phải totaldeviceVRAM.

### Tổng hợp đúng phạm vi

Manifest `configs/algorithm_010_studies.json`, output `runs/algorithm_010_evidence_20261016/ALGORITHM_STUDY.json` và `.md`. `algorithm-report` raw-audit/matched-protocol checks thành công, không training và không sửa studies. Required methods chỉ hai method chung adaptive/random; các baseline thiếu ở CIFAR được ghi riêng.

Hai dataset content identities, một CPU host và một physical GPU trên cùng laptop; không gộp R1×S2×2domains thành nhiều independent GPU/search repeats. Bootstrap CI không xuất vì thiếu5searchrepeats; `statistical_superiority_established=false`, `unseen_hardware_generalization_established=false`.

### Nhánh learned acquisition đã thực thi thật trên GPU

Run `runs/algorithm_010_fit_integration/20261016_185012_023962`: genuine CIFAR-10 full train1000/val300, C4LR/E6/S2/B600/FP32; **8/8 full workers hoàn tất**, không probe. Acquisition trace:

| Step | Config index / LR | Fitted | Số full configs dùng fit |
|---|---|---|---:|
|1|0 / 0.001|bootstrap|0|
|2|1 / 0.0003|false — novelty/cost fallback|1|
|3|3 / 0.003|true|2|
|4|2 / 0.002|true|3|

Đạt điều kiện **2 fitted receipts + 8/8 full outcomes**. Tổngonline394.078s, workers337.610s, budget600s, overrun0. Chọn LR0.002, width16/img32/batch16/accum2/AdamW; Qvalidation0.278333±SD0.021213, workerTmean29.6015s, CUDAreserved68MiB. Đây là empirical argmax trong đúng C4/S2/E6 đã đo đủ, không phải optimum kỳ vọng hoặc quality production.

Lượt này chứng minh module fit/rank/reacquire thật sự điều khiển các lượt full training, không chỉ unit mocks. Nó **không là efficiency benchmark**: budget đủ đo hết grid nhỏ, không có pairedbaseline/reference độc lập, grid khác C12; không dùng nó để tuyên bố giảm regret hay cải thiện accuracy so với studies bên trên. Không chuyển quality của lượt này sang bất cứ online run nào khác.

### YOLO detection integration — C3/E3/S2 full thật

Run `runs/algorithm_010_detection_integration/20261016_185734_558578`: Penn-Fudan130train/20val/20test (testđóng), local pretrained weights; FP16/img160/batch4/accum4/AdamW/LR0.001/decay0.0005. **6/6 full workers hoàn tất** và actual-config/checkpoint verified. Tổngonline379.203s, workers347.984s, grant600s, overrun0.

| Config | Freeze | Val mAP50-95 mean | SD | WorkerTmean(s) | CUDAreservedMiB |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0 | 0.525065 | 0.017770 | 55.2110 | 222 |
| YOLO11n | 5 | 0.531710 | 0.094059 | 89.8435 | 218 |
| YOLO11s | 0 | 0.456595 | 0.031784 | 28.9375 | 546 |

Trace index1bootstrap → index0fallback → index2**fitted=true,2fullconfigs**. Chứng minh mixed-model/freeze acquisition cũng chạy thật qua detection adapter, không chỉ classification. Balanced theo weights/scales đã khóa chọn YOLO11s; Accuracy/Memory preset chọn YOLO11n/freeze5. Tất cả nằm trên observed Pareto front.

Không diễn giải YOLO11s có compute nhanh hơn YOLO11n nói chung: các workers có startup/hosttime biến thiên mạnh, cold-worker T là counter đãkhóa. R1/S2/E3 chỉ là integration và finite observedargmax; không production/convergence/causal speedup hoặc unseenGPUproof. Không so mAP khác horizon với0.9.1để tuyên bố tăngchấtlượng.

## Audit nghiệm thu cuối

- **17 run roots mới / 93 attempts** đều raw-audit đạt: **85 full workers hoàn tất hợp lệ, 7 full timeouts và 1 probe timeout**. Số này có bao gồm ba child roots của pilot dở dang; không phải 17 completed benchmarks. Một timeout/partial cũng có `actual_config_verified=true`, nên không dùng tổng 86 actual-config receipts để thay cho 85 full outcomes.
- Cả 17 protocol có **60 source hashes khớp source cuối**. Chỉ tài liệu được hoàn thiện sau source freeze; không sửa thuật toán giữa study và reference. Wheel có 59 packaged source hashes (không đóng gói root launcher), đều khớp workspace.
- Audit hồi quy **7 run roots lịch sử 0.9.1 / 69 attempts** đạt; **19/19 file `yolo_tuner`** giữ nguyên SHA-256 so với protocol lịch sử. Không sửa checkpoint, nhãn hay số đo cũ để đạt audit.
- Learned-fit integration xác minh 8/8 full workers dùng FP32, batch16, accumulation2/effective32, shape32px và loader/cache/pinning thực tế. Detection xác minh 6/6 full workers dùng FP16, batch4, accumulation4/effective16, freeze0/5; freeze5 được kiểm tra qua `requires_grad`. YOLO cache tắt rõ ràng, không giả là đã bật.
- Phân tích lại manifest không ghi output cho kết quả khớp chính xác báo cáo tổng hợp đã lưu, digest `d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5`. Không có `test_history.json` trong năm nhóm run mới; các phép chọn và báo cáo trên chỉ dùng validation, test vẫn đóng. Kiểm tra process cuối không còn worker ML đã biết đang chạy.

## Giới hạn phải giữ

Chỉ một host/RTX3050 vật lý, dataset local từng tham gia development. Full E6 là fixed horizon, không chứng minh hội tụ; S2 không đại diện phân phối seed tương lai. Kernel uncertainty chưa hiệu chuẩn, không seed-level CI hoặc upper bound; unknown resource forecasts không chứng minh feasibility. Chi phí quá bảo thủ vẫn có thể bỏ lỡ cấu hình chạy vừa. Không có universal-optimum hoặc statistical-superiority claim.
