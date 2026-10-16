# Tiến độ theo kế hoạch nâng cấp

Thiết kế gốc: [RESEARCH_UPGRADE_PLAN.md](RESEARCH_UPGRADE_PLAN.md). Hướng dẫn sử dụng: [PIPELINE_V2.md](PIPELINE_V2.md). Kết quả có artifact/raw audit: [UPGRADE_VALIDATION.md](UPGRADE_VALIDATION.md).

| Nhóm | Trạng thái thực tế | Còn thiếu / phạm vi |
|---|---|---|
| G0 protocol/data | Đã triển khai schema4, hash/IDs, task metrics, normalized manifests/external YAML/COCO, declared-group checks | Không tự suy true scene IDs; near-duplicate/pretraining overlap chưa loại trừ; final independent preregistration chưa có |
| G1 costs/integrity | Đã triển khai shared clocks, ledger, phase counters, raw/committed-state audit, integrity-first resume/reconcile, post-commit error separation | Hash không attestation; hard-crash unknown cost phải dừng điều tra; sampled peaks/contended environment có giới hạn |
| G2 candidate space | Core12, custom<=512, controlled detection6, canonical model content IDs, unsupported/inactive fields reject | Không enumerate full512 mặc định; class optimizer/decay và caption encoder giới hạn; workers/cache fixed |
| G3 reserve/gating | 0.7.0 paired CPU benchmark đã đo; 0.7.1 startup-aware initial/targeted probes và GPU full-space verification đã qua | Forecast margin không upper bound; không superiority claim cho policy mới từ pilot cũ |
| G4 predictors | Point T/M fitter, disjoint partitions, source/runtime/hardware/range checks; Fashion36fit/CIFAR18calibration/MNIST18heldout diagnostics đã hoàn tất | Chưa validated safety intervals, coverage hoặc unseen GPU; 72rows chỉ3domains, không statistical generalization proof |
| G5 advanced search | Random/default/profile-all/no-reserve và one-rung quality/utility-SH có code | Chưa multi-rung matched study, quality surrogate, cost-aware EI/VOI, BOHB/NEHVI; không gọi implementation hiện tại là Bayesian optimizer |
| G6 evidence | CPU core12×3seeds reference36; GPU detectionC6×3seeds×8epoch online18 + audit/test | Các datasets lịch sử là development; chưa final blind statistically powered benchmark hoặc nhiều GPU vật lý |
| G7 CLI/portability | Root/console CLI, frozen plan start, inspect/status/audit/resume, deploy/eval, Markdown/JSON reports, installer không overwrite | Clean-install máy khác/MPS/ROCm chưa kiểm chứng; relocatable historical bundles chưa hoàn tất |
| G8 tests/release |162tests CUDA, CPU1CUDA-specific skip; dependency/compile checks qua | Không thay real multi-GPU/failure/convergence evidence bằng tests; full checkpoint-continuation không hỗ trợ |

Các mốc quan trọng đã thực hiện: metadata tuple/list bug và Windows helper encoding bug được giữ log/khác biệt source, sửa bằng regression tests; strict CUDA caption pooling lỗi được thay bằng toán tương đương và kiểm output/gradient. Source/dataset/checkpoint lịch sử giữ nguyên, không sửa findings để tạo advantage.

Ở GPU production: lần đầu chưa đủ C nên best observed; additional budget900s được ghi và chỉ jobs thiếu chạy. Sau18/18full outcomes mới empirical finite optimum. Việc dự báo bảo thủ bỏ config dù budget thực tế có thể đủ vẫn là một giới hạn đã quan sát; không coi skip là infeasibility proof.

Không đánh dấu toàn bộ đề tài hoàn thành khi chưa có hardware/final-domain/statistical evidence. Phần mềm lõi đã chạy thật và có scientific finite-space findings; các claims còn thiếu phải được kiểm định trong giai đoạn tiếp theo, không hứa tối ưu mọi máy/dataset.
