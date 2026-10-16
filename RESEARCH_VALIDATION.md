# Kiểm chứng phương pháp chọn cấu hình huấn luyện

Ngày thực nghiệm: 12/10/2026. Đây là nhánh nghiên cứu mới; số liệu caption v2 trong PIPELINE_VALIDATION.md là kiểm chứng model/pipeline vận hành, không được dùng thay cho exhaustive reference ở đây.

## Trạng thái

Đã hoàn tất **24/24 full-reference trainings**: ViECAP CUDA và CIFAR-10 CPU, mỗi dataset 4 cấu hình × 3 seed × đúng 8 epoch trên toàn bộ train/val của dataset nghiên cứu. Audit đối chiếu đủ 24 request/result gốc, protocol và decisions. Không dùng số liệu mô phỏng hoặc trial tìm kiếm làm reference độc lập.

## Thiết kế đã khóa

- ViECAP captioning, RTX 3050 Laptop 6 GB, torch 2.6.0+cu124.
- Train/val/test: 2.773/944/331 ảnh; train/val: 13.858/4.714 caption.
- Encoder ResNet18 pretrained frozen, spatial features 7×7, attention GRU, ảnh 224, effective batch 64, FP16, AdamW, dropout 0.3, label smoothing 0.05.
- c00 hidden256/batch32/LR0.001; c01 hidden384/batch32/LR0.001; c02 hidden256/batch32/LR0.0003; c03 hidden128/batch16/LR0.001.
- Full horizon 8 epoch, early stopping tắt, training seeds 17/29/43, cùng dữ liệu/initialization policy.
- Balanced weights 0.6/0.25/0.15, time anchor 600s, memory anchor 1024 MiB, memory cap 4117.6 MiB, min Q 0.
- Default, Random full-fidelity, Successive Halving, Pareto racing. Cap 120s/phương pháp/seed; common feature preparation và offline exhaustive reference tính riêng.
- Search decisions đã khóa trước reference. Mỗi cấu hình được đo trên 3 seed, tổng 12 full trainings, không dùng test để chọn.

Study: `runs/research_viecap_fixed/20261012_210220_854944`. Protocol SHA256: `c444a93f5a0934899bf505f5297c4a5dd9f2b052fc87a971fd9e6a8a679f15ae`.

Decisions SHA256: `cc83ae0098b833d0ed3873f76de0eebafa53fb290b28ef7f06b87b60438fd731`. Các tiện ích CLI/audit/report được hoàn thiện trong khi benchmark chạy; không đổi worker/model/optimizer/metric/fidelity của study đang chạy. Protocol ViECAP này chưa có trường checksum source; các study khởi động bằng bản mã 0.4 hoàn thiện sẽ lưu source snapshot/checksum và phiên bản runtime chi tiết.

Pilot `runs/research_viecap/20261012_205208_068718` đã dừng và giữ nguyên artifact: domain batch ban đầu phụ thuộc transient RAM, không dùng làm kết quả nghiên cứu chính.

## Kết quả ViECAP đầy đủ

| Cấu hình | Q mean ± SD | Full time mean ± SD (s) | Peak allocator max MiB |
|---|---:|---:|---:|
| c00: hidden256 / batch32 / LR0.001 | 0.463397 ± 0.008551 | 156.67 ± 14.69 | 196 |
| c01: hidden384 / batch32 / LR0.001 | 0.459770 ± 0.007585 | 159.61 ± 18.79 | 228 |
| c02: hidden256 / batch32 / LR0.0003 | 0.482271 ± 0.001998 | 166.93 ± 43.53 | 196 |
| c03: hidden128 / batch16 / LR0.001 | 0.479469 ± 0.002110 | 244.59 ± 31.18 | 144 |

**Nghiệm balanced thực nghiệm: c02**, cũng là Q trung bình cao nhất trong C này. Validation checkpoint tốt nhất mỗi seed: BLEU-4 **42.0793 ± 0.1312**, chrF **54.3749 ± 0.2710**. Chọn checkpoint bằng validation; không tính test mới hoặc dùng test để chọn. Q = (BLEU4+chrF)/200, không phải accuracy phần trăm. Thời gian là đủ 8 epoch, không chỉ tới best checkpoint. Pareto: c00/c02/c03; c01 bị c00 trội theo các số đo trung bình/max đã khai báo.

| Phương pháp | Feasible | Balanced regret mean | Search seconds mean | Recovered HV |
|---|---:|---:|---:|---:|
| Default | 100% | 0.007049 | 0 | 0.915852 |
| Random full-fidelity | 0% | không xác định | 120.32 | 0 |
| Successive Halving | 100% | 0.015137 | 71.43 | 0.867777 |
| Pareto racing | 100% | 0.009745 | 98.81 | 0.940243 |

Các quyết định online không chọn được c02: SH chọn c01 cả 3 seed; racing chọn c01 một lần, c00 hai lần. **Racing chưa vượt default về regret balanced**, dù recovered HV cao hơn. Random không đủ 120s để hoàn tất 8 epoch trên full data; không gọi đây là bằng chứng superiority phổ quát của proxy search.

Sai số proxy so với full cùng seed: quality MAE SH/racing **0.221607 / 0.211557** trên thang Q 0–1; time MAPE **27.33% / 24.33%**; memory relative error 0 trên các trial hoàn tất này. Memory khớp không chứng minh proxy quality đáng tin. Learning rate thấp c02 bị loại sớm trong search nhưng đạt Q full tốt nhất: **learning-curve/subset ranking có thể đảo**, không thể chứng nhận optimum từ vài epoch proxy.

Common setup **78.20s**, exhaustive reference **2183.42s (36.39 phút)**, toàn study **3134.91s (52.25 phút)**. `selected.json` dùng nghiệm từ exhaustive reference, không phải nghiệm online trong budget 120s. CIFAR download/preparation có chạy nền trong một phần lượt GPU; không có CPU training chạy đồng thời, nhưng các timing laptop này vẫn là observational measurements, không phải controlled latency guarantees. Biến thiên time của c02 lớn và 3 seed không đủ để bảo đảm ranking balanced ổn định trong các lần chạy tương lai.

Artifacts:

- `runs/research_viecap_fixed/20261012_210220_854944/selected.json`
- `runs/research_viecap_fixed/20261012_210220_854944/RESEARCH_AUDITED.md`
- `runs/research_viecap_fixed/20261012_210220_854944/study.audited.json`
- `runs/research_viecap_fixed/20261012_210220_854944/decisions.locked.json`
- `runs/research_viecap_fixed/20261012_210220_854944/reference/`: checkpoint best/last, request, result và log từng config/seed.

```powershell
.venv\Scripts\python.exe train_config.py research-show --run "runs/research_viecap_fixed/20261012_210220_854944"
.venv\Scripts\python.exe train_config.py train --selection "runs/research_viecap_fixed/20261012_210220_854944/selected.json" --device auto
```

Đã có checkpoint dùng được, không cần train lại để đọc report. Ví dụ best checkpoint c02 seed17: `runs/research_viecap_fixed/20261012_210220_854944/reference/c02/seed_17/trial_0001/best.pt`.

## CIFAR-10 CPU: đối chứng task và runtime thứ hai

Torchvision tải và kiểm tra archive CIFAR-10 chính thức. `scripts/prepare_cifar_research.py` giữ nhãn gốc, lấy mẫu phân tầng seed17: **1000 train / 300 val / 500 test**, 10 lớp. Train/val từ original train, không giao nhau; test từ official test. `PROVENANCE.json` lưu chỉ số nguồn. Đây là **subset CIFAR-10**, không phải full 50.000 ảnh train hoặc thực nghiệm SOTA.

CPU cùng laptop vật lý, torch **2.14.1+cpu**, torchvision **0.29.1+cpu**, Python3.12.14, 4 compute threads, data-loader workers0, FP32. Không chạy đồng thời với GPU training. Hai backend/task không thay nhiều máy/GPU độc lập; không gộp trực tiếp Q giữa task hoặc CUDA allocator/RSS CPU để suy ra chuyển miền.

Grid: CNN ảnh32, batch16, effective batch32; width16/32 × LR0.001/0.0003, AdamW, fixed8 epoch, seeds17/29/43. Cùng objective/anchors 0.6/0.25/0.15 và 600s/1024MiB; memory cap ghi trong protocol. Thêm đối chứng `random_proxy` cùng subset/rung4. Decisions khóa trước reference; source snapshot/checksum và runtime versions lưu trước search.

| Cấu hình | Val accuracy mean ± SD | Full time mean ± SD (s) | Peak RSS max MiB |
|---|---:|---:|---:|
| c00: width16 / LR0.001 | 27.00% ± 2.52 điểm % | 10.95 ± 1.73 | 290.3 |
| c01: width32 / LR0.001 | 34.00% ± 2.31 điểm % | 14.26 ± 2.63 | 295.1 |
| c02: width16 / LR0.0003 | 24.56% ± 2.41 điểm % | 15.05 ± 3.74 | 290.1 |
| c03: width32 / LR0.0003 | 30.89% ± 1.39 điểm % | 14.29 ± 3.84 | 294.9 |

**Nghiệm balanced thực nghiệm: c01**. Đây là validation accuracy epoch8, không phải test accuracy; không dùng test để chọn. Model còn yếu, study kiểm chứng phương pháp chứ không chứng nhận classifier chất lượng cao. Pareto strict dùng đúng số đo; khác biệt RSS rất nhỏ có thể khiến membership nhạy nhiễu, không nhất thiết là lợi thế bộ nhớ thực tế.

| Phương pháp | Feasible | Balanced regret mean | Search mean (s) | Recovered HV |
|---|---:|---:|---:|---:|
| Default | 100% | 0.039920 | 0 | 0.795956 |
| Random full-fidelity | 100% | 0 | 65.50 | 1.000000 |
| Random cùng subset/fidelity | 100% | 0.012434 | 22.09 | 0.969736 |
| Successive Halving | 100% | 0.032065 | 41.95 | 0.835052 |
| Pareto racing | 100% | 0.006217 | 85.79 | 0.964900 |

Random full chọn c01 cả3 seed; racing c01 hai lần/c03 một lần; random_proxy c01 một lần/c03 hai lần; SH chọn c02/c01/c00. Racing tốt hơn default/SH về regret ở bộ này, **nhưng không vượt Random full** và tốn nhiều search time hơn. Khi train đủ toàn grid trong budget, phương pháp đơn giản full-fidelity có thể tốt hơn racing. Khi cap không đủ một full trial như ViECAP, proxy chưa chứng nhận được nghiệm cuối.

Time MAPE random_proxy/SH/racing: **23.51% / 44.15% / 48.93%**; memory relative error **1.93% / 1.96% / 1.96%**; quality proxy MAE **0.100356 / 0.147366 / 0.142530**. Không suy ra calibrated coverage từ chỉ hai dataset.

Common setup **11.31s**, full-reference **163.64s**, toàn study **821.17s (13.69 phút)**. Download/preparation CIFAR là bước riêng, không nằm trong cap120s. Chi phí offline exhaustive không phải online miễn phí.

Study: `runs/research_cifar_cpu/20261012_215515_139042`. Protocol SHA256: `f6279bfece85f421c9d1301912c754dd7596b8a8fd7c414fed00592449b30321`; decisions SHA256: `31b5baeac6c6a5463ca83f9eeed43a13fa7c79f8788a5089a134a3db443b0503`.

Artifacts: `selected.json`, `RESEARCH_AUDITED.md`, `study.audited.json`, `source_snapshot/`, `reference/` trong study trên. CLI:

```powershell
.venv-cpu\Scripts\python.exe train_config.py research-show --run "runs/research_cifar_cpu/20261012_215515_139042"
```

## Kiểm thử và thực thi bản 0.4

- **78/78 tests đạt trong cả `.venv` CUDA và `.venv-cpu`**, gồm17 tests nghiên cứu mới: objective cố định, worst-seed feasibility, regret/full outcomes, exact3D HV, proxy error cùng seed, search không tham khảo oracle, equal-fidelity interrupted rung, random full/proxy riêng, khóa decisions trước reference, source/ảnh thay đổi, resume chỉ reference thiếu và giữ failed cost, chuyển horizon/runtime mất optimality scope. Unit mocks chỉ kiểm tra logic; kết quả nghiên cứu ở trên là training thật.
- `research-audit` đối chiếu24 full requests/results gốc và hashes, không training mới; report audit riêng, giữ original reports/decisions.
- **Deployment CLI chạy thật** với c01 CIFAR và `train --epochs 1`: đủ1000train/300val, hoàn tất1epoch, val accuracy17%. Không dùng lượt này làm reference8epoch. CLI cảnh báo horizon/config đổi; deployment selection có status `modified_config_optimality_not_verified`. Run: `runs/research_deployment_check/20261012_221341_949030/training_report.json`.
- Console `train-config.exe research-show` hiển thị đủ12/12 ViECAP, không khởi động training mới.
- Package **0.4.0** đã cài editable vào hai venv; source/installed version khớp. `uv pip check` cả hai báo48 packages compatible; compileall đạt. Không đổi torch/torchvision khi nâng package.
- Giữ nguyên các run/checkpoint cũ, loại pilot khỏi kết luận; không reset/xóa dataset hoặc kết quả người dùng.

## Phạm vi kết luận

Nghiên cứu tìm nghiệm tối ưu thực nghiệm của objective đã khóa trong đúng không gian cấu hình đã đo, không bảo đảm tối ưu toàn cục hoặc mọi dataset/máy. Random bị timeout dưới cap nhỏ không chứng minh ưu thế phổ quát của racing. Nghiên cứu chỉ tối ưu metric, không chứng nhận chất lượng caption theo con người. Test ViECAP đã được xem trong các lượt engineering trước, không gọi là untouched outer test.

Xem RESEARCH_METHOD.md cho công thức, cách chạy, audit và resume. Checkpoint, raw requests/results, logs và decisions được giữ để đối chiếu.
