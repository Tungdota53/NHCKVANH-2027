# Nhánh nghiên cứu detection và minh chứng lựa chọn cấu hình

Đây là extension0.6.1 trong `research_pipeline`, dùng helper package0.5 bất biến để bảo toàn các job/checkpoint cũ. Entry point thống nhất là `python train_config.py`; console `train-config` cũng đã cài lại trỏ vào dispatcher mới ở cả2venvs. `yolo-tune` giữ nhánh legacy. Không thay dependencies/runtime để làm đẹp thực nghiệm.

## Dùng với dataset detection của bạn

```powershell
.venv\Scripts\python.exe train_config.py research --data "D:/datasets/detection/data.yaml" --task detection --device auto --full-epochs 20 --search-budget 10 --methods default random random_proxy successive_halving adaptive_verified adaptive_no_profile adaptive_no_strata --output runs/research_detection
```

Grid mặc định compact4 configs, expanded16 configs với2models: YOLO11n/s architecture từ runtime đã cài, image320/640, batch2/8, accumulation giữ nominal effective16, LR0.001/0.0003. YAML là random initialization, không giả vờ pretrained; chỉ local `.pt` mới dùng pretrained weights. Có thể cần nhiều hơn10phút mỗi seed/phương pháp; timeout không là thành công. Chưa đo hết grid thì không có optimum certificate. AdamW/SGD explicit; không optimizer auto. Mỗi horizon/split/grid/objective khác là protocol mới.

Model names/precision/image sizes là lựa chọn trong grid đã khai báo, không bảo đảm model lớn chạy trên máy yếu. Mục tiêu feasibility dùng giới hạn peak allocator/RSS đã đo, không tổng VRAM hệ thống. Khuyến nghị budget/grid phù hợp; không suy ra mọi dataset đều đọc được hoặc mọi máy đều đủ RAM/VRAM.

Với task caption/classification/reconstruction, các lệnh `research`/`verify-config` vẫn chuyển cho runner0.5 theo metric riêng. Ablation mới chỉ ở detection. Không ép ViECAP thành detection, không gộp BLEU/chrF với mAP.

## Custom grid đầy đủ

JSON là list overrides các khóa `model, imgsz, batch, precision, accumulation, optimizer, lr0, weight_decay`. Ví dụ cấu hình của thầy là **một ứng viên**, chưa phải khuyến nghị đã xác nhận:

```json
[
  {"model":"yolo11s.yaml", "imgsz":640, "batch":8, "precision":"fp16", "accumulation":2, "optimizer":"AdamW", "lr0":0.001, "weight_decay":0.0005}
]
```

Chỉ có1ứng viên thì optimum trong grid1 gần như định nghĩa, không chứng minh thuật toán tìm kiếm có ích. Cần grid đủ đa dạng, đối chứng và ablation. Checkpoint `.pt` chỉ nên từ nguồn tin cậy; worker không tải model âm thầm. Architecture/weights được snapshot/hash trước search.

```powershell
.venv\Scripts\python.exe train_config.py research --data "D:/datasets/detection/data.yaml" --task detection --grid my_grid.json --objective balanced --weights 0.6 0.25 0.15 --time-scale 3600 --memory-scale 6144 --max-memory 4800 --min-quality 0.3
```

Các con số trên là lựa chọn protocol minh họa, không là số đo/constraint đã chứng minh hợp lý cho mọi GPU/dataset. Phải chốt theo yêu cầu sử dụng và khả năng máy trước khi xem kết quả; mAP0.3 không có ý nghĩa chất lượng ứng dụng phổ quát.

## Production: đo lại tại máy/dataset hiện tại

```powershell
.venv\Scripts\python.exe train_config.py verify-config --data "D:/datasets/detection/data.yaml" --task detection --grid my_grid.json --full-epochs 20 --search-budget 30 --device auto --output runs/verified_detection
```

Mọi profile/full confirmation tính online. Không có independent oracle miễn phí. Report phân biệt `empirical_optimal_in_declared_space`, `fully_trained_best_observed`, `no_verified_feasible_selection`. Test không dùng chọn cấu hình. U và Pareto tính trên validation/full measurements, không lấy proxy làm final quality.

## Máy đang bận: chuẩn bị trước, không benchmark tranh GPU

```powershell
.venv\Scripts\python.exe train_config.py research --data "D:/datasets/detection/data.yaml" --task detection --grid my_grid.json --prepare-only --output runs/prepared_detection
.venv\Scripts\python.exe train_config.py research-start --run "runs/prepared_detection/<run>" --timeout-hours 3
```

Prepare chỉ snapshot/export/protocol, không có certificate. Start kiểm nguồn/runtime/hardware/data và yêu cầu máy không có ML job được nhận diện. `--allow-contention` là smoke override, **luôn** đánh dấu không chứng nhận kể cả sau khi full grid hoàn tất; không dùng nó để né kiểm tra rồi gọi benchmark sạch. Preflight không thay thế máy cô lập/kiểm soát nhiệt/công suất/tác vụ ngoài Python.

## Xem, audit, phân tích và resume

```powershell
.venv\Scripts\python.exe train_config.py research-show --run "runs/research_detection/<run>"
.venv\Scripts\python.exe train_config.py research-audit --run "runs/research_detection/<run>"
.venv\Scripts\python.exe train_config.py research-analysis --run "runs/research_detection/<run>"
.venv\Scripts\python.exe train_config.py research-resume --run "runs/research_detection/<run>" --timeout-hours 3
```

Resume chỉ missing/failed full confirmations sau decisions lock, cùng initialization/horizon, không replay search. Có thêm budget/active cost riêng; giữ previous attempts. Live parent/orphan worker hoặc không kiểm tra được process đều chặn duplicate. Thay source/runtime/hardware/input bytes phải dùng protocol mới hoặc phục hồi đúng snapshot; không sửa checksum lịch sử.

Analysis không đổi primary selection. Detection weight scenarios đã khóa trong protocol; legacy runs chưa khóa scenarios thì analysis ghi **post-hoc exploratory**, không giả vờ preregistered. Report có tất cả configs/seed coverage/failures, Q/T/M, U/Pareto, runner-up margin, prediction vs measurement khi có số đo; prediction là heuristic nếu chưa calibration. Không tự điền mAP0.XXX/5.4GB/X giờ.

`research` selected.json xuất từ **full-reference offline**, không phải lựa chọn online miễn phí trong budget B. Muốn biết từng search policy chọn gì, đọc `decisions.locked.json`. Report comparison dùng quyết định đã khóa này, không lấy selected oracle làm kết quả của heuristic.

## Training và test sau lựa chọn

```powershell
.venv\Scripts\python.exe train_config.py train --selection "runs/research_detection/<run>/selected.json" --device auto --output runs/production_detection
.venv\Scripts\python.exe train_config.py evaluate-detection --run "runs/research_detection/<run>" --seed 17
```

Đổi epoch/device/data/source/runtime làm optimality scope bị hạ hoặc từ chối. Không continuation checkpoint trong fixed-horizon detection research vì scheduler/cost phải nhất quán. Test đánh giá checkpoint best của config đã chọn, seed đã khai báo; không chọn lại config/weights theo test. Test exposure ghi trước worker, kể cả failure; rerun phải explicit `--allow-retest` và công bố exposure. Không tuyên bố test luôn untouched nếu nó từng bị engineering xem trước.

## Dataset thực nghiệm hiện tại

Penn-Fudan từ [nguồn chính thức](https://www.cis.upenn.edu/~jshi/ped_html/),170 ảnh/345pedestrians. Archive tải rõ URL, SHA256 `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`. Script `scripts/prepare_detection_research.py` lấy bounding boxes từ instance masks gốc, không gán nhãn detection giả từ ảnh classification. Chuẩn bị130train/20val/20test, seed101; groupingcampus×blocks10filename giữ các ảnh liên tiếp cùng split. Đây là heuristic chống overlap views, **không có ground-truth scene ID** để bảo đảm không còn near-duplicate leakage. Exact-content duplicates bị chặn.

YOLO11n/s pretrained COCO weights tải từ [Ultralytics official assets release](https://github.com/ultralytics/assets/releases/tag/v8.3.0), dùng checksum bytes trong từng protocol. Không gọi training từ random initialization khi đã dùng weights pretrained; pretraining có thể ảnh hưởng generalization, cần công bố rõ nguồn.

Pilot `configs/detection_research_pilot.json` chỉ3config,4epoch,3seed,60s/search/seed/method. Các ứng viên khác nhiều yếu tố cùng lúc; không suy ra causal effect của LR hay batch riêng từ ranking. Ablation đánh giá scheduler components, không đánh giá causal hyperparameter effect. Đây là thực nghiệm nhỏ, không SOTA/full convergence20epoch hoặc bảo đảm universal portability.

## Báo cáo nguồn bằng chứng

Đọc [RESEARCH_PLAN.md](RESEARCH_PLAN.md) cho tiến độ từng bước, [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) cho công thức/tính chất/pseudocode, và [DETECTION_VALIDATION.md](DETECTION_VALIDATION.md) cho số đo thật sau khi hoàn tất. Các baseline/Hyperband references là cơ sở so sánh, không là bằng chứng thuật toán ghép của mình có superiority hay regret bound.
