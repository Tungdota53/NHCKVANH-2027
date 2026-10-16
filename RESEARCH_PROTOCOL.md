# Phương pháp và protocol thực nghiệm

## Mục tiêu, không gian và tính khả thi

Đầu vào D/H/P/C/B: dataset, hardware, split/horizon/metric/seed, tập cấu hình và ngân sách. Detection dùng validation mAP50-95 của checkpoint tốt nhất trong E epoch; vẫn đo toàn bộ chi phí E epoch. Caption/classification dùng metric riêng, không gộp trực tiếp với mAP. Test không dùng để chọn cấu hình/checkpoint hoặc điều chỉnh weights.

U(c) = wA*mean(A(c)) - wT*mean(T(c))/Tref - wM*max(M(c))/Mref.

T là wall time worker gồm startup/train/validation/checkpoint. M là peak PyTorch reserved allocator CUDA, hoặc process RSS CPU/MPS; CUDA allocator không phải toàn bộ VRAM hệ thống. Trọng số không âm được chuẩn hóa, anchors dương, tất cả khóa trước search. Ví dụ0.6/0.25/0.15 là ưu tiên sử dụng, không là hằng số khách quan. Báo sensitivity scenarios riêng, không thay nghiệm chính sau khi thấy oracle.

Feasible khi full-data đủ E epoch, đủ mọi seed, không OOM/timeout, actual config đúng yêu cầu và đạt giới hạn chất lượng/thời gian/bộ nhớ đã khai báo. Lỗi/OOM/timeout/chưa đo không đồng nghĩa chất lượng thấp. Nếu chưa đo hết C, chỉ best observed đủ seed, không chứng nhận toàn bộ tập.

## Tính chất toán học và giới hạn

Với wA,wT,wM>0 và các anchors>0: nếu d có meanA>=c, meanT<=c, maxM<=c và ít nhất một bất đẳng thức nghiêm thì U(d)>U(c). Vì thế argmax U trong tập khả thi nằm trên Pareto front theo đúng ba đại lượng trên. Khi có weight0 chỉ suy ra Pareto theo các trục có weight dương, có thể có tie trên trục bỏ qua; không khẳng định strict Pareto phổ quát.

Exhaustive full measurements của tập C hữu hạn cho phép tính argmax thực nghiệm của U đã khóa. Không có chứng minh global optimum ngoài C, regret bound của heuristic, hoặc feasibility trên mọi seed/máy tương lai. Mean/SD/min/max qua3seed không phải confidence interval đã hiệu chỉnh.

## Pseudocode của adaptive_verified

1. Khóa protocol, hash nguồn/runtime/data/model; khởi tạo budget B.
2. Với mỗi c theo thứ tự đã khóa, train subset tới cùng rung cao nhất R, scheduler vẫn có horizon E. Q sớm chỉ chẩn đoán, không pruning.
3. Ngoại suy full time theo số ảnh train/val và worker setup. Tổng forecast×1.25+2s/config là scheduling heuristic, không confidence bound.
4. Nếu full sweep có vẻ nằm trong budget còn lại, giữ thứ tự sweep; nếu không, lần lượt khám phá các LR strata trước khi lặp một LR. Chỉ thử full c khi projection còn có thể fit; c bị bỏ do budget là unresolved.
5. Full-data train đủ E, không implicit OOM batch reduction/precision fallback. Ghi actual parameters, Q/T/M, checkpoint, raw logs. Mọi chi phí trial/profile/failure tính online.
6. Production tổng hợp đủ tất cả seeds trước lựa chọn. Independent research khóa decisions trước exhaustive oracle, tính regret từ oracle riêng. Không lén dùng oracle để sửa decisions.

Không viện dẫn Hyperband để gán bảo đảm của Hyperband cho heuristic này. Cơ sở đối chứng: [Random Search, Bergstra & Bengio2012](https://www.jmlr.org/papers/v13/bergstra12a.html); resource allocation liên quan [Hyperband, Li et al.2018](https://www.jmlr.org/papers/v18/16-558.html). Mã hiện tại triển khai Successive Halving riêng, không toàn bộ Hyperband.

## Ablation và đánh giá

- adaptive_no_profile: bỏ profile, full train theo cùng LR-stratified order.
- adaptive_no_strata: giữ profile/time margin, full train theo thứ tự ngẫu nhiên không phân tầng.
- Full policy: adaptive_verified. SH cung cấp đối chứng có quality pruning sớm; Random proxy cung cấp matched-fidelity control.
- Default/Random full/SH và mọi ablation cùng wall-cap B; báo actual search cost, overrun, feasibility, failed/censored trials. Không có lựa chọn thì regret null, không0.
- Regret = U(reference_best)-U(online_chosen), chỉ khi reference đủ và online choice feasible. Oracle cost tách riêng, không làm thuật toán online trở nên miễn phí.
- Sensitivity weights mặc định balanced0.6/0.25/0.15; accuracy-first0.8/0.1/0.1; time-first0.4/0.5/0.1; memory-first0.4/0.1/0.5. Đây là phân tích chứ không bốn giả thuyết độc lập đã có kiểm định p-value.
- Nghiên cứu tiếp: nhiều B, E, grid, dataset và GPU vật lý độc lập. Seed3 là pilot biến thiên, không đủ kết luận superiority thống kê.

## YOLO-specific kiểm soát

YOLO warmup_epochs=0 để accumulation đúng nguyên đã khai báo ngay từ đầu; nbs=batch×accumulation để weight_decay scaling không đổi decay đầu vào. Optimizer phải explicit (AdamW/SGD), không auto chọn LR/optimizer. Khóa precision; không implicit AMP-equivalence download/fallback. Kiểm actual batch/imgsz/precision/accumulation/optimizer/initial LR/decay trước xác nhận. Model YAML/weights được snapshot/hash, seed trước khởi tạo model. Đối chiếu API theo [Ultralytics train](https://docs.ultralytics.com/modes/train/) và mã version đã cài; không bảo đảm mọi phiên bản tương lai tương thích.

## Tính toàn vẹn không phải attestation

Hash/snapshots/raw audit phát hiện thay đổi artifact thông thường, không phải đăng ký khoa học độc lập hoặc chống giả mạo có chủ đích. Hardware UUID không là chữ ký phần cứng. Benchmark có job ML khác đồng thời phải bị chặn hoặc đánh dấu contended smoke, không công bố thời gian đó như clean benchmark.
