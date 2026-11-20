# Kết quả thực nghiệm cũ

Các tệp ở đây là artifact được giữ lại để tra cứu, không dùng làm kết quả hiện tại trong báo cáo so sánh cuối.

- `baseline_dev.json` và `baseline_dev.jsonl`: baseline trước khi chuyển sang ngưỡng V2C.
- `hybrid.json`: thí nghiệm hybrid cũ, không nằm trong so sánh cuối hiện tại.
- `multitask_overfit_seed42.json`: smoke/overfit diagnostic, không phải checkpoint chọn cuối.
- `phobert_thresholds.json`: ngưỡng thử nghiệm cũ; checkpoint seed 42 hiện lưu ngưỡng Dev chính thức.
- `multitask_seed123_dev.jsonl`, `multitask_seed456_dev.jsonl`: prediction từ các lần chạy khác; không ghép với kết quả seed 42 hiện tại để báo cáo mean/std.

Nguồn kết quả cuối: `outputs/metrics/comparison.csv` và các tệp JSON Test cùng tên mô hình trong `outputs/metrics/`.
