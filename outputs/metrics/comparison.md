# So sánh cuối trên tập Test

Hai mô hình được đánh giá trên cùng 2.224 mẫu. Ngưỡng TF-IDF V2C được chọn trên Dev; ngưỡng PhoBERT lấy từ checkpoint đã chọn trên Dev. PhoBERT là một lần huấn luyện với seed 42.

| Mô hình | Aspect Micro-F1 | Aspect Macro-F1 | Sentiment Macro-F1 (gold aspect) | Joint Precision | Joint Recall | Joint F1 |
|:--|--:|--:|--:|--:|--:|--:|
| TF-IDF V2C | 0.8606 | 0.8344 | 0.7720 | 0.7609 | 0.7254 | 0.7427 |
| PhoBERT MultiTask, seed 42 | **0.9002** | **0.8672** | **0.8282** | **0.8085** | **0.8197** | **0.8141** |

## Nhận xét

- PhoBERT hơn TF-IDF 0.0713 điểm Joint F1 tuyệt đối.
- PhoBERT đạt F1 cao hơn ở 9/10 khía cạnh về aspect detection. TF-IDF tốt hơn ở STORAGE (0.7119 so với 0.6250).
- Với Joint F1 theo khía cạnh, TF-IDF tốt hơn ở STORAGE (0.5424 so với 0.3750). PhoBERT vẫn tốt hơn tổng thể.
- PhoBERT được huấn luyện một seed; kết quả chưa đo độ dao động giữa các seed.

Nguồn số liệu: `multitask_seed42_test.json`, `baseline_test_v2c.json`. Không dùng Test để chọn ngưỡng hay tinh chỉnh mô hình.
