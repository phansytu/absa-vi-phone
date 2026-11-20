# Phân tích cảm xúc theo khía cạnh cho đánh giá điện thoại tiếng Việt

**Đề tài:** NLP-34 — Aspect-Based Sentiment Analysis (ABSA)  
**Nhóm:** NLP-34  
**Phiên bản kết quả:** 20/11/2026  
**Trạng thái:** kết quả thực nghiệm hiện có; cần điền tên thành viên và xác nhận đóng góp trước khi nộp.

## Tóm tắt

Đề tài xây dựng hệ thống nhận xét đánh giá điện thoại tiếng Việt theo từng khía cạnh. Hệ thống phát hiện khía cạnh được nhắc tới trong câu và gán một trong ba nhãn Positive, Negative, Neutral cho mỗi khía cạnh hiện diện. Thử nghiệm so sánh baseline TF-IDF với PhoBERT Multi-task trên UIT-ViSFD. Trên 2.224 câu Test, PhoBERT seed 42 đạt Joint F1 0,8141, cao hơn baseline TF-IDF V2C đạt 0,7427. PhoBERT có kết quả thấp hơn baseline ở STORAGE theo Joint F1 (0,3750 so với 0,5424), và tập rỗng vẫn khó nhận diện. PhoBERT hiện chỉ có một seed; chưa có ước lượng độ dao động giữa các lần huấn luyện.

## Chương 1. Bài toán và mục tiêu

ABSA cần xác định đồng thời **đối tượng được nói tới** và **thái độ dành cho đối tượng đó**. Đầu vào là một câu đánh giá điện thoại. Đầu ra là danh sách cặp khía cạnh–cảm xúc. Hệ thống dùng 10 khía cạnh: SCREEN, CAMERA, FEATURES, BATTERY, PERFORMANCE, STORAGE, DESIGN, PRICE, GENERAL, SER&ACC; cảm xúc gồm Positive, Negative và Neutral.

Một khía cạnh không được nhắc tới phải bị bỏ khỏi kết quả, không được gán Neutral. Vì vậy bài toán có hai phần: phân loại hiện diện đa nhãn và phân loại cảm xúc có điều kiện trên khía cạnh hiện diện. Mục tiêu thực nghiệm là so sánh TF-IDF + Logistic Regression với encoder PhoBERT có hai đầu ra đa nhiệm và đánh giá đầu ra kết hợp bằng Joint F1.

## Chương 2. Dữ liệu và tiền xử lý

Sử dụng các split chính thức của UIT-ViSFD: Train 7.786 câu, Dev 1.112 câu, Test 2.224 câu, tổng cộng 11.122 câu. Dữ liệu giữ nguyên split; cột số sao không được dùng làm đầu vào mô hình. Văn bản được chuẩn hóa Unicode, teencode và ký tự lặp theo cấu hình của từng mô hình. PhoBERT dùng văn bản đã tách từ; inference thực hiện cùng bước tách từ và chuẩn hóa như lúc huấn luyện.

Nhãn cảm xúc của khía cạnh vắng được mask bằng -100 để không tính sentiment loss tại vị trí đó. Dữ liệu gốc được tải riêng theo điều khoản của nguồn; repo không nên phân phối dữ liệu nếu chưa được phép.

## Chương 3. Phương pháp

### 3.1 Baseline TF-IDF

Baseline biểu diễn câu bằng TF-IDF word n-gram (1, 2), sau đó dùng các bộ Logistic Regression độc lập cho dự đoán hiện diện và cảm xúc từng khía cạnh. Ngưỡng hiện diện riêng cho từng khía cạnh được chọn trên Dev bằng quy tắc V2C. Baseline giúp kiểm tra mức hiệu quả có thể đạt được từ đặc trưng từ vựng mà không cần encoder sâu.

### 3.2 PhoBERT Multi-task

Encoder chính là `vinai/phobert-base`, mean pooling, dropout 0,1 và hai đầu ra: 10 logits hiện diện cùng 30 logits cảm xúc (10 khía cạnh × 3 lớp). Mô hình được huấn luyện với batch size 32, tối đa 8 epoch, learning rate encoder 2e-5, learning rate các đầu ra 5e-4, weight decay 0,01. Sentiment loss bỏ qua khía cạnh vắng bằng nhãn -100. Checkpoint được chọn theo macro joint metric trên Dev; checkpoint tốt nhất là epoch 7. Ngưỡng 10 khía cạnh được lưu cùng checkpoint và được chọn trên Dev.

## Chương 4. Thiết lập đánh giá và chi phí

Hai mô hình được đánh giá trên cùng split và cùng hàm metric. Aspect micro/macro F1 đo phát hiện khía cạnh; sentiment macro F1 được tính khi biết trước khía cạnh gold và chỉ là chỉ số chẩn đoán; Joint Precision/Recall/F1 yêu cầu đúng cả khía cạnh lẫn cảm xúc. Tập Test chỉ được dùng sau khi cấu hình và ngưỡng đã chốt.

PhoBERT seed 42 được huấn luyện trên CPU do `torch.cuda.is_available()` trả về False. Log lần chạy ghi 13.723 giây (khoảng 3 giờ 49 phút), checkpoint khoảng 540 MB. Các cảnh báo HF Hub về xác thực và symlink không làm dừng tải mô hình. Không có thông tin RAM/CPU chi tiết trong log nên không đưa ra so sánh phần cứng.

## Chương 5. Kết quả

| Mô hình | Số câu | Aspect Micro-F1 | Aspect Macro-F1 | Sentiment Macro-F1 (gold aspect) | Joint Precision | Joint Recall | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF V2C | 2.224 | 0,8606 | 0,8344 | 0,7720 | 0,7609 | 0,7254 | 0,7427 |
| PhoBERT Multi-task, seed 42 | 2.224 | **0,9002** | **0,8672** | **0,8282** | **0,8085** | **0,8197** | **0,8141** |

Chênh lệch Joint F1 là 0,0713 điểm tuyệt đối, nghiêng về PhoBERT. PhoBERT cao hơn ở 9/10 aspect detection F1. STORAGE là ngoại lệ: aspect F1 0,6250 và Joint F1 0,3750, trong khi TF-IDF đạt tương ứng 0,7119 và 0,5424. Chênh lệch này cần được diễn giải thận trọng vì STORAGE hiếm và mô hình PhoBERT mới có một seed.

PhoBERT Dev có 1.112 câu, aspect micro-F1 0,9101, macro-F1 0,8952, sentiment macro-F1 0,8183 và Joint F1 0,8227. Có 18 câu Dev không có khía cạnh; mô hình dự đoán rỗng đúng 3/18 (16,7%). Trên Test có 37 câu rỗng; dự đoán rỗng đúng 14/37 (37,8%).

Test gồm 6.722 cặp khía cạnh–cảm xúc gold hiện diện và 37 câu rỗng. F1 theo từng aspect được liệt kê dưới đây; báo cáo kèm số liệu này để thể hiện khác biệt giữa mức phát hiện aspect và mức dự đoán đúng cả polarity.

| Aspect | TF-IDF Aspect F1 | PhoBERT Aspect F1 | TF-IDF Joint F1 | PhoBERT Joint F1 |
|---|---:|---:|---:|---:|
| SCREEN | 0,7542 | 0,8359 | 0,6543 | 0,7737 |
| CAMERA | 0,9303 | 0,9615 | 0,7529 | 0,8411 |
| FEATURES | 0,8604 | 0,8710 | 0,7292 | 0,7754 |
| BATTERY | 0,9593 | 0,9639 | 0,8012 | 0,8605 |
| PERFORMANCE | 0,8788 | 0,9167 | 0,7455 | 0,8161 |
| STORAGE | 0,7119 | 0,6250 | 0,5424 | 0,3750 |
| DESIGN | 0,7755 | 0,8663 | 0,6862 | 0,7866 |
| PRICE | 0,8622 | 0,9174 | 0,6890 | 0,7814 |
| GENERAL | 0,8048 | 0,8735 | 0,7575 | 0,8347 |
| SER&ACC | 0,8069 | 0,8404 | 0,7455 | 0,7870 |

## Chương 6. Phân tích lỗi trên Dev

Phân tích lỗi dùng dự đoán Dev mới nhất, không dùng Test để điều chỉnh. Theo số khía cạnh gold, PhoBERT đạt Joint F1 0,7397 trên 156 câu có một khía cạnh, 0,9015 trên 261 câu có hai khía cạnh, và 0,9268 trên 677 câu có từ ba khía cạnh trở lên. Nhóm không có khía cạnh gồm 18 câu; tỷ lệ dự đoán rỗng đúng chỉ 16,7%.

Trong 265 câu có cảm xúc trái chiều Positive và Negative, 132 câu được dự đoán chính xác toàn bộ các cặp gold (49,8%). Ví dụ `dev-449`: “Mong đợi là sẽ có 5g.hơi thất vọng.nhưng về cấu hình và mức giá này thì là quá ok”. Gold gồm PERFORMANCE→Positive, PRICE→Positive, GENERAL→Negative; dự đoán đảo cảm xúc PRICE và GENERAL. Lỗi này cho thấy mô hình có thể nhận ra nhiều khía cạnh nhưng gán nhầm cảm xúc khi các ý trái chiều nối trong cùng câu.

Ví dụ false positive trên câu gold rỗng `dev-859`: người viết đang hỏi ý kiến cộng đồng và nhắc lại đánh giá của người khác; mô hình vẫn gán PRICE→Negative và GENERAL→Negative. Từ “chê”, “xót tiền” có thể khiến mô hình diễn giải nội dung được trích dẫn như quan điểm của người viết. Kết quả nên được cải thiện bằng cách bổ sung ví dụ rỗng/ý kiến được dẫn lại vào Train hoặc phân tích đặc trưng trên Dev, sau đó chỉ đánh giá Test khi có một cấu hình mới được chốt cho chu kỳ nghiên cứu mới.

Một câu demo người dùng từng thử, “Điện thoại đắt nhưng mà nó hiện đại”, chỉ phát hiện PRICE→Negative, bỏ sót DESIGN→Positive. Đây là ví dụ vận hành minh họa, không phải mẫu Test chính thức; kết quả cho thấy mô hình còn bỏ sót quan hệ ngữ nghĩa và khía cạnh ngầm.

## Chương 7. Kết luận, giới hạn và tái lập

Dự án hiện có baseline, encoder đa nhiệm, pipeline dự đoán, CLI/web demo, kiểm thử đơn vị và kết quả so sánh có lưu dự đoán theo ID. Với cấu hình hiện tại, PhoBERT tốt hơn TF-IDF về Joint F1 tổng thể. Các giới hạn chính là chỉ có một seed PhoBERT, độ bao phủ thấp trên mẫu rỗng, khía cạnh STORAGE ít dữ liệu, và chi phí huấn luyện CPU cao. Chưa thể khẳng định mức cải thiện ổn định qua lần chạy.

Trước khi nộp, nhóm cần (1) thay TV1–TV5 bằng tên thật và xác nhận đóng góp, (2) chạy `python -m pytest -q tests` trên môi trường nộp, (3) xác nhận near-duplicate audit và ghi rõ nếu chưa làm, (4) lưu checkpoint trong nơi bàn giao được phép hoặc chuẩn bị hướng dẫn train lại, (5) diễn tập demo với câu bình thường, câu khó và đầu vào rỗng/sai định dạng. Nếu còn thời gian tính toán, chạy seed 43 và 44 trên cùng cấu hình rồi báo mean ± std; không trộn các kết quả cũ khác phiên bản vào thống kê này.

### Lệnh tái lập chính

```powershell
python -m pytest -q tests
python -m src.run_baseline data/raw outputs
python -m src.train --config configs/multitask.yaml --seed 42
python -m src.demo --model multitask --detailed
```

Chấm Test chỉ dùng đúng lệnh có cờ `--test` với baseline sau khi cấu hình được đóng băng; PhoBERT đã có kết quả cuối được lưu trong `outputs/metrics/multitask_seed42_test.json` và dự đoán trong `outputs/predictions/multitask_seed42_test.jsonl`.

## Tài liệu tham khảo

1. UIT-ViSFD, bộ dữ liệu phân tích cảm xúc theo khía cạnh cho đánh giá điện thoại tiếng Việt: https://github.com/LuongPhan/UIT-ViSFD
2. Nguyen, D. Q. & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese*. Findings of EMNLP 2020. https://aclanthology.org/2020.findings-emnlp.92/
3. VinAI Research, PhoBERT model repository: https://github.com/VinAIResearch/PhoBERT
4. scikit-learn, `f1_score` documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
5. Yêu cầu đề tài NLP-34 trong tài liệu “Danh sách Đề tài nhóm NLP”, mục 4.34.
