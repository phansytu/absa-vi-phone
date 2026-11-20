# Sổ tay tiền xử lý văn bản

**Dự án:** NLP-34 ABSA điện thoại tiếng Việt  
**Cập nhật:** 20/11/2026

## 1. Mục tiêu và nguyên tắc

Tiền xử lý nhằm gom các biến thể viết cùng một ý trong phản hồi tiếng Việt, đồng thời giữ những tín hiệu cần cho khía cạnh và cảm xúc. Mọi thay đổi dùng chung phải có lý do, ví dụ đối chứng và kiểm thử. Không thêm luật chỉ vì một câu demo được dự đoán sai; việc đó có thể làm hỏng các câu khác.

- Giữ nguyên split chính thức. Chỉ dùng Train để học từ vựng/IDF; dùng Dev để chọn cấu hình và ngưỡng.
- Cùng một chuẩn hóa cho huấn luyện và suy luận. PhoBERT còn dùng tách từ trước tokenizer; baseline dùng đặc trưng TF-IDF của riêng nó.
- Không coi khía cạnh vắng là Neutral; nhãn sentiment vắng được mask trong loss.
- Không sửa nội dung CSV gốc bằng tay. Thay đổi chuẩn hóa phải nằm trong `src/data/preprocess.py` và có test.

## 2. Các bước chuẩn hóa đang dùng

Hàm `normalize()` trong `src/data/preprocess.py` chạy theo thứ tự:

1. Chuẩn Unicode NFC, chuyển chữ thường, rút gọn khoảng trắng.
2. Tùy chọn rút ký tự lặp từ 3 lần trở lên, ví dụ `đẹppppp` thành `đẹp`; không rút chuỗi chữ số.
3. Tùy chọn chuẩn dấu thanh ở cuối từ, như `hoà` thành `hòa`. Mẫu giữa từ không đổi để tránh sửa nhầm `toàn`, `thoại`, `khoảng`.
4. Tùy chọn chuẩn hóa một danh sách viết tắt có ranh giới từ, ví dụ `ko`, `đt`, `sp`, `nv`, `mn`. Từ `k` chỉ đổi khi đứng riêng; `600k` và `600 k` được giữ.

Các cờ chuẩn hóa lưu cùng cấu hình/checkpoint. Cấu hình PhoBERT hiện dùng `tone=false`, `teencode=true`, `repeat=true`; baseline dùng cấu hình riêng ở `configs/baseline.yaml`.

## 3. Phủ định và từ nối

Baseline có thể dùng `mark_negation()` để gắn tiền tố `n_` lên tối đa ba token sau từ phủ định, dừng ở dấu câu hoặc từ nối đối lập như `nhưng`, `mà`. Ví dụ `pin không tốt nhưng màn hình đẹp` chỉ gắn `n_` cho `tốt`. PhoBERT không dùng luật này; encoder nhận văn bản đã chuẩn hóa/tách từ và tự học quan hệ ngữ cảnh.

Đây là đặc trưng từ vựng đơn giản, không phải bộ phân tích cú pháp. Phủ định dài, phủ định lồng nhau, câu thiếu dấu câu và cách nói mỉa mai vẫn có thể bị xử lý sai. Không diễn giải `n_` như một nhãn riêng trong đầu ra.

## 4. Ma trận giữ / chuẩn hóa / không tự động sửa

| Hiện tượng | Quyết định hiện tại | Lý do và giới hạn |
|---|---|---|
| Unicode tổ hợp và khoảng trắng lặp | Chuẩn hóa | Tránh hai dạng mã Unicode hoặc định dạng tương đương bị coi là khác nhau. |
| Chữ kéo dài (`đẹpppp`) | Rút còn một lần lặp tối đa | Giảm biến thể nhấn mạnh; có thể làm mất cường độ kéo dài. |
| Teencode trong danh sách cho phép | Đổi về dạng chuẩn | Gom viết tắt phổ biến; từ điển còn giới hạn, không bao quát mọi địa phương/ngữ cảnh. |
| `600k`, `600 k` | Giữ nguyên | Chữ `k` là đơn vị nghìn, không phải phủ định. |
| `sd`, `c`, `s` | Giữ nguyên | Có thể mang nghĩa thẻ SD, cổng Type-C, dòng Galaxy S; đổi máy móc có thể gây sai aspect. |
| `không`, `chưa`, dấu câu, emoji, chữ số | Giữ | Có thể mang tín hiệu cảm xúc, phủ định, giá hoặc cấu trúc câu. |
| Đại từ `nó`, từ nối `nhưng mà` | Không viết luật xóa/đổi riêng | Nghĩa phụ thuộc ngữ cảnh; chỉ bổ sung xử lý khi có phân tích lỗi lặp lại và phản ví dụ. |

## 5. Tokenizer và đồng bộ huấn luyện–suy luận

PhoBERT dùng văn bản word-segmented theo `use_segmented: true`. `ABSAPipeline` gọi `segment_text()` rồi chuẩn hóa theo cấu hình checkpoint trước khi đưa cho tokenizer. Khi sửa bước tách từ/chuẩn hóa, phải kiểm tra cả `src/data/dataset.py`, cấu hình trong checkpoint và pipeline inference để bảo đảm cùng thứ tự. Baseline nhận câu gốc vì vectorizer thực hiện chuẩn hóa riêng trong mô hình baseline.

Không khẳng định PhoBERT hiểu mọi teencode, đại từ quy chiếu hay câu ghép. Demo là dự đoán xác suất theo mẫu học được, không phải suy luận ngữ nghĩa hoàn hảo.

## 6. Kiểm chứng thay đổi

- Unit tests cho các luật hiện có nằm ở `tests/test_preprocess.py`: phủ định, teencode, đơn vị `k`, ký tự lặp, NFC, dấu thanh và phạm vi `mark_negation()`.
- Chạy `python -m pytest -q tests` sau khi sửa code. Lần xác nhận gần nhất ngày 20/11/2026: 25 passed.
- `experiments/audit_text.py` đo hiện tượng trong Train; `experiments/ablation_baseline.py` cung cấp thí nghiệm 5-fold CV cho biến thể baseline.
- Chưa lưu một bảng kết quả ablation gần nhất cùng cấu hình hiện tại trong artifact bàn giao. Vì vậy không kết luận một luật cụ thể làm tăng F1 nếu chưa chạy lại và lưu kết quả.
- Phân tích lỗi hiện tại dựa trên Dev; không dùng Test để chọn luật tiền xử lý.

## 7. Quy trình đề xuất khi thêm một luật

1. Ghi lại ví dụ thật và đếm tần suất trên Train; xem ngữ cảnh xung quanh.
2. Viết ít nhất một ví dụ cần đổi và một phản ví dụ không được đổi.
3. Thêm luật có phạm vi hẹp và test hồi quy.
4. Xem danh sách token bị thay đổi, kiểm tra tác động tới token aspect/giá/phủ định.
5. Chạy baseline/ablation trên Train CV hoặc đánh giá Dev theo quy trình đã định; ghi cấu hình và metrics.
6. Chỉ chốt luật sau khi so sánh có kiểm soát. Không quay lại Test để chọn luật.

## 8. Hạng mục còn mở

- Kiểm tra gần-trùng giữa các split và ghi lại phương pháp/kết quả.
- Nếu còn ngân sách máy, chạy ablation đã cấu hình và lưu artifact thay vì chỉ giữ script.
- Không thêm ViSoBERT vào tài liệu kết quả hiện tại; checkpoint đang bàn giao dùng PhoBERT `vinai/phobert-base`.
