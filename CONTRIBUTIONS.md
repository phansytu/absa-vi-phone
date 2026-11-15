# Bảng Phân Công Công Việc & Nhật Ký Đóng Góp (Contributions Log)
**Dự án:** Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA) trên dữ liệu UIT-ViSFD  
**Nhóm:** NLP-34  

---

## 1. Phân công vai trò và trách nhiệm (5 Thành viên)

| Thành viên | Vai trò chính | Module / File phụ trách chính | Trạng thái |
| :--- | :--- | :--- | :---: |
| **TV1** | **Kỹ sư Dữ liệu (Data Engineer)** | `src/data/dataset.py`, `src/data/schema.py`, `src/data/load.py`, `notebooks/01_eda.ipynb`, `tests/test_schema.py` | Hoàn thành |
| **TV2** | **Kỹ sư Đánh giá (Evaluation & Metric Lead)** | `src/evaluate.py`, `tests/test_metrics.py`, `notebooks/02_baseline_check.ipynb`, Quản lý đánh giá độc lập tập Test | Hoàn thành |
| **TV3** | **Kỹ sư Mô hình Sâu (Deep Learning Engineer)** | `src/models/multitask.py`, `src/train.py`, `configs/multitask.yaml`, `tests/test_loss_mask.py`, `scripts/run_seeds.sh` | Hoàn thành |
| **TV4** | **Kiến trúc sư Thí nghiệm & CI/CD (MLOps & Docs)** | `src/models/baseline.py`, `experiments/ablation_baseline.py`, `experiments/audit_text.py`, `configs/*.yaml`, `.github/workflows/ci.yml`, `README.md`, `CONTRIBUTIONS.md` | Hoàn thành |
| **TV5** | **Kỹ sư Triển khai & Ứng dụng (Deployment & App)** | `src/pipeline.py`, `src/predict.py`, `demo/app.py`, `tests/test_pipeline.py`, `notebooks/03_error_analysis.ipynb` | Hoàn thành |

---

## 2. Chi tiết đóng góp của từng thành viên

### Thành viên 1 (TV1): Data Processing & PyTorch Dataset
- [x] **Xây dựng Schema nhãn (`src/data/schema.py`):**
  - Định nghĩa 10 khía cạnh (`ASPECTS`), 3 cực tính (`POLARITES`).
  - Hàm `parse_label`: Xử lý định dạng chuỗi nhãn của UIT-ViSFD, xử lý nhãn đặc biệt `{OTHERS}`.
  - Hàm `to_vectors` & `from_vectors`: Chuyển đổi nhãn sang vector và ngược lại, áp dụng quy tắc nền tảng gán `-100` (`IGNORE_INDEX`) cho khía cạnh vắng mặt.
- [x] **Trích xuất dữ liệu (`src/data/load.py`):**
  - Đọc file CSV gốc, kiểm tra số mẫu chuẩn: Train (7.786), Dev (1.112), Test (2.224).
  - Xuất dữ liệu sang định dạng JSONL `data/processed/*.jsonl`.
  - Bổ sung bảo mật chống tấn công Path Traversal.
- [x] **PyTorch Dataset (`src/data/dataset.py`):**
  - Xây dựng lớp `ABSADataset` và hàm `make_collate` phục vụ huấn luyện mô hình sâu.
- [x] **Kiểm thử Schema (`tests/test_schema.py`):**
  - Viết bộ unit test kiểm tra parse nhãn, bắt lỗi nhãn không hợp lệ hoặc lặp khía cạnh.
- [x] **Notebook EDA (`notebooks/01_eda.ipynb`):**
  - Khảo sát phân bố độ dài, tần suất các khía cạnh và tỷ lệ cảm xúc.

### Thành viên 2 (TV2): Metric System & Evaluation
- [x] **Bộ chỉ số đánh giá chuẩn (`src/evaluate.py`):**
  - Cài đặt đầy đủ các metric theo tiêu chuẩn Mục 13:
    1. *Aspect Micro-F1* & *Aspect Macro-F1* (kèm F1 từng khía cạnh).
    2. *Sentiment Macro-F1 given gold aspect* (chỉ số chẩn đoán cực tính).
    3. *Joint Precision, Recall, F1* (chỉ số End-to-End quan trọng nhất).
    4. *Empty text accuracy* (tỷ lệ dự đoán đúng mẫu rỗng).
- [x] **Kiểm chứng Metric bằng tính tay (`tests/test_metrics.py`):**
  - Cài đặt test case tính tay Mục 6.4 để bảo vệ tính chính xác của metric trước khi áp dụng cho mô hình.
- [x] **Notebook đánh giá Baseline (`notebooks/02_baseline_check.ipynb`):**
  - Đánh giá chi tiết kết quả của Baseline trên tập Dev, so sánh độ nhạy của các khía cạnh.
- [x] **Giữ quyền bảo mật tập Test (Quy tắc 7.1.3):**
  - Đảm bảo toàn bộ quá trình tinh chỉnh siêu tham số và luật tiền xử lý chỉ được dùng trên Train và Dev.

### Thành viên 3 (TV3): Multi-Task Deep Learning Model
- [x] **Kiến trúc mô hình Multi-task (`src/models/multitask.py`):**
  - Xây dựng lớp `MultiTaskABSA` sử dụng encoder tiền huấn luyện `uitnlp/visobert`.
  - Thiết kế 2 đầu ra (Dual Task Heads): 1 đầu ra nhị phân phát hiện hiện diện khía cạnh (`Linear(H, 10)`), 1 đầu ra đa lớp phân loại cảm xúc (`Linear(H, 30)`).
  - Cơ chế pooling (`mean` / `cls`).
- [x] **Hàm Loss Masking (`multitask_loss`):**
  - Kết hợp Binary Cross Entropy cho hiện diện và Cross Entropy cho cảm xúc.
  - Sử dụng mask `sent_y != -100` để chỉ tính loss cực tính cho các khía cạnh có mặt, tránh mô hình học sai rằng vắng mặt là Neutral.
  - Xử lý an toàn trường hợp batch không có khía cạnh nào để triệt tiêu lỗi `NaN`.
- [x] **Vòng lặp huấn luyện (`src/train.py`):**
  - Tích hợp AdamW, Linear Warmup Scheduler, Automatic Mixed Precision (`torch.amp`).
  - Hỗ trợ các cờ chạy: `--limit` (smoke test), `--overfit` (kiểm tra hội tụ), `--set` (ghi đè siêu tham số).
- [x] **Thực nghiệm đa Seed (`scripts/run_seeds.sh`, `scripts/run_seeds.ps1`):**
  - Tự động hóa chạy 3 seed (42, 43, 44) và tổng hợp kết quả Mean ± Std.

### Thành viên 4 (TV4): Baseline Model, Ablation & MLOps
- [x] **Mô hình Baseline (`src/models/baseline.py`):**
  - Xây dựng lớp `TfidfBaseline`: TF-IDF n-gram (1,2) + 10 bộ Logistic Regression hiện diện + 10 bộ Logistic Regression cảm xúc.
  - Xử lý fallback cho trường hợp lớp đơn.
- [x] **Runner Baseline (`src/run_baseline.py`):**
  - Quét tìm ngưỡng tối ưu toàn cục (Global Threshold) trên tập Dev theo Joint F1.
  - Đạt kết quả xuất sắc: **Joint F1 Dev = 0.751** tại ngưỡng 0.45.
- [x] **Thí nghiệm Ablation (`experiments/ablation_baseline.py`):**
  - Chạy 5-fold CV trên Train đo lường tác động của từng bước tiền xử lý (NFC, dấu thanh, teencode, phủ định, bigram).
- [x] **Kiểm kê dữ liệu thô (`experiments/audit_text.py`):**
  - Đo đạc tỷ lệ bất thường khoảng trắng, biến thể dấu thanh và lỗi telex trên dữ liệu thực.
- [x] **Thiết lập CI/CD & Cấu hình:**
  - Hoàn thiện `.github/workflows/ci.yml`, các file cấu hình `configs/baseline.yaml`, `configs/separate.yaml`, `configs/aspect_conditioned.yaml`.
  - Soạn thảo tài liệu `README.md`, `LICENSE`, `CONTRIBUTIONS.md`.

### Thành viên 5 (TV5): End-to-End Pipeline, CLI & Web Application
- [x] **End-to-End Pipeline (`src/pipeline.py`):**
  - Xây dựng lớp `ABSAPipeline` đóng gói trọn vẹn luồng suy luận: Tiền xử lý -> Mô hình (ViSoBERT / Baseline) -> Áp dụng ngưỡng -> Xuất kết quả.
  - Tích hợp giải nghĩa tiếng Việt và biểu tượng trực quan cho 10 khía cạnh và 3 cực tính.
- [x] **Giao diện dòng lệnh CLI (`src/predict.py`):**
  - Hỗ trợ chế độ dự đoán 1 câu trực tiếp (`--text`).
  - Hỗ trợ chế độ tương tác gõ câu liên tục (`--interactive` / REPL).
  - Hỗ trợ dự đoán theo file hàng loạt (`--file`).
- [x] **Giao diện Web Gradio (`demo/app.py`):**
  - Xây dựng ứng dụng Web tương tác hiện đại với Gradio.
  - Hiển thị thẻ màu (Xanh/Đỏ/Vàng) cho từng khía cạnh, thanh điều chỉnh ngưỡng thời gian thực, biểu đồ phân bố xác suất và các câu ví dụ mẫu.
- [x] **Kiểm thử Pipeline (`tests/test_pipeline.py`):**
  - Viết bộ unit test cho pipeline (chuỗi rỗng, batch, định dạng output).
- [x] **Notebook Phân tích lỗi (`notebooks/03_error_analysis.ipynb`):**
  - Đào sâu phân tích nguyên nhân lỗi trên 3 nhóm thử thách: đa khía cạnh, cảm xúc trái chiều, và mẫu rỗng `{OTHERS}`.

---

## 3. Nhật ký các Pull Request & Cột mốc chính

| PR / Commit | Tác giả | Nội dung chính | Ngày |
| :--- | :--- | :--- | :---: |
| **PR #1** | TV1 & TV4 | Khởi tạo cấu trúc dự án, schema nhãn, script tải dữ liệu thô và unit test tiền xử lý | 18/10/2026 |
| **PR #2** | TV4 | Bổ sung text audit, ablation script và sửa lỗi bảo mật path traversal | 19/10/2026 |
| **PR #3** | TV2 & TV4 | Triển khai mô hình Baseline TF-IDF, bộ metric chuẩn Mục 13 và runner dev | 19/10/2026 |
| **PR #4** | TV1 & TV3 | Tích hợp kiến trúc Multi-task ViSoBERT, PyTorch Dataset, hàm loss mask và vòng lặp huấn luyện | 24/10/2026 |
| **Feature TV5** | TV5 | Hoàn thiện End-to-End Pipeline, CLI tương tác trực tiếp, Gradio Web UI và bộ Unit test kiểm thử | 02/11/2026 |
| **Finalization** | Cả nhóm | Hoàn thiện toàn bộ hệ thống configs, utility, notebook EDA & Error Analysis, tài liệu kỹ thuật | 07/11/2026 |
