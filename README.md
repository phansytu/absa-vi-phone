# 📱 Phân Tích Cảm Xúc Theo Khía Cạnh Cho Đánh Giá Điện Thoại Tiếng Việt (ABSA UIT-ViSFD)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án nghiên cứu và triển khai hệ thống **Aspect-Based Sentiment Analysis (ABSA)** trên tập dữ liệu phản hồi sinh viên về điện thoại thông minh (**UIT-ViSFD** gồm 11.122 câu đánh giá).

---

## 📌 1. Giới thiệu Bài toán

Cho một câu đánh giá tiếng Việt bất kỳ, hệ thống tự động:
1. Xác định câu có nhắc đến khía cạnh nào trong **10 khía cạnh điện thoại** hay không:
   - `SCREEN` (Màn hình)
   - `CAMERA` (Máy ảnh)
   - `FEATURES` (Tính năng & Tiện ích)
   - `BATTERY` (Pin & Nguồn)
   - `PERFORMANCE` (Hiệu năng & Tốc độ)
   - `STORAGE` (Bộ nhớ lưu trữ)
   - `DESIGN` (Thiết kế & Ngoại hình)
   - `PRICE` (Giá cả)
   - `GENERAL` (Đánh giá chung)
   - `SER&ACC` (Dịch vụ & Phụ kiện đi kèm)
2. Phân loại cảm xúc tương ứng cho từng khía cạnh có mặt thành **3 cực tính**:
   - `Positive` (Tích cực / Khen)
   - `Negative` (Tiêu cực / Chê)
   - `Neutral` (Trung tính)

> ⚠️ **Nguyên tắc nền tảng:** Khía cạnh **không được nhắc tới thì không xuất ra** và **tuyệt đối không được gán nhãn `Neutral`**. Trong huấn luyện, khía cạnh vắng được gán nhãn `-100` (`IGNORE_INDEX`) để hàm Loss bỏ qua.

---

## 🏗️ 2. Luồng Xử Lý Tổng Thể (System Architecture)

```
Văn bản đánh giá thô
       │
       ▼
Tiền xử lý (src/data/preprocess.py)
   ├── Chuẩn Unicode NFC, chữ thường
   ├── Chuẩn hóa dấu thanh cuối từ (hoà -> hòa)
   ├── Gộp teencode/viết tắt (ko, k, đt, sp, nv, mn, 600k...)
   └── Rút gọn ký tự lặp (đẹpppp -> đẹp)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Mô hình Đối chứng Baseline             Mô hình Sâu Multi-task
(TF-IDF + 10 LR Hiện diện + 10 LR Cảm xúc)   (Pre-trained ViSoBERT + Dual Heads)
       │                                         │
       ▼                                         ▼
Xác suất hiện diện [10] & Cực tính [10]      Xác suất hiện diện [10] & Cực tính [10]
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
              Áp dụng Ngưỡng tối ưu (Threshold)
                            │
                            ▼
           Kết quả cuối: Danh sách [{aspect, sentiment}]
           (Chỉ chứa các khía cạnh có mặt thực tế)
```

---

## 🚀 3. Hướng Dẫn Cài Đặt và Chạy Từ Đầu Đến Cuối

### Bước 1: Chuẩn bị môi trường ảo

```powershell
# Di chuyển vào thư mục gốc của dự án
cd absa-vi-phone

# Khởi tạo và kích hoạt virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Trên Linux/macOS: source .venv/bin/activate

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
pip install protobuf
```

*(Khuyến nghị nếu máy có GPU NVIDIA, hãy cài PyTorch hỗ trợ CUDA để tăng tốc huấn luyện mô hình sâu: `pip install torch --index-url https://download.pytorch.org/whl/cu121`)*

---

### Bước 2: Chuẩn bị và chuyển đổi dữ liệu

Dữ liệu thô đã được lưu tại `data/raw/` (`Train.csv`, `Dev.csv`, `Test.csv`). Chạy lệnh sau để chuyển đổi sang định dạng JSONL chuẩn hóa:

```powershell
python -m src.data.load data/raw data/processed
```
*Kết quả:* Sinh ra `train.jsonl` (7.786 dòng), `dev.jsonl` (1.112 dòng), `test.jsonl` (2.224 dòng) tại `data/processed/`.

---

### Bước 3: Chạy Unit Test kiểm thử hệ thống

Chạy toàn bộ bộ test kiểm tra tiền xử lý, tính toán metric, hàm loss mask, schema và pipeline:

```powershell
python -m pytest -v tests
```
*Kết quả kỳ vọng:* Toàn bộ các test case vượt qua (`passed`).

---

### Bước 4: Chạy mô hình Baseline (TF-IDF + Logistic Regression)

Huấn luyện mô hình baseline trên tập Train, quét ngưỡng tối ưu trên tập Dev và lưu kết quả:

```powershell
python -m src.run_baseline data/raw outputs
```

*Kết quả đạt được trên tập Dev:*
- **Aspect Micro-F1:** `0.868`
- **Aspect Macro-F1:** `0.822`
- **Sentiment Macro-F1 (given gold):** `0.777`
- **Joint Aspect–Polarity F1 (End-to-End):** **`0.751`**
- **Ngưỡng tối ưu:** `0.45`
- File sinh ra: `outputs/metrics/baseline_dev.json`, `outputs/predictions/baseline_dev.jsonl`.

---

### Bước 5: Chạy Thí nghiệm Ablation Tiền xử lý (5-Fold CV trên Train)

```powershell
python experiments/ablation_baseline.py data/raw
```

So sánh 8 cấu hình tiền xử lý để đánh giá sự đóng góp của chuẩn NFC, vị trí dấu thanh, từ điển teencode, tiền tố phủ định `n_` và n-gram.

---

### Bước 6: Huấn luyện Mô hình Sâu Đa nhiệm (Multi-task ViSoBERT)

Huấn luyện mô hình mạng nơ-ron sâu dùng cấu hình `configs/multitask.yaml`:

```powershell
# Huấn luyện chính thức Seed 42
python -m src.train --config configs/multitask.yaml --seed 42

# (Tùy chọn) Chạy tự động 3 seed (42, 43, 44) để tính Mean ± Std theo Quy tắc 7.1.6:
powershell scripts/run_seeds.ps1      # Trên Linux: bash scripts/run_seeds.sh
```

---

### Bước 7: Thực Hành Dự Đoán (Inference)

#### Cách 1: Giao diện dòng lệnh tương tác trực tiếp (CLI REPL)
```powershell
python -m src.predict
```
Nhập bất kỳ câu đánh giá nào và nhận bảng phân tích tức thì:
```text
👉 Nhập câu đánh giá: Pin trâu dùng 2 ngày nhưng màn hình mờ, camera chụp đêm xấu

📝 Câu gốc       : Pin trâu dùng 2 ngày nhưng màn hình mờ, camera chụp đêm xấu
🔍 Đã chuẩn hóa  : pin trâu dùng 2 ngày nhưng màn hình mờ, camera chụp đêm xấu
📌 Phát hiện được: 3 khía cạnh
----------------------------------------------------------------------
Khía cạnh (Aspect)     | Giải nghĩa         | Cực tính (Sentiment)   | Độ tin cậy
----------------------------------------------------------------------
BATTERY                | Pin & Nguồn        | 👍 Positive (Tích cực) | 96.42%
SCREEN                 | Màn hình           | 👎 Negative (Tiêu cực) | 88.15%
CAMERA                 | Máy ảnh            | 👎 Negative (Tiêu cực) | 91.20%
----------------------------------------------------------------------
```

#### Cách 2: Khởi chạy Giao diện Web (Gradio Web UI)
```powershell
python demo/app.py
```
Mở trình duyệt tại **`http://127.0.0.1:7860`** để sử dụng giao diện đồ họa trực quan với các thẻ màu, thanh điều chỉnh ngưỡng và biểu đồ xác suất 10 khía cạnh.

---

## 📂 4. Cấu Trúc Thư Mục Dự Án

```
absa-vi-phone/
├── configs/                      # Cấu hình siêu tham số thí nghiệm (YAML)
│   ├── baseline.yaml             # Cấu hình TF-IDF Baseline
│   ├── multitask.yaml            # Cấu hình Multi-task ViSoBERT
│   ├── separate.yaml             # Cấu hình Separate Models
│   └── aspect_conditioned.yaml   # Cấu hình Aspect-Conditioned Model
├── data/
│   ├── download_data.sh          # Script tải dữ liệu UIT-ViSFD từ tác giả
│   ├── raw/                      # Dữ liệu gốc: Train.csv, Dev.csv, Test.csv (Không commit)
│   └── processed/                # Dữ liệu đã chuyển đổi: train/dev/test.jsonl
├── demo/
│   └── app.py                    # Ứng dụng Web UI tương tác bằng Gradio
├── docs/
│   ├── HUONG_DAN_NHOM.md         # Sổ tay hướng dẫn chi tiết quy tắc làm việc nhóm
│   └── SO_TAY_TIEN_XU_LY.md      # Tài liệu phân tích chuyên sâu các luật tiền xử lý
├── experiments/
│   ├── audit_text.py             # Script kiểm kê và đo lường bất thường dữ liệu thô
│   └── ablation_baseline.py      # Thí nghiệm 5-fold CV khảo sát tác động tiền xử lý
├── notebooks/                    # Jupyter Notebooks nghiên cứu & trực quan hóa
│   ├── 01_eda.ipynb              # Khám phá phân bố dữ liệu, khía cạnh & cực tính
│   ├── 02_baseline_check.ipynb   # Đánh giá và kiểm tra kết quả Baseline trên Dev
│   └── 03_error_analysis.ipynb   # Phân tích lỗi đa khía cạnh, trái chiều & mẫu rỗng
├── outputs/                      # Thư mục lưu kết quả thực nghiệm
│   ├── checkpoints/              # File trọng số mô hình (.pt)
│   ├── metrics/                  # Chỉ số đánh giá chi tiết (.json)
│   └── predictions/              # Dự đoán từng câu trên tập Dev (.jsonl)
├── scripts/
│   ├── checks.py                 # Kiểm tra tính toàn vẹn dữ liệu
│   ├── export_clean_data.py      # Xuất dữ liệu đã làm sạch sang CSV
│   ├── run_seeds.sh              # Chạy thực nghiệm 3 seed (Bash)
│   └── run_seeds.ps1             # Chạy thực nghiệm 3 seed (PowerShell)
├── src/                          # Mã nguồn chính của dự án
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset & Collate function
│   │   ├── load.py               # Nạp CSV, kiểm tra mẫu, xuất JSONL
│   │   ├── preprocess.py         # Hàm normalize() & mark_negation()
│   │   └── schema.py             # Định nghĩa nhãn, parse_label, to/from_vectors
│   ├── models/
│   │   ├── baseline.py           # Mô hình TF-IDF + Logistic Regression
│   │   └── multitask.py          # Mô hình Deep Multi-task ViSoBERT + Loss mask
│   ├── evaluate.py               # Bộ metric chuẩn Mục 13 (Aspect, Sentiment, Joint)
│   ├── pipeline.py               # Lớp ABSAPipeline đóng gói suy luận End-to-End
│   ├── predict.py                # Giao diện dòng lệnh CLI tương tác
│   ├── run_baseline.py           # Runner chạy huấn luyện và đánh giá baseline
│   ├── train.py                  # Vòng lặp huấn luyện mô hình sâu
│   └── utils.py                  # Các hàm tiện ích dùng chung (Seed, IO, Timer)
├── tests/                        # Hệ thống Unit test tự động
│   ├── test_loss_mask.py         # Kiểm thử loss mask và batch rỗng không NaN
│   ├── test_metrics.py           # Kiểm thử metric bằng tính tay Mục 6.4
│   ├── test_pipeline.py          # Kiểm thử pipeline suy luận
│   ├── test_preprocess.py        # Kiểm thử 9 trường hợp chuẩn hóa văn bản
│   └── test_schema.py            # Kiểm thử schema nhãn và chuyển đổi vector
├── CONTRIBUTIONS.md              # Bảng phân công nhiệm vụ và nhật ký đóng góp
├── LICENSE                       # Giấy phép mã nguồn mở MIT
└── requirements.txt              # Danh sách thư viện phụ thuộc
```

---

## 📊 5. Bảng Kết Quả Thực Nghiệm Trên Tập Dev

| Mô hình | Aspect Micro-F1 | Aspect Macro-F1 | Sentiment Macro-F1 | **Joint F1 (End-to-End)** |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | **0.868** | **0.822** | **0.777** | **0.751** |
| **Multi-task ViSoBERT** *(Full)* | Đang huấn luyện | Đang huấn luyện | Đang huấn luyện | Mục tiêu $\ge 0.76$ |

---

## 📜 6. Quy Tắc Học Thuật
- Toàn bộ luật tiền xử lý, từ vựng và ngưỡng chỉ được fit trên tập **Train** và kiểm chứng trên tập **Dev**.
- Tập **Test** chỉ được chấm **duy nhất 1 lần** vào thời điểm nghiệm thu cuối cùng của đề tài.
- Dự án tuân thủ bản quyền dữ liệu UIT-ViSFD cho mục đích học tập và nghiên cứu học thuật.
