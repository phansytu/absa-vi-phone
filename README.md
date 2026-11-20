# 📱 Phân Tích Cảm Xúc Theo Khía Cạnh Cho Đánh Giá Điện Thoại Tiếng Việt (ABSA UIT-ViSFD)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án nghiên cứu và triển khai hệ thống **Aspect-Based Sentiment Analysis (ABSA)** cho phản hồi điện thoại bằng dữ liệu UIT‑ViSFD gồm 11.122 câu. Hệ thống phát hiện 10 khía cạnh và gán Positive, Negative hoặc Neutral cho từng khía cạnh được nhắc tới.

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
(TF-IDF + 10 LR Hiện diện + 10 LR Cảm xúc)   (PhoBERT vinai/phobert-base + 2 đầu ra đa nhiệm)
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
```

*(Khuyến nghị nếu máy có GPU NVIDIA, hãy cài PyTorch hỗ trợ CUDA để tăng tốc huấn luyện mô hình sâu: `pip install torch --index-url https://download.pytorch.org/whl/cu121`)*

---

### Bước 2: Chuẩn bị và chuyển đổi dữ liệu

Dữ liệu thô đã được lưu tại `data/raw/` (`Train.csv`, `Dev.csv`, `Test.csv`). Chạy lệnh sau để chuyển đổi sang định dạng JSONL chuẩn hóa:

```powershell
python -m src.data.load data/raw data/processed
```
*Kết quả:* Sinh ra `train.jsonl` (7.786 dòng), `dev.jsonl` (1.112 dòng), `test.jsonl` (2.224 dòng) tại `data/processed/`.

Chuẩn bị dữ liệu đã tách từ cho PhoBERT:

```powershell
python -m src.data.segment --in-dir data/processed --out-dir data/processed_segmented
python -m src.data.prepare_final --in-dir data/processed_segmented --out-dir data/processed_final --no-tone
```

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

Lệnh trên chọn ngưỡng trên Dev và lưu kết quả Dev. Chỉ sau khi chốt cấu hình mới chấm Test:

```powershell
python -m src.run_baseline data/raw outputs --test
```

Kết quả Dev V2C hiện được lưu ở `outputs/metrics/baseline_dev_v2c.json`; ngưỡng riêng theo khía cạnh được chọn trên Dev. Kết quả Test đã chấm một lần và được lưu tại `outputs/metrics/baseline_test_v2c.json`.

---

### Bước 5: Chạy Thí nghiệm Ablation Tiền xử lý (5-Fold CV trên Train)

```powershell
python experiments/ablation_baseline.py data/raw
```

So sánh 8 cấu hình tiền xử lý để đánh giá sự đóng góp của chuẩn NFC, vị trí dấu thanh, từ điển teencode, tiền tố phủ định `n_` và n-gram.

---

### Bước 6: Huấn luyện mô hình đa nhiệm PhoBERT

Huấn luyện mô hình mạng nơ-ron sâu dùng cấu hình `configs/multitask.yaml`:

```powershell
# Huấn luyện chính thức Seed 42
python -m src.train --config configs/multitask.yaml --seed 42

# (Tùy chọn, tốn thời gian) Chạy 3 seed (42, 43, 44):
powershell -ExecutionPolicy Bypass -File scripts/run_seeds.ps1
```

---

### Bước 7: Thực Hành Dự Đoán (Inference)

#### Cách 1: Giao diện dòng lệnh tương tác trực tiếp (CLI REPL)
```powershell
python -m src.demo --model multitask --detailed
```
Nhập câu sau dấu nhắc; gõ `q` để thoát. Câu đầu tiên có thể mất thêm thời gian nạp PhoBERT. Dùng `--model baseline` để chạy TF-IDF. Checkpoint PhoBERT phải tồn tại tại `outputs/checkpoints/multitask_seed42.pt`.
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
│   └── multitask.yaml            # Cấu hình Multi-task PhoBERT
├── data/
│   ├── download_data.sh          # Script tải dữ liệu UIT-ViSFD từ tác giả
│   ├── raw/                      # Dữ liệu gốc: Train.csv, Dev.csv, Test.csv (Không commit)
│   └── processed/                # Dữ liệu đã chuyển đổi: train/dev/test.jsonl
├── demo/
│   └── app.py                    # Ứng dụng Web UI tương tác bằng Gradio
├── docs/
│   ├── HUONG_DAN_NHOM.md         # Sổ tay hướng dẫn chi tiết quy tắc làm việc nhóm
├── experiments/
│   ├── audit_text.py             # Script kiểm kê và đo lường bất thường dữ liệu thô
│   └── ablation_baseline.py      # Thí nghiệm 5-fold CV khảo sát tác động tiền xử lý
├── notebooks/                    # Jupyter Notebooks nghiên cứu & trực quan hóa
│   ├── 01_eda.ipynb              # Khám phá phân bố dữ liệu, khía cạnh & cực tính
│   ├── 02_baseline_check.ipynb   # Đánh giá và kiểm tra kết quả Baseline trên Dev
│   └── 03_error_analysis.ipynb   # Phân tích lỗi đa khía cạnh, trái chiều & mẫu rỗng
├── outputs/                      # Thư mục lưu kết quả thực nghiệm
│   ├── checkpoints/              # File trọng số mô hình (.pt)
│   ├── metrics/                  # Metrics và bảng/biểu đồ so sánh cuối
│   ├── predictions/              # Dự đoán Dev/Test theo ID (.jsonl)
│   ├── analysis_phobert/         # Phân tích lỗi trên Dev
│   ├── analysis_baseline/        # Phân tích lỗi trên Dev
│   └── archive/                  # Kết quả thí nghiệm cũ, không dùng báo cáo cuối
├── report/                       # Báo cáo Markdown và PDF
├── slides/                       # Slide thuyết trình PowerPoint
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
│   │   └── multitask.py          # PhoBERT Multi-task + loss mask
│   ├── evaluate.py               # Bộ metric chuẩn Mục 13 (Aspect, Sentiment, Joint)
│   ├── pipeline.py               # Lớp ABSAPipeline đóng gói suy luận End-to-End
│   ├── demo.py                   # Giao diện dòng lệnh nhập câu mới
│   ├── predict.py                # Sinh dự đoán trên split để đánh giá
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

## 📊 5. Kết quả cuối trên tập Test

| Mô hình | Aspect Micro-F1 | Aspect Macro-F1 | Sentiment Macro-F1 | **Joint F1** |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF baseline | 0.8606 | 0.8344 | 0.7720 | 0.7427 |
| PhoBERT MultiTask, seed 42 | **0.9002** | **0.8672** | **0.8282** | **0.8141** |

Hai mô hình được chấm trên cùng 2.224 câu Test; ngưỡng được chọn trên Dev. PhoBERT đạt Joint F1 cao hơn 0.0713 điểm tuyệt đối. Kết quả PhoBERT hiện có một seed nên chưa đo được độ dao động giữa các lần huấn luyện. `STORAGE` là khía cạnh yếu nhất của PhoBERT theo Joint F1 (0.3750); TF-IDF đạt 0.5424 ở khía cạnh này.

---

## 📜 6. Quy Tắc Học Thuật
- Toàn bộ luật tiền xử lý, từ vựng và ngưỡng chỉ được fit trên tập **Train** và kiểm chứng trên tập **Dev**.
- Chọn checkpoint và ngưỡng trên **Dev**; chỉ chấm **Test** sau khi chốt cấu hình. Không dùng điểm Test để tinh chỉnh.
- Dự án tuân thủ bản quyền dữ liệu UIT-ViSFD cho mục đích học tập và nghiên cứu học thuật.

## Nguồn dữ liệu và mô hình

- Dữ liệu: [UIT‑ViSFD chính thức](https://github.com/LuongPhan/UIT-ViSFD). Repo không chứa bản sao dữ liệu thô; hãy tải theo điều khoản của tác giả và không phân phối lại khi chưa được phép.
- Encoder: [PhoBERT của VinAI](https://github.com/VinAIResearch/PhoBERT), checkpoint `vinai/phobert-base`.
- Công thức đánh giá F1: [scikit‑learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Checkpoint `.pt` không được Git theo dõi. Trên bản clone mới, cần chuẩn bị dữ liệu và chạy huấn luyện PhoBERT trước khi chọn `--model multitask`.

## Tái lập và trạng thái bàn giao

- Kết quả cuối hiện có là một lần huấn luyện PhoBERT với seed 42. Tập Test đã được chấm cho cả hai mô hình; không chạy lại Test để thay đổi lựa chọn mô hình/ngưỡng.
- Chạy kiểm thử bằng `python -m pytest -q tests`.
- Dự đoán PhoBERT trên Dev: `python -m src.predict --seed 42 --split dev --thresholds ckpt`.
- Kết quả Test, dự đoán theo mã mẫu, bảng so sánh và biểu đồ nằm trong `outputs/`. Phân tích lỗi chỉ dùng Dev.
- Checkpoint seed 42 không được commit vì dung lượng lớn; để chạy demo PhoBERT trên bản clone mới, lấy checkpoint từ kênh bàn giao nội bộ hoặc huấn luyện lại. Không phát tán dữ liệu gốc nếu điều khoản bộ dữ liệu không cho phép.
- Sổ tay nhóm còn một số mục kế hoạch cũ; ghi chú ở cuối sổ tay mô tả trạng thái hiện tại.
