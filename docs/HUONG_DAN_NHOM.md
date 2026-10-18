# Hướng dẫn dự án NLP-34: Phân tích cảm xúc theo khía cạnh (UIT-ViSFD)

Tài liệu này dành cho **cả nhóm**: cách cài đặt và chạy từ đầu đến cuối, ý nghĩa của từng file, định dạng dữ liệu, các khái niệm bắt buộc phải hiểu, quy tắc làm việc nhóm và cách mở rộng. Có thể đọc theo thứ tự hoặc nhảy tới mục cần.

> **Mức độ tin cậy của tài liệu:** mọi lệnh và số liệu trong tài liệu đã được chạy thử trên một bản sao sạch của dự án (Python 3.12; numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, pytest 9.1.1). Số liệu là kết quả trên **Train (cross-validation) và Dev**. **Tập Test chưa bị dùng.** Máy và phiên bản thư viện khác nhau có thể cho lệch nhẹ ở chữ số thập phân cuối.

---

## 1. Dự án làm gì

**Bài toán.** Cho một phản hồi tiếng Việt về điện thoại, hệ thống trả về danh sách các cặp `{aspect, sentiment}` cho **các khía cạnh có được nhắc tới**.

Ví dụ: `"Pin trâu nhưng màn hình dở"` → `BATTERY → Positive`, `SCREEN → Negative` (2 đầu ra).

- **10 khía cạnh:** `SCREEN`, `CAMERA`, `FEATURES`, `BATTERY`, `PERFORMANCE`, `STORAGE`, `DESIGN`, `PRICE`, `GENERAL`, `SER&ACC`.
- **3 cực tính:** `Positive`, `Negative`, `Neutral`.
- **Khía cạnh không được nhắc tới thì không xuất ra**, và **không được gán `Neutral`**. Đây là nguyên tắc nền tảng của cả dự án.
- **Dữ liệu:** UIT-ViSFD, 11.122 phản hồi, chia sẵn Train 7.786 / Dev 1.112 / Test 2.224. Chỉ dùng đúng split của tác giả.

**Ba khó khăn mà đề yêu cầu phân tích riêng:** nhiều khía cạnh trong một câu, sentiment trái chiều trong một câu, và lớp vắng mặt (câu không có khía cạnh nào, hoặc khía cạnh không được nhắc).

---

## 2. Luồng dữ liệu tổng quát

```
data/download_data.sh
        │  tải zip từ GitHub tác giả, giải nén
        ▼
data/raw/Train.csv, Dev.csv, Test.csv        ← dữ liệu gốc (KHÔNG commit)
        │  src/data/load.py (+ schema.py)    parse cột label, kiểm tra số mẫu
        ▼
data/processed/train.jsonl, dev.jsonl, test.jsonl
        │
        │  src/data/preprocess.py            chuẩn hóa văn bản (NFC, dấu thanh, teencode...)
        ▼
src/models/baseline.py                       TF-IDF + 10 bộ hiện diện + 10 bộ sentiment
        │  predict() → (xác suất hiện diện [N,10], cực tính dự đoán [N,10])
        ▼
src/run_baseline.py                          chọn ngưỡng trên Dev
        │  gọi src/evaluate.py
        ▼
outputs/metrics/baseline_dev.json            chỉ số
outputs/predictions/baseline_dev.jsonl       dự đoán từng câu
```

Các file phụ trợ nằm ngoài luồng chính: `experiments/` (ablation, kiểm kê dữ liệu), `tests/`, `docs/`.

---

## 3. Cài đặt và chạy từ đầu đến cuối

Mọi lệnh chạy **từ thư mục gốc `absa-vi-phone/`**. Nếu chạy ở thư mục khác sẽ bị lỗi `ModuleNotFoundError: No module named 'src'`.

### Bước 0. Chuẩn bị môi trường

```bash
cd absa-vi-phone
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # file do nhóm đã tạo
```

Thư viện tối thiểu cần có: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pytest`. (Phần chạy hiện tại không cần GPU, torch hay transformers. Những thứ đó chỉ cần khi làm mô hình sâu ở tuần 4–5.)

### Bước 1. Tải và chuẩn bị dữ liệu

```bash
bash data/download_data.sh
```

Kết quả mong đợi:
- Thư mục `data/raw/` có `Train.csv`, `Dev.csv`, `Test.csv`, `SHA256SUMS.txt`, `UIT-ViSFD.zip`.
- Thư mục `data/processed/` có `train.jsonl` (7.786 dòng), `dev.jsonl` (1.112 dòng), `test.jsonl` (2.224 dòng).
- Terminal in ra ba dòng mã SHA256. Khi chạy thử, ba mã bắt đầu bằng `cd4426cf…` (Train), `ab29bf9e…` (Dev), `a30901f0…` (Test). **Nếu mã của bạn khác**, tác giả có thể đã cập nhật dữ liệu: báo cả nhóm trước khi làm tiếp.

Thời gian: vài giây (tùy mạng). Script **dừng ngay** nếu số mẫu không đúng 7.786 / 1.112 / 2.224.

> **Windows:** script là bash. Dùng Git Bash hoặc WSL. Nếu không có, làm thủ công: tải `https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip`, giải nén ba file CSV vào `data/raw/`, rồi chạy `python -m src.data.load data/raw data/processed`.

### Bước 2. Chạy test

```bash
python -m pytest -q tests
```

Kết quả mong đợi: `9 passed`. Chưa pass thì **đừng làm tiếp**, vì các bước sau phụ thuộc vào `normalize()`.

### Bước 3. Kiểm kê dữ liệu thô (tùy chọn nhưng nên làm)

```bash
python experiments/audit_text.py data/raw/Train.csv
```

Mất dưới 1 giây. In ba nhóm số liệu: bất thường khoảng trắng/ký tự (A), biến thể vị trí dấu thanh (B), token không phải âm tiết tiếng Việt (C). Đây là bằng chứng cho các quyết định tiền xử lý.

### Bước 4. Chạy baseline

```bash
python -m src.run_baseline data/raw outputs
```

(hoặc không tham số: `python -m src.run_baseline`, mặc định `data/raw` và `outputs`). Mất khoảng 7 giây trên CPU.

Kết quả mong đợi khi chạy thử:

| Chỉ số (Dev) | Giá trị |
|---|---|
| Aspect micro-F1 | 0,868 |
| Aspect macro-F1 | 0,822 |
| Sentiment macro-F1 (biết trước khía cạnh) | 0,777 |
| **Joint aspect–polarity F1 (end-to-end)** | **0,751** |
| Ngưỡng được chọn trên Dev | 0,45 |
| Mẫu rỗng được dự đoán đúng là rỗng | 22% (4/18 mẫu) |

File sinh ra: `outputs/metrics/baseline_dev.json` và `outputs/predictions/baseline_dev.jsonl`.

### Bước 5. Chạy ablation tiền xử lý (tùy chọn, lâu hơn)

```bash
python experiments/ablation_baseline.py data/raw
```

Mất khoảng 2–3 phút (5-fold CV, 8 cấu hình, chỉ dùng Train). Kết quả chạy thử (joint F1 trung bình qua 5 fold ± độ lệch giữa fold):

| Cấu hình | Joint F1 |
|---|---|
| V0 token mặc định, chỉ NFC/lower | 0,733 ± 0,004 |
| V1 + sửa token pattern | 0,731 ± 0,006 |
| V2 + chuẩn dấu thanh | 0,732 ± 0,006 |
| V3 + teencode, `k` → không | 0,735 ± 0,008 |
| V4 + đánh dấu phủ định (phạm vi 2) | 0,738 ± 0,005 |
| V4 + C = 10 (**cấu hình đang dùng**) | **0,741 ± 0,005** |
| V4 + chỉ unigram | 0,701 ± 0,006 |
| V4 + n-gram (1,3) | 0,738 ± 0,005 |

**Cách đọc:** độ lệch giữa các fold (0,004–0,008) bằng hoặc lớn hơn mức tăng của từng bước làm sạch. Đừng viết trong báo cáo rằng từng bước "cải thiện" kết quả; chỉ có thể nói "không gây hại và cộng lại tăng nhẹ". Hiệu ứng lớn thật là **bigram** (bỏ đi còn 0,701).

### Bước 6. Thử với một câu tùy ý

```python
from src.data.load import load_split
from src.data.schema import from_vectors
from src.models.baseline import TfidfBaseline

model = TfidfBaseline().fit(load_split("data/raw", "train"))
cau = ["Pin trâu nhưng màn hình dở, camera chụp mờ", "Máy ko đc mượt, giá hơi cao"]
prob, sent = model.predict([{"text": c} for c in cau])
for c, p, s in zip(cau, prob, sent):
    print(c, "->", from_vectors((p >= 0.45).astype(int).tolist(), s.tolist()))
```

Chạy thử cho kết quả hợp lý (ví dụ câu đầu ra `SCREEN→Negative`, `CAMERA→Negative`, `BATTERY→Positive`). Đây mới là mô hình baseline, chưa phải demo hoàn chỉnh (demo Gradio thuộc phần việc của TV5).

---

## 4. Ý nghĩa và nội dung từng file

### 4.1. Thư mục `data/`

#### `data/download_data.sh`
- **Là gì:** script bash tải dữ liệu gốc và chuẩn bị dữ liệu xử lý.
- **Làm gì, theo thứ tự:** (1) tải `UIT-ViSFD.zip` từ GitHub của tác giả nếu chưa có `Train.csv`; (2) giải nén vào `data/raw/`; (3) ghi `SHA256SUMS.txt` để mọi người xác nhận dùng cùng một phiên bản dữ liệu; (4) gọi `python -m src.data.load` để kiểm tra số mẫu và sinh `data/processed/*.jsonl`.
- **Vì sao quan trọng:** đề yêu cầu "người khác chạy lại được" và "không phân phối lại corpus bị hạn chế". Repo **chỉ chứa script tải, không chứa dữ liệu** (đã có trong `.gitignore`).
- **Lưu ý bản quyền:** dữ liệu thuộc tác giả, dùng cho nghiên cứu/học tập; dùng thương mại phải xin phép tác giả.

### 4.2. Thư mục `src/data/`

#### `src/data/schema.py` (định nghĩa nhãn)
- **Là gì:** nơi duy nhất định nghĩa "nhãn là gì" của cả dự án. Mọi người import từ đây, **không ai tự parse nhãn**.
- **Hằng số:** `ASPECTS` (10 khía cạnh, thứ tự cố định), `POLARITIES` (Positive/Negative/Neutral), `ASPECT2ID`, `POL2ID`, `IGNORE_INDEX = -100`, `OTHERS_TOKEN`.
- **Hàm:**
  - `parse_label(raw)` đọc chuỗi nhãn gốc, trả về `({aspect: polarity}, has_others)`. Báo lỗi (`ValueError`) nếu gặp token lạ hoặc khía cạnh lặp, để lỗi dữ liệu không âm thầm lọt vào huấn luyện.
  - `to_vectors(pairs)` chuyển thành `presence[10]` ∈ {0,1} và `sentiment[10]` ∈ {0,1,2} hoặc **−100** nếu khía cạnh vắng.
  - `from_vectors(presence, sentiment)` chuyển ngược về `[{aspect, sentiment}]`, **chỉ cho khía cạnh có mặt**. Dùng cho đầu ra cuối của pipeline/demo.
- **Định dạng nhãn gốc:** `{CAMERA#Positive};{BATTERY#Negative};{OTHERS};`
- **Về `{OTHERS}`:** không phải khía cạnh trong 10 khía cạnh, không có cực tính. Được lưu thành cờ `has_others` và **không đưa vào đầu ra**. Phản hồi chỉ có `{OTHERS}` nghĩa là **không có khía cạnh nào** (115 mẫu Train, 18 Dev, 37 Test). Đây là trường hợp hợp lệ; đầu ra đúng là danh sách rỗng.

#### `src/data/load.py` (đọc dữ liệu gốc)
- **Là gì:** đọc ba file CSV gốc và biến mỗi dòng thành một bản ghi chuẩn.
- **Hàm chính:** `load_split(raw_dir, split)` với `split` ∈ `train`/`dev`/`test`, trả về danh sách `dict` (xem mục 5.2). `main(raw_dir, out_dir)` ghi ra `*.jsonl` và **kiểm tra số mẫu** (`EXPECTED`: 7.786 / 1.112 / 2.224).
- **Chạy riêng:** `python -m src.data.load data/raw data/processed`.
- **Ai dùng:** `run_baseline.py`, `ablation_baseline.py`, và mọi mô hình sau này.

#### `src/data/preprocess.py` (tiền xử lý văn bản)
- **Là gì:** một hàm chuẩn hóa văn bản **dùng chung cho mọi mô hình và demo**. Không ai tự viết bản riêng.
- **`normalize(text, tone=True, teencode=True, repeat=True)`** làm theo thứ tự:
  1. Chuẩn Unicode NFC và chữ thường (3,1% văn bản gốc bị tách dấu, không NFC).
  2. Gộp khoảng trắng và xuống dòng thừa.
  3. Rút ký tự lặp từ 3 lần trở lên (`đẹppppp` → `đẹp`).
  4. Chuẩn vị trí dấu thanh **chỉ ở cuối từ** (`hoà` → `hòa`, `khoẻ` → `khỏe`). Không đổi `toàn`, `thoại`, `khoảng` (đã từng là lỗi, có test hồi quy).
  5. Gộp teencode: `ko`, `k` (đứng riêng), `kg`, `hok`… → `không`; `đc`, `dc` → `được`; `đt`, `dt` → `điện thoại`; `sp`, `nv`, `mn`, `vs`… `600k` (số liền trước `k`) được giữ nguyên.
- **`mark_negation(text, scope=3)`**: thêm tiền tố `n_` cho vài từ đứng sau từ phủ định (`không tốt` → `không n_tốt`), dừng ở dấu câu hoặc từ đối lập (`nhưng`, `mà`). **Chỉ dùng cho TF-IDF**, không dùng cho transformer.
- **Giữ nguyên có chủ đích:** từ phủ định, dấu câu, emoji, chữ số, dấu tiếng Việt, stopword.
- Cơ sở của từng luật nằm ở `docs/SO_TAY_TIEN_XU_LY.md`.

### 4.3. Thư mục `src/models/`

#### `src/models/baseline.py` (mô hình đối chứng)
- **Là gì:** baseline theo đề: **TF-IDF + bộ phân loại riêng cho từng khía cạnh**.
- **Cấu trúc:** một bộ TF-IDF (từ 1–2 gram, giữ token 1 ký tự và dấu câu) **fit chỉ trên Train**; **10 bộ hiện diện** (Logistic Regression nhị phân, `class_weight="balanced"`, C=10); **10 bộ sentiment** (Logistic Regression 3 lớp, C=10), mỗi bộ chỉ huấn luyện trên các mẫu **có mặt** khía cạnh đó. Nhờ vậy khía cạnh vắng không bao giờ bị gán `Neutral`.
- **Lớp `TfidfBaseline`:** `fit(rows)` huấn luyện; `predict(rows)` trả về `(presence_prob[N,10], sent_pred[N,10])`.
- **Hàm `prep(text, neg_scope=2)`:** `normalize()` rồi `mark_negation()`.
- **Hằng số `TOKEN_PATTERN`:** giữ lại token 1 ký tự và dấu câu. Cấu hình mặc định của sklearn làm rơi `k` (= "không", 1.223 lần) và các số 1–5.
- **Quan trọng:** `predict()` trả cực tính cho **cả 10 khía cạnh** (kể cả khía cạnh dự đoán vắng). Việc chọn khía cạnh nào xuất ra thuộc về ngưỡng, áp dụng ở bên ngoài.

#### Hợp đồng giao diện cho mô hình mới (rất quan trọng cho TV3, TV4)
Mọi mô hình mới (multitask, separate, aspect-conditioned) nên có hàm `predict(rows)` trả về cùng dạng: **`(presence_prob [N,10], sent_pred [N,10])`**. Khi đó `evaluate.py` và `run_baseline.py`-style script dùng lại được, và các mô hình được so sánh **cùng một cách tính**.

### 4.4. File đánh giá và chạy

#### `src/evaluate.py` (metric dùng chung)
- **Là gì:** hàm `evaluate(gold_rows, presence_pred, sent_pred)` tính tất cả chỉ số theo mục 13 của đề. **Mọi mô hình dùng đúng hàm này.**
- **Chỉ số trả về:**
  - `aspect_micro_f1`, `aspect_macro_f1` (+ F1 từng khía cạnh): phát hiện khía cạnh trên ma trận hiện diện, tính trên **mọi mẫu, kể cả mẫu rỗng**.
  - `sentiment_macro_f1_given_gold_aspect`: macro-F1 cực tính trên các (mẫu, khía cạnh) mà **gold có mặt**. Đây là số **chẩn đoán** (biết trước khía cạnh), **không phải end-to-end**.
  - `joint_precision`, `joint_recall`, `joint_f1` (+ theo khía cạnh): **chỉ số end-to-end**. So tập cặp `(aspect, polarity)` dự đoán với tập gold; một cặp chỉ đúng khi **đúng cả khía cạnh lẫn cực tính**.
  - `n_samples`, `n_gold_present_pairs`, `n_gold_empty`, `gold_empty_predicted_empty_rate`: số mẫu và việc xử lý lớp vắng mặt.
- **Quy tắc đã chốt:** đầu ra rỗng hoặc dự đoán thừa **không bị loại bỏ**; dự đoán thừa tính là FP.

#### `src/run_baseline.py` (chạy baseline từ đầu đến cuối)
- **Là gì:** script gắn các mảnh lại: fit trên Train → dự đoán Dev → chọn ngưỡng → đánh giá → lưu.
- **Chọn ngưỡng:** một ngưỡng **chung** thử từ 0,20 đến 0,80 (bước 0,05), chọn ngưỡng cho **joint F1 trên Dev cao nhất**. Dùng ngưỡng chung vì ngưỡng riêng từng khía cạnh dễ quá khớp Dev (khía cạnh `STORAGE` chỉ có 14 mẫu Dev).
- **Đầu ra:** `outputs/metrics/baseline_dev.json`, `outputs/predictions/baseline_dev.jsonl`.
- **Chỉ chấm Dev. Chưa và không được chấm Test ở giai đoạn này.**

### 4.5. Thư mục `experiments/`

#### `experiments/ablation_baseline.py`
- **Là gì:** bộ khung thí nghiệm: hàm `cv(texts, ...)` chạy 5-fold CV **chỉ trên Train** (vectorizer fit lại trong từng fold), hàm `show(...)` in kết quả. Phần `__main__` chạy 8 cấu hình ở mục 3, bước 5.
- **Dùng khi:** muốn kiểm chứng một thay đổi tiền xử lý/siêu tham số. Sao chép cách gọi trong `__main__`, **chỉ đổi một yếu tố mỗi lần**.

#### `experiments/audit_text.py`
- **Là gì:** kiểm kê dữ liệu thô của Train, dựa trên số đếm thật.
- **Nhóm A:** khoảng trắng và ký tự (không NFC, xuống dòng, 2 khoảng trắng, khoảng trắng quanh dấu câu…).
- **Nhóm B:** biến thể vị trí dấu thanh, kèm tỉ lệ (dạng ít gặp chỉ 0,087% token, 2,92% văn bản).
- **Nhóm C:** token không phải âm tiết tiếng Việt hợp lệ, và nghi ngờ dính từ.
- **Lưu ý:** hai dòng "khoảng trắng trước/sau dấu câu" là giới hạn trên (có lẫn emoticon). Phải đọc mẫu trước khi kết luận.

### 4.6. Thư mục `tests/`

#### `tests/test_preprocess.py` (9 test)
Bảo vệ `normalize()` và `mark_negation()`: giữ phủ định; gộp `ko`/`k` → `không`; giữ `600k`; rút ký tự lặp; giữ chữ lặp hợp lệ (`boong`); chuẩn NFC; chuẩn dấu thanh ở cuối từ (`hoà` → `hòa`) mà vẫn giữ `quý`; **không đổi** `toàn`, `thoại`, `khoảng`…; phạm vi phủ định dừng ở `nhưng`. Mỗi luật tiền xử lý mới **phải** có test kèm phản ví dụ.

### 4.7. Thư mục `docs/`

| File | Nội dung |
|---|---|
| `SO_TAY_TIEN_XU_LY.md` | Phương pháp quan sát, chọn lọc và kiểm chứng tiền xử lý; ma trận giữ/bỏ; ablation; tìm chỗ chưa tối ưu; nhật ký quyết định mẫu |
| `HUONG_DAN_NHOM.md` | Tài liệu này |

### 4.8. File gốc dự án

| File | Ý nghĩa |
|---|---|
| `.gitignore` | Loại `data/raw/`, `data/processed/`, kết quả và checkpoint, cache. Không đẩy dữ liệu của tác giả lên repo |
| `requirements.txt` | Danh sách thư viện (do nhóm tạo) |
| `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py` | File rỗng để Python coi các thư mục là package; **cần giữ** vì code dùng `from src...` |

---

## 5. Định dạng dữ liệu

### 5.1. CSV gốc (`data/raw/*.csv`)

Năm cột: `index`, `comment` (văn bản phản hồi), `n_star` (số sao 1–5), `date_time`, `label` (chuỗi nhãn).

> `n_star` **không được dùng làm đầu vào mô hình**: đề quy định đầu vào là văn bản phản hồi và tập khía cạnh. Có thể dùng riêng để phân tích.

### 5.2. Bản ghi đã xử lý (`data/processed/*.jsonl`, và kết quả của `load_split`)

Mỗi dòng một đối tượng JSON:

| Trường | Ý nghĩa |
|---|---|
| `id` | Dạng `dev-0`, `train-15`… (split + chỉ số gốc) |
| `text` | Văn bản gốc, **chưa chuẩn hóa** (chuẩn hóa làm khi cần, bằng `normalize()`) |
| `n_star` | Số sao |
| `pairs` | `{khía cạnh: cực tính}` của các khía cạnh có mặt, ví dụ `{"GENERAL": "Positive"}` |
| `has_others` | Có token `{OTHERS}` hay không |
| `presence` | 10 số 0/1 theo thứ tự `ASPECTS` |
| `sentiment` | 10 số: 0=Positive, 1=Negative, 2=Neutral, hoặc **−100** nếu khía cạnh vắng |

Ví dụ: `"presence": [0,0,0,0,0,0,0,0,1,1]`, `"sentiment": [-100,...,-100,0,0]` nghĩa là có `GENERAL` và `SER&ACC`, cả hai Positive.

### 5.3. Dự đoán (`outputs/predictions/*.jsonl`)

`id`, `text`, `pred` (danh sách `{aspect, sentiment}` của các khía cạnh dự đoán có mặt), `prob` (xác suất hiện diện của từng khía cạnh).

### 5.4. Chỉ số (`outputs/metrics/*.json`)

Các trường của `evaluate()` (mục 4.4), thêm `threshold`, `split`, `model`.

---

## 6. Khái niệm cốt lõi mà mọi thành viên phải hiểu

### 6.1. Hai bài toán con
1. **Hiện diện (presence):** với mỗi khía cạnh, câu có nhắc tới không? (nhị phân)
2. **Cực tính (sentiment):** nếu có, khen, chê hay trung tính? (3 lớp)

Hệ thống end-to-end = hiện diện **rồi** cực tính trên các khía cạnh được dự đoán là có mặt.

### 6.2. Vì sao có nhãn −100
Khía cạnh vắng **không có cực tính**. Nhãn `−100` báo cho hàm loss (`ignore_index=-100`) biết "bỏ qua vị trí này". Nếu gán `Neutral` cho khía cạnh vắng, mô hình học sai rằng "không nhắc" là "trung tính".

> **Cảnh báo cho TV3:** nếu một batch **không có khía cạnh nào hiện diện** (Train có 115 mẫu như vậy), loss sentiment trung bình trên tập rỗng có thể ra NaN. Cần viết test cho trường hợp này (ví dụ chia tổng cho `max(số phần tử hợp lệ, 1)`).

### 6.3. Chỉ số đúng và chỉ số "chẩn đoán"
- **Joint F1** là chỉ số end-to-end: phạt cả khía cạnh bỏ sót, khía cạnh dự đoán thừa, và cực tính sai.
- **Sentiment macro-F1 biết trước khía cạnh** chỉ cho biết phần cực tính tốt đến đâu. Nếu chỉ báo số này rồi gọi là "end-to-end" thì mắc đúng lỗi mà đề cảnh báo.

### 6.4. Ví dụ tính tay (đã kiểm chứng bằng `evaluate()`)

Hai câu:
- Câu 1: gold = {BATTERY→Positive, SCREEN→Negative}; dự đoán = {BATTERY→Positive, SCREEN→**Positive**, CAMERA→Positive}.
- Câu 2: gold rỗng (chỉ có `OTHERS`); dự đoán rỗng.

Joint (cặp đúng cả khía cạnh lẫn cực tính): TP = 1 (BATTERY); FP = 2 (SCREEN sai cực tính, CAMERA thừa); FN = 1 (SCREEN gold không được khớp). Precision = 1/3, Recall = 1/2, **Joint F1 = 0,4**.

Aspect micro-F1 (chỉ xét có mặt hay không): phát hiện đúng BATTERY và SCREEN (TP = 2), thừa CAMERA (FP = 1), không bỏ sót → **0,8**. Một câu dự đoán đúng khía cạnh nhưng sai cực tính vẫn tăng aspect F1 mà không tăng joint F1.

### 6.5. Ngưỡng
`presence_prob ≥ ngưỡng` thì xem khía cạnh là có mặt. Ngưỡng cao ⇒ ít dự đoán thừa nhưng bỏ sót nhiều hơn. Ngưỡng **chọn trên Dev**, không chọn trên Test.

---

## 7. Quy tắc làm việc nhóm

### 7.1. Quy tắc dữ liệu và đánh giá (vi phạm là sai kết quả)

1. **Giữ nguyên split của tác giả.** Không chia lại.
2. **Mọi thứ học từ dữ liệu chỉ fit trên Train:** từ vựng, IDF, siêu tham số, ngưỡng (ngưỡng chọn bằng Dev).
3. **Test chỉ chấm một lần**, sau khi cả nhóm xác nhận đã chốt cấu hình (tuần 7). **TV2 giữ quyền chấm Test.** File `data/processed/test.jsonl` tồn tại nên về kỹ thuật ai cũng đọc được; nhóm cam kết không dùng nó để chọn mô hình, ngưỡng hay luật tiền xử lý.
4. **Mọi mô hình dùng chung** `normalize()` (với mô hình sâu thì thêm bước riêng của tokenizer, ví dụ tách từ cho PhoBERT) và `evaluate()`.
5. **So sánh phải cùng split, cùng quy tắc metric, cùng ngân sách tuning.**
6. **Mô hình có ngẫu nhiên:** chạy 3 seed, báo mean ± std.
7. **Không báo chênh lệch nhỏ hơn độ lệch giữa các fold/seed như thể là thắng lợi.**
8. **Không giả định mô hình chính thắng baseline** và không ghi con số hiệu năng trước khi chạy thực nghiệm.

### 7.2. Quy tắc code
- Mỗi mô hình mới nằm trong `src/models/<tên>.py`, tuân theo hợp đồng giao diện ở mục 4.3.
- Mỗi luật tiền xử lý mới: có tần suất đo được, đã đọc ngữ cảnh, test có phản ví dụ, và **đã áp dụng lên toàn bộ từ vựng rồi đọc danh sách từ bị đổi** (xem mục 3.6 của sổ tay tiền xử lý).
- Cấu hình thí nghiệm lưu thành file trong `configs/` (ghi seed và siêu tham số); kết quả lưu vào `outputs/metrics/`, dự đoán vào `outputs/predictions/`, tên theo dạng `<mô hình>_<tập>` hoặc `<mô hình>_s<seed>`.
- Mỗi người một nhánh git, merge qua pull request, người rà soát khác người viết (theo vòng tròn trong file phân công).

### 7.3. Cách thêm một mô hình mới (ví dụ `multitask.py`)
1. Viết `src/models/multitask.py` với `fit(rows)` và `predict(rows) → (presence_prob, sent_pred)`.
2. Viết script chạy (theo mẫu `run_baseline.py`): fit Train, dự đoán Dev, chọn ngưỡng Dev, gọi `evaluate()`, lưu metrics và predictions.
3. Viết test (đặc biệt: loss mask, batch không có khía cạnh nào, hình dạng tensor).
4. So sánh với baseline bằng **đúng cùng** `evaluate()`.

---

## 8. Việc chưa có trong repo và người phụ trách

| File dự kiến | Nội dung | Phụ trách (theo phân công 5 người) |
|---|---|---|
| `src/data/dataset.py` | Dataset PyTorch, tokenize, collate, nhãn −100 | TV1 |
| `src/models/multitask.py`, `src/train.py`, `configs/multitask.yaml` | Encoder đa nhiệm, huấn luyện, 3 seed | TV3 |
| `src/models/separate.py`, `src/models/conditioned.py` | Đối chứng và mở rộng aspect-conditioned | TV4 |
| `tests/test_schema.py`, `tests/test_evaluate.py` | Test schema và metric bằng ví dụ tính tay (mục 6.4 dùng được ngay) | TV1, TV2 |
| `src/pipeline.py`, `tests/test_pipeline.py`, `demo/app.py` | Text → `[{aspect, sentiment}]`, test trường hợp biên, demo Gradio | TV5 |
| `notebooks/01_eda.ipynb`, `02_baseline_check.ipynb`, `03_error_analysis.ipynb` | Khảo sát dữ liệu, kiểm tra tay, phân tích lỗi | TV1, TV2, TV5 |
| `README.md` (gốc), `CONTRIBUTIONS.md` | Hướng dẫn chạy lại, nhật ký đóng góp và khảo sát | TV4 |
| `.github/workflows/ci.yml` | CI nhẹ: pytest + ruff | TV4 |

---

## 9. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy không phải từ thư mục gốc, hoặc thiếu `__init__.py` | `cd absa-vi-phone`; chạy bằng `python -m src.run_baseline`, không phải `python src/run_baseline.py` |
| `FileNotFoundError: data/raw/Train.csv` | Chưa chạy bước tải dữ liệu | `bash data/download_data.sh` |
| `curl` lỗi / không tải được | Mạng chặn GitHub | Tải zip thủ công (mục 3, bước 1) |
| `bash: command not found` trên Windows | Không có bash | Git Bash hoặc WSL, hoặc làm thủ công |
| `AssertionError` về số mẫu khi tải | Dữ liệu tải về khác phiên bản chuẩn | So SHA256; báo cả nhóm; không tự sửa số `EXPECTED` |
| `UnicodeDecodeError` khi đọc CSV | Mở bằng sai mã hóa | Luôn dùng `encoding="utf-8"` |
| `ConvergenceWarning` | Logistic Regression chưa hội tụ trong giới hạn lặp | Thường vô hại với cấu hình hiện tại; nếu bật, tăng `max_iter` |
| Kết quả lệch vài phần nghìn so với tài liệu | Phiên bản thư viện khác | Bình thường; lệch lớn thì kiểm tra lại các bước |
| `ValueError: Token nhãn không hợp lệ` | Dữ liệu có nhãn lạ | Không bỏ qua; báo TV1 để xem xét schema |

---

## 10. Danh sách kiểm tra cho thành viên mới (khoảng 15 phút)

- [ ] Cài môi trường, `pip install -r requirements.txt`.
- [ ] `bash data/download_data.sh`, so SHA256 với nhóm.
- [ ] `python -m pytest -q tests` → 9 passed.
- [ ] `python -m src.run_baseline` → joint F1 Dev khoảng 0,75.
- [ ] Chạy đoạn code ở mục 3 (bước 6) với một câu của chính bạn.
- [ ] Đọc `src/data/schema.py` và giải thích được vì sao có `−100`.
- [ ] Đọc mục 6 (khái niệm cốt lõi) và tự tính lại ví dụ 6.4.
- [ ] Đọc mục 7.1 (quy tắc dữ liệu và đánh giá).
- [ ] Đọc ít nhất mục 3 và mục 4 của `docs/SO_TAY_TIEN_XU_LY.md`.

---

## 11. Giới hạn của kết quả hiện có (cần nói thật khi báo cáo)

- Các số liệu trên là kết quả của **một cấu hình baseline xác định** trên **Train (CV)** và **Dev**. Chúng chưa phải kết quả cuối của đề tài; số chính thức chỉ lấy trên Test, một lần, sau khi chốt cấu hình.
- Cấu hình baseline được chọn sau khi thử nhiều biến thể trên cùng một bộ CV, nên con số CV hơi lạc quan.
- Baseline TF-IDF đạt khoảng 0,75 joint F1 trên Dev, **khá mạnh**. Mô hình sâu có thể chỉ hơn ít, hoặc kém ở vài khía cạnh. Đề không yêu cầu mô hình chính thắng; yêu cầu là so sánh có kiểm soát và phân tích có minh chứng.
- Các khía cạnh ít mẫu (`STORAGE`: khoảng 91 mẫu Train, 14 mẫu Dev) cho số đo không ổn định; luôn báo kèm số mẫu.
- Lớp vắng mặt vẫn là điểm yếu: baseline chỉ nhận ra khoảng 17–28% mẫu không có khía cạnh nào (tùy ngưỡng; chỉ 18 mẫu Dev nên rất bất định).
- Chưa kiểm tra **gần-trùng** giữa Train/Dev/Test (mới kiểm tra khớp tuyệt đối: 0 câu trùng). Đây là việc cần làm ở tuần 2–3.
- Hành vi của tokenizer (PhoBERT, ViSoBERT) với văn bản đã chuẩn hóa **chưa được đo**; cần làm ở tuần 4.
