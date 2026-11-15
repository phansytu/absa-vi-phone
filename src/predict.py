from pathlib import Path

from src.data.load import load_split
from src.data.schema import from_vectors
from src.models.baseline import TfidfBaseline

print("Đang huấn luyện baseline từ Train.csv...")
train = load_split(Path("data/raw"), "train")
model = TfidfBaseline().fit(train)
print("Sẵn sàng. Nhập q để thoát.")

while True:
    cau = input("\nNhập câu: ").strip()

    if cau.lower() in {"q", "quit", "thoat"}:
        break
    if not cau:
        continue

    prob, sent = model.predict([{"text": cau}])
    ket_qua = from_vectors(
        (prob[0] >= 0.45).astype(int).tolist(),
        sent[0].tolist(),
    )
    print(ket_qua if ket_qua else "Không phát hiện khía cạnh")