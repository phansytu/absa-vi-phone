from src.data.load import load_split
from src.data.schema import from_vectors
from src.models.baseline import TfidfBaseline

# Huấn luyện nhanh baseline trên tập train
print("Đang huấn luyện baseline...")
model = TfidfBaseline().fit(load_split("data/raw", "train"))

# Các câu bạn muốn thử nghiệm
cau = [
    "Pin trâu nhưng màn hình dở, camera chụp mờ",
    "Máy ko đc mượt, giá hơi cao",
    "Giao hàng nhanh, đóng gói cẩn thận, xài rất thích",
    "Chơi game mượt, pin nhanh hết, camera chụp đẹp",
]

prob, sent = model.predict([{"text": c} for c in cau])

print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
for c, p, s in zip(cau, prob, sent):
    print(c, "->", from_vectors((p >= 0.45).astype(int).tolist(), s.tolist()))
