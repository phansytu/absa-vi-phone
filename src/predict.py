"""Giao diện dòng lệnh (CLI) để dự đoán cảm xúc theo khía cạnh trên văn bản tùy ý.

Cách dùng:
1. Dự đoán 1 câu trực tiếp:
   python -m src.predict --text "Pin trâu nhưng màn hình hơi ám vàng, chụp ảnh đẹp"

2. Chế độ tương tác liên tục (gõ và xem kết quả trực tiếp):
   python -m src.predict --interactive
   (hoặc chỉ cần chạy: python -m src.predict)

3. Dự đoán theo file văn bản (mỗi dòng một câu):
   python -m src.predict --file inputs.txt --output results.json

4. Lựa chọn mô hình:
   python -m src.predict --model multitask --checkpoint outputs/checkpoints/multitask_seed42.pt
   python -m src.predict --model baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import ABSAPipeline, ASPECT_METADATA, POLARITY_METADATA


def print_banner(model_name: str, device: str):
    print("=" * 70)
    print("      HỆ THỐNG PHÂN TÍCH CẢM XÚC THEO KHÍA CẠNH (ABSA VI-PHONE)")
    print(f"      • Mô hình đang dùng: [{model_name.upper()}]  |  Thiết bị: [{device}]")
    print("=" * 70)


def format_prediction_result(res: dict, detailed: bool = False) -> str:
    lines = []
    lines.append(f"\n📝 Câu gốc       : {res['text']}")
    lines.append(f"🔍 Đã chuẩn hóa  : {res['normalized']}")
    
    preds = res["predictions"]
    if not preds:
        lines.append("📌 Kết quả       : [KHÔNG CÓ KHÍA CẠNH NÀO ĐƯỢC NHẮC TỚI / OTHERS]")
        lines.append("                  (Tương ứng với nhãn rỗng hợp lệ theo bài toán)")
    else:
        lines.append(f"📌 Phát hiện được: {len(preds)} khía cạnh")
        lines.append("-" * 70)
        lines.append(f"{'Khía cạnh (Aspect)':<22} | {'Giải nghĩa':<18} | {'Cực tính (Sentiment)':<22} | {'Độ tin cậy'}")
        lines.append("-" * 70)
        for p in preds:
            asp = p["aspect"]
            asp_vn = p["aspect_vn"]
            pol = p["sentiment"]
            pol_info = POLARITY_METADATA.get(pol, {})
            pol_str = f"{pol_info.get('symbol', '')} {pol} ({pol_info.get('vn', '')})"
            conf = f"{p['confidence']:.2%}"
            lines.append(f"{asp:<22} | {asp_vn:<18} | {pol_str:<22} | {conf}")
        lines.append("-" * 70)

    if detailed and "aspect_details" in res:
        lines.append("\n📊 Bảng xác suất chi tiết 10 khía cạnh:")
        lines.append(f"{'Khía cạnh':<15} | {'Xác suất':<10} | {'Ngưỡng':<10} | {'Có mặt?':<10} | {'Cảm xúc dự đoán'}")
        lines.append("-" * 65)
        for asp, det in res["aspect_details"].items():
            pres_str = "CÓ" if det["is_present"] else "KHÔNG"
            sent_str = det["sentiment"] or "-"
            lines.append(
                f"{asp:<15} | {det['presence_prob']:<10.4f} | {det['threshold']:<10.4f} | {pres_str:<10} | {sent_str}"
            )
        lines.append("-" * 65)

    return "\n".join(lines)


def run_interactive(pipeline: ABSAPipeline, detailed: bool = False):
    print_banner(pipeline.active_model, pipeline.device)
    print("\n💡 Hướng dẫn: Nhập bất kỳ câu đánh giá nào về điện thoại rồi nhấn Enter.")
    print("   Gõ 'q', 'exit' hoặc 'quit' để kết thúc chương trình.\n")

    examples = [
        "Pin trâu nhưng màn hình xấu, dùng lâu hơi giật lag",
        "Máy xài rất mượt, giá 600k rẻ bất ngờ, nhân viên tư vấn nhiệt tình",
        "Camera chụp mờ, loa ngoài rè, pin tụt nhanh",
        "Giao hàng nhanh, đóng gói cẩn thận",
    ]
    print("Ví dụ câu bạn có thể thử:")
    for i, ex in enumerate(examples, 1):
        print(f"  [{i}] {ex}")
    print()

    while True:
        try:
            user_input = input("👉 Nhập câu đánh giá: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "exit", "quit"):
                print("\nCảm ơn bạn đã sử dụng hệ thống! Tạm biệt.")
                break

            result = pipeline.predict(user_input, detailed=True)
            print(format_prediction_result(result, detailed=detailed))
            print()
        except KeyboardInterrupt:
            print("\nĐã ngắt chương trình.")
            break
        except Exception as e:
            print(f"❌ Có lỗi xảy ra trong quá trình dự đoán: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dự đoán ABSA trên phản hồi điện thoại tiếng Việt.")
    parser.add_argument("--text", type=str, help="Văn bản cần phân tích trực tiếp")
    parser.add_argument("--file", type=str, help="File chứa danh sách câu cần phân tích (mỗi dòng một câu)")
    parser.add_argument("--output", type=str, help="Đường dẫn file lưu kết quả JSON (nếu dùng --file)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Chạy chế độ tương tác gõ câu liên tục")
    parser.add_argument("--model", choices=["auto", "multitask", "baseline"], default="auto", help="Loại mô hình")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/multitask_seed42.pt", help="Đường dẫn file .pt")
    parser.add_argument("--threshold", type=float, default=None, help="Ghi đè ngưỡng phát hiện khía cạnh")
    parser.add_argument("--detailed", "-d", action="store_true", help="In thông tin xác suất chi tiết")
    args = parser.parse_args()

    # Khởi tạo pipeline
    print("⏳ Đang nạp mô hình...")
    pipeline = ABSAPipeline(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    print(f"✅ Mô hình sẵn sàng: {pipeline.active_model.upper()} trên thiết bị [{pipeline.device}]")

    # 1. Dự đoán theo 1 câu cụ thể
    if args.text:
        result = pipeline.predict(args.text, detailed=True)
        print(format_prediction_result(result, detailed=args.detailed))
        return

    # 2. Dự đoán theo file
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File không tồn tại: {file_path}")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Đang phân tích {len(lines)} câu từ file {file_path}...")
        results = pipeline.predict(lines, detailed=True)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Đã lưu kết quả phân tích vào: {out_p}")
        else:
            for r in results:
                print(format_prediction_result(r, detailed=args.detailed))
        return

    # 3. Mặc định vào chế độ tương tác nếu không truyền cờ nào
    run_interactive(pipeline, detailed=args.detailed)


if __name__ == "__main__":
    main()

