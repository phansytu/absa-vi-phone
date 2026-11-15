"""Giao diện web trực quan (Gradio Web UI) cho hệ thống ABSA phản hồi điện thoại.

Khởi chạy ứng dụng:
    python demo/app.py
    (Sau đó mở trình duyệt tại địa chỉ http://127.0.0.1:7860)
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np

from src.pipeline import ABSAPipeline, ASPECT_METADATA, POLARITY_METADATA
from src.data.schema import ASPECTS

# Khởi tạo pipeline mặc định
pipeline = ABSAPipeline(model_type="auto")


def analyze_text(text: str, model_choice: str, threshold_val: float):
    if not text or not text.strip():
        return (
            "<div style='padding: 20px; text-align: center; color: #888; font-size: 16px;'>"
            "⚠️ Vui lòng nhập ít nhất một câu phản hồi để hệ thống phân tích.</div>",
            "",
            {},
        )

    # Đổi mô hình nếu người dùng chọn khác với mô hình hiện tại
    target_type = "multitask" if "MultiTask" in model_choice else "baseline"
    if pipeline.active_model != target_type:
        pipeline.model_type = target_type
        pipeline._load_model()

    # Cập nhật ngưỡng
    pipeline.thresholds = np.full(len(ASPECTS), float(threshold_val))

    result = pipeline.predict(text, detailed=True)
    preds = result["predictions"]

    # 1. Tạo giao diện HTML trực quan
    html_cards = []
    html_cards.append("<div style='font-family: Arial, sans-serif;'>")
    html_cards.append(f"<p><b>Văn bản sau tiền xử lý:</b> <i>{result['normalized']}</i></p>")

    if not preds:
        html_cards.append(
            "<div style='background-color: #f8f9fa; border-left: 5px solid #6c757d; padding: 15px; margin: 15px 0; border-radius: 4px;'>"
            "<h4>⚪ Không phát hiện khía cạnh nào trong 10 nhóm mục tiêu (OTHERS)</h4>"
            "<p style='margin: 0; color: #555;'>Câu đánh giá này có thể thuộc nhóm dịch vụ chung ngoài lề hoặc không chứa nhận xét về các bộ phận điện thoại.</p>"
            "</div>"
        )
    else:
        html_cards.append(f"<h3 style='color: #2c3e50;'>Phát hiện {len(preds)} khía cạnh:</h3>")
        html_cards.append("<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-top: 10px;'>")

        for p in preds:
            asp = p["aspect"]
            asp_vn = p["aspect_vn"]
            pol = p["sentiment"]
            pol_meta = POLARITY_METADATA.get(pol, {})
            desc = ASPECT_METADATA.get(asp, {}).get("desc", "")

            # Phối màu cho từng cực tính
            if pol == "Positive":
                bg_color = "#e8f5e9"
                border_color = "#4caf50"
                badge_bg = "#2e7d32"
                symbol = "👍"
            elif pol == "Negative":
                bg_color = "#ffebee"
                border_color = "#f44336"
                badge_bg = "#c62828"
                symbol = "👎"
            else:
                bg_color = "#fffde7"
                border_color = "#fbc02d"
                badge_bg = "#f57f17"
                symbol = "😐"

            card = f"""
            <div style='background: {bg_color}; border: 1.5px solid {border_color}; border-radius: 8px; padding: 12px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 16px; font-weight: bold; color: #222;'>{asp_vn} ({asp})</span>
                    <span style='background: {badge_bg}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;'>
                        {symbol} {pol}
                    </span>
                </div>
                <div style='font-size: 12px; color: #666; margin-bottom: 6px;'>{desc}</div>
                <div style='font-size: 13px; color: #333;'>
                    Độ tin cậy hiện diện: <b>{p['confidence']:.1%}</b>
                </div>
            </div>
            """
            html_cards.append(card)

        html_cards.append("</div>")

    html_cards.append("</div>")
    html_output = "".join(html_cards)

    # 2. Bảng xác suất 10 khía cạnh
    prob_dict = {
        f"{ASPECT_METADATA[a]['vn']} ({a})": det["presence_prob"]
        for a, det in result["aspect_details"].items()
    }

    # 3. Kết quả JSON
    json_output = json.dumps(
        {
            "text": result["text"],
            "predictions": preds,
            "threshold": threshold_val,
            "model": pipeline.active_model,
        },
        ensure_ascii=False,
        indent=2,
    )

    return html_output, json_output, prob_dict


def create_demo():
    custom_css = """
    .gradio-container { max-width: 1000px !important; margin: auto !important; }
    """

    with gr.Blocks(title="ABSA Tiếng Việt - Đánh Giá Điện Thoại", css=custom_css) as demo:
        gr.Markdown(
            """
            # 📱 Phân Tích Cảm Xúc Theo Khía Cạnh (ABSA) - UIT-ViSFD
            Hệ thống tự động phát hiện **10 khía cạnh điện thoại** và phân loại **3 cực tính (Tích cực / Tiêu cực / Trung tính)**.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    label="Nhập bình luận đánh giá điện thoại:",
                    placeholder="Ví dụ: Pin trâu dùng 2 ngày thoải mái, nhưng camera chụp đêm hơi mờ...",
                    lines=3,
                )
                with gr.Row():
                    btn_submit = gr.Button("🔍 Phân tích cảm xúc", variant="primary")
                    btn_clear = gr.Button("Xóa")

            with gr.Column(scale=2):
                model_choice = gr.Radio(
                    choices=["MultiTask ViSoBERT (Mô hình sâu)", "Baseline TF-IDF"],
                    value="MultiTask ViSoBERT (Mô hình sâu)",
                    label="Chọn mô hình:",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="Ngưỡng phát hiện khía cạnh (Threshold):",
                    info="Xác suất >= ngưỡng sẽ được xem là có mặt.",
                )

        gr.Markdown("### 🎯 Kết quả phân tích trực quan:")
        output_html = gr.HTML()

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📊 Biểu đồ xác suất hiện diện 10 khía cạnh:")
                output_chart = gr.Label(num_top_classes=10, label="Xác suất theo khía cạnh")

            with gr.Column():
                gr.Markdown("#### 📋 Dữ liệu JSON:")
                output_json = gr.Code(language="json", label="JSON Output")

        # Các ví dụ mẫu đa dạng
        gr.Markdown("### 💡 Các câu thử nghiệm mẫu:")
        gr.Examples(
            examples=[
                ["Pin trâu dùng 2 ngày thoải mái, nhưng camera chụp đêm hơi mờ tịt, màn hình hiển thị đẹp."],
                ["Máy xài rất mượt, giá 600k thế này quá rẻ, nhân viên tư vấn nhiệt tình chu đáo."],
                ["Cực kỳ thất vọng, máy hay bị đơ giật lag, pin tụt nhanh dã man, loa rè."],
                ["Shop giao hàng nhanh chóng, đóng gói cẩn thận 2 lớp chống sốc."],
                ["Thiết kế sang trọng bắt mắt nhưng giá hơi chát so với cấu hình 64GB."],
                ["máy ko đc mượt lắm, chơi game hơi nóng nhưng pin trâu nhé."],
            ],
            inputs=[input_text],
        )

        btn_submit.click(
            fn=analyze_text,
            inputs=[input_text, model_choice, threshold_slider],
            outputs=[output_html, output_json, output_chart],
        )
        input_text.submit(
            fn=analyze_text,
            inputs=[input_text, model_choice, threshold_slider],
            outputs=[output_html, output_json, output_chart],
        )
        btn_clear.click(lambda: ("", "", "", {}), outputs=[input_text, output_html, output_json, output_chart])

    return demo


if __name__ == "__main__":
    app = create_demo()
    app.launch(share=False)

