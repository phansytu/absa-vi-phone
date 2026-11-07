"""Pipeline xử lý từ đầu đến cuối (End-to-End ABSA Pipeline) cho tiếng Việt.

Nhận văn bản thô -> Tiền xử lý -> Mô hình (MultiTask ViSoBERT hoặc Baseline) -> Áp dụng ngưỡng ->
Trả về danh sách các cặp {aspect, sentiment} cho các khía cạnh có mặt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.preprocess import normalize
from src.data.schema import ASPECTS, POLARITES, IGNORE_INDEX
from src.models.multitask import MultiTaskABSA


# Thông tin giải nghĩa tiếng Việt thân thiện cho 10 khía cạnh
ASPECT_METADATA = {
    "SCREEN": {"vn": "Màn hình", "desc": "Độ sáng, hiển thị, cảm ứng, tấm nền"},
    "CAMERA": {"vn": "Máy ảnh", "desc": "Chụp ảnh, quay phim, camera trước/sau, chụp đêm"},
    "FEATURES": {"vn": "Tính năng", "desc": "Vân tay, loa, mic, sạc nhanh, kết nối"},
    "BATTERY": {"vn": "Pin & Nguồn", "desc": "Thời lượng pin, tụt pin, pin trâu, độ bền pin"},
    "PERFORMANCE": {"vn": "Hiệu năng", "desc": "Tốc độ, độ mượt, giật lag, chơi game, nhiệt độ"},
    "STORAGE": {"vn": "Lưu trữ", "desc": "Dung lượng bộ nhớ trong, RAM, thẻ nhớ"},
    "DESIGN": {"vn": "Thiết kế", "desc": "Ngoại hình, màu sắc, viền, cảm giác cầm nắm"},
    "PRICE": {"vn": "Giá cả", "desc": "Đắt, rẻ, mức giá, đáng tiền, hợp túi tiền"},
    "GENERAL": {"vn": "Tổng quan", "desc": "Đánh giá chung máy tốt/xấu, độ hài lòng"},
    "SER&ACC": {"vn": "Dịch vụ & Phụ kiện", "desc": "Giao hàng, đóng gói, bảo hành, quà tặng, nhân viên"},
}

POLARITY_METADATA = {
    "Positive": {"vn": "Tích cực (Khen)", "color": "green", "symbol": "👍"},
    "Negative": {"vn": "Tiêu cực (Chê)", "color": "red", "symbol": "👎"},
    "Neutral": {"vn": "Trung tính", "color": "gray", "symbol": "😐"},
}


class ABSAPipeline:
    """Pipeline phân tích khía cạnh và cảm xúc cho phản hồi điện thoại tiếng Việt."""

    def __init__(
        self,
        model_type: str = "auto",
        checkpoint_path: str | Path = "outputs/checkpoints/multitask_seed42.pt",
        raw_data_dir: str | Path = "data/raw",
        threshold: Optional[float | List[float] | np.ndarray] = None,
        device: Optional[str] = None,
    ):
        """Khởi tạo pipeline.

        Args:
            model_type: 'auto' (ưu tiên checkpoint, thiếu thì sang baseline), 'multitask', hoặc 'baseline'.
            checkpoint_path: Đường dẫn tới file trọng số mô hình sâu (.pt).
            raw_data_dir: Đường dẫn tới thư mục chứa dữ liệu thô (cho baseline nếu cần).
            threshold: Ngưỡng phát hiện khía cạnh (số thực hoặc mảng 10 phần tử). Mặc định lấy theo checkpoint hoặc 0.45.
            device: 'cuda', 'cpu' hoặc None (tự động nhận diện).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.raw_data_dir = Path(raw_data_dir)
        self.model_type = model_type
        self.user_threshold = threshold

        self.model = None
        self.tokenizer = None
        self.cfg: Dict[str, Any] = {}
        self.thresholds: np.ndarray = np.full(len(ASPECTS), 0.45)
        self.norm_kwargs: Dict[str, Any] = {"tone": True, "teencode": True, "repeat": True}

        self._load_model()

    def _load_model(self):
        use_multitask = False
        if self.model_type in ("auto", "multitask"):
            if self.checkpoint_path.exists():
                use_multitask = True
            elif self.model_type == "multitask":
                raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {self.checkpoint_path}")

        if use_multitask:
            self._load_multitask_model()
            self.active_model = "multitask"
        else:
            self._load_baseline_model()
            self.active_model = "baseline"

    def _load_multitask_model(self):
        # Nạp checkpoint PyTorch an toàn
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        self.cfg = ckpt.get("cfg", {})
        encoder_name = self.cfg.get("encoder", "uitnlp/visobert")
        pooling = self.cfg.get("pooling", "mean")
        self.norm_kwargs = self.cfg.get("norm", self.norm_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        model = MultiTaskABSA(encoder_name, pooling=pooling)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        self.model = model

        # Lấy ngưỡng lưu trong checkpoint
        if self.user_threshold is not None:
            if isinstance(self.user_threshold, (int, float)):
                self.thresholds = np.full(len(ASPECTS), float(self.user_threshold))
            else:
                self.thresholds = np.asarray(self.user_threshold, dtype=float)
        elif "thresholds" in ckpt and ckpt["thresholds"] is not None:
            self.thresholds = np.asarray(ckpt["thresholds"], dtype=float)
        else:
            self.thresholds = np.full(len(ASPECTS), 0.5)

    def _load_baseline_model(self):
        from src.data.load import load_split
        from src.models.baseline import TfidfBaseline

        if not (self.raw_data_dir / "Train.csv").exists():
            raise FileNotFoundError(
                f"Không tìm thấy {self.raw_data_dir / 'Train.csv'} để huấn luyện baseline!"
            )
        train_data = load_split(self.raw_data_dir, "train")
        self.model = TfidfBaseline().fit(train_data)
        if self.user_threshold is not None:
            th = float(self.user_threshold) if isinstance(self.user_threshold, (int, float)) else self.user_threshold[0]
            self.thresholds = np.full(len(ASPECTS), th)
        else:
            self.thresholds = np.full(len(ASPECTS), 0.45)

    def preprocess_text(self, text: str) -> str:
        """Chuẩn hóa văn bản bằng hàm normalize dùng chung của dự án."""
        if not text or not str(text).strip():
            return ""
        return normalize(str(text), **self.norm_kwargs)

    def predict(
        self,
        texts: Union[str, List[str]],
        detailed: bool = False,
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]], Dict[str, Any], List[Dict[str, Any]]]:
        """Dự đoán cảm xúc theo khía cạnh cho một hoặc nhiều văn bản.

        Args:
            texts: Một chuỗi hoặc danh sách các chuỗi phản hồi cần phân tích.
            detailed: Nếu True, trả về thêm thông tin chi tiết (xác suất, văn bản đã chuẩn hóa).

        Returns:
            - Nếu detailed=False:
                + Một câu: [{'aspect': 'BATTERY', 'sentiment': 'Positive'}, ...]
                + Nhiều câu: danh sách các danh sách kết quả trên.
            - Nếu detailed=True:
                + Dict chứa 'text', 'normalized', 'predictions', 'aspect_details'.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        if not text_list:
            return {} if is_single and detailed else []

        # 1. Tiền xử lý
        normalized_texts = [self.preprocess_text(t) for t in text_list]

        # 2. Suy luận (Inference)
        if self.active_model == "multitask":
            prob_matrix, sent_matrix, sent_prob_matrix = self._infer_multitask(normalized_texts)
        else:
            prob_matrix, sent_matrix, sent_prob_matrix = self._infer_baseline(normalized_texts)

        # 3. Tổng hợp kết quả theo từng câu
        results = []
        for i, (orig_text, norm_text) in enumerate(zip(text_list, normalized_texts)):
            preds_for_doc: List[Dict[str, Any]] = []
            aspect_details: Dict[str, Any] = {}

            # Nếu chuỗi rỗng sau khi tiền xử lý -> dự đoán rỗng
            if not norm_text:
                for idx, a in enumerate(ASPECTS):
                    aspect_details[a] = {
                        "name_vn": ASPECT_METADATA[a]["vn"],
                        "presence_prob": 0.0,
                        "threshold": float(self.thresholds[idx]),
                        "is_present": False,
                        "sentiment": None,
                        "sentiment_probs": None,
                    }
                res_item = {
                    "text": orig_text,
                    "normalized": norm_text,
                    "predictions": [],
                    "aspect_details": aspect_details,
                    "has_aspects": False,
                }
                results.append(res_item if detailed else [])
                continue

            for idx, a in enumerate(ASPECTS):
                p_prob = float(prob_matrix[i, idx])
                thr = float(self.thresholds[idx])
                is_present = bool(p_prob >= thr)
                s_idx = int(sent_matrix[i, idx])
                s_name = POLARITES[s_idx]

                s_probs = None
                if sent_prob_matrix is not None:
                    s_probs = {
                        pol: float(sent_prob_matrix[i, idx, p_i])
                        for p_i, pol in enumerate(POLARITES)
                    }

                aspect_details[a] = {
                    "name_vn": ASPECT_METADATA[a]["vn"],
                    "presence_prob": round(p_prob, 4),
                    "threshold": round(thr, 4),
                    "is_present": is_present,
                    "sentiment": s_name if is_present else None,
                    "sentiment_probs": s_probs,
                }

                if is_present:
                    pred_entry = {
                        "aspect": a,
                        "aspect_vn": ASPECT_METADATA[a]["vn"],
                        "sentiment": s_name,
                        "sentiment_vn": POLARITY_METADATA[s_name]["vn"],
                        "confidence": round(p_prob, 4),
                    }
                    if not detailed:
                        # Định dạng chuẩn theo schema của dự án
                        preds_for_doc.append({"aspect": a, "sentiment": s_name})
                    else:
                        preds_for_doc.append(pred_entry)

            if detailed:
                results.append({
                    "text": orig_text,
                    "normalized": norm_text,
                    "predictions": preds_for_doc,
                    "aspect_details": aspect_details,
                    "has_aspects": len(preds_for_doc) > 0,
                })
            else:
                results.append(preds_for_doc)

        return results[0] if is_single else results

    def _infer_multitask(self, texts: List[str]):
        # Thay thế câu rỗng bằng khoảng trắng để tokenizer không lỗi
        safe_texts = [t if t.strip() else "không" for t in texts]
        enc = self.tokenizer(
            safe_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.get("max_length", 128),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            pl, sl = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prob = torch.sigmoid(pl.float()).cpu().numpy()
            sent_probs = torch.softmax(sl.float(), dim=-1).cpu().numpy()
            sent = sl.argmax(dim=-1).cpu().numpy()

        return prob, sent, sent_probs

    def _infer_baseline(self, texts: List[str]):
        prob, sent = self.model.predict([{"text": t} for t in texts])
        return prob, sent, None

    def predict_one(self, text: str, detailed: bool = False):
        """Tiện ích phân tích 1 câu đơn lẻ."""
        return self.predict(text, detailed=detailed)

