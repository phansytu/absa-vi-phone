import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..data.preprocess import mark_negation, normalize
from ..data.schema import ASPECTS, IGNORE_INDEX

TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def prep(text: str, neg_scope: int = 2) -> str:
    t = normalize(text)
    return mark_negation(t, neg_scope) if neg_scope else t


class TfidfBaseline:
    def __init__(
        self,
        C_presence=10.0,
        C_sent=10.0,
        neg_scope=2,
        ngram=(1, 2),
        min_df=2,
        seed=42,
    ):
        self.Cp, self.Cs, self.neg_scope = C_presence, C_sent, neg_scope
        self.ngram, self.min_df, self.seed = ngram, min_df, seed
        self.vec = None
        self.pres = []
        self.sent = []

    def fit(self, rows):
        texts = [prep(r["text"], self.neg_scope) for r in rows]
        self.vec = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=self.ngram,
            min_df=self.min_df,
            sublinear_tf=True,
            lowercase=False,
        )
        X = self.vec.fit_transform(texts)
        P = np.array([r["presence"] for r in rows])
        S = np.array([r["sentiment"] for r in rows])

        self.pres, self.sent = [], []
        for i in range(len(ASPECTS)):
            # 1. Fit Presence model (có kiểm tra fallback 1 lớp)
            y_p = P[:, i]
            unique_p = np.unique(y_p)
            if len(unique_p) < 2:
                self.pres.append(int(unique_p[0]) if len(unique_p) else 0)
            else:
                self.pres.append(
                    LogisticRegression(
                        C=self.Cp,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=self.seed,
                    ).fit(X, y_p)
                )

            # 2. Fit Sentiment model (chỉ trên mẫu khía cạnh xuất hiện)
            m = S[:, i] != IGNORE_INDEX
            y_s = S[:, i][m]
            unique_s = np.unique(y_s)
            if len(unique_s) < 2:
                self.sent.append(int(unique_s[0]) if len(unique_s) else 0)
            else:
                self.sent.append(
                    LogisticRegression(
                        C=self.Cs,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=self.seed,
                    ).fit(X[m], y_s)
                )
        return self

    def predict(self, rows):
        """Trả về (presence_prob[N, 10], sent_pred[N, 10])."""
        X = self.vec.transform([prep(r["text"], self.neg_scope) for r in rows])
        n_samples = X.shape[0]

        # Dự đoán xác suất hiện diện (Presence probabilities)
        prob_cols = []
        for c in self.pres:
            if isinstance(c, (int, float, np.integer)):
                prob_cols.append(np.full(n_samples, float(c)))
            else:
                # Đảm bảo lấy đúng cột xác suất của lớp 1
                idx_1 = np.where(c.classes_ == 1)[0][0]
                prob_cols.append(c.predict_proba(X)[:, idx_1])
        prob = np.column_stack(prob_cols)

        # Dự đoán nhãn cảm xúc (Sentiment predictions)
        sent = np.zeros((n_samples, len(ASPECTS)), dtype=int)
        for i, c in enumerate(self.sent):
            if isinstance(c, (int, float, np.integer)):
                sent[:, i] = int(c)
            else:
                sent[:, i] = c.predict(X)

        return prob, sent

    def predict_aspects(self, rows, threshold=0.5):
        """Hàm tiện ích: Trả về nhãn sentiment hoàn chỉnh [N, 10].
        Nếu presence_prob < threshold -> trả về IGNORE_INDEX (-1 hoặc -100).
        """
        prob, sent = self.predict(rows)
        final_pred = np.full_like(sent, IGNORE_INDEX)
        mask = prob >= threshold
        final_pred[mask] = sent[mask]
        return final_pred