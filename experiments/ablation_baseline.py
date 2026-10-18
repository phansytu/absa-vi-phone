import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")  # Chạy từ thư mục gốc: python experiments/ablation_baseline.py data/raw

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.load import load_split
from src.data.preprocess import normalize, mark_negation
from src.data.schema import IGNORE_INDEX
from src.evaluate import evaluate

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
TR = load_split(RAW_DIR, "train")
P = np.array([r["presence"] for r in TR])
S = np.array([r["sentiment"] for r in TR])
RAW = [r["text"] for r in TR]

DEF = r"(?u)\b\w\w+\b"
FIX = r"(?u)\b\w+\b|[^\w\s]"


def cv(texts, tp=FIX, ngram=(1, 2), C=4.0, min_df=2, k=5, ret_oof=False):
    kf = KFold(k, shuffle=True, random_state=0)
    res = []
    oof_prob = np.zeros(P.shape)
    oof_sent = np.zeros(P.shape, dtype=int)

    for tr, te in kf.split(texts):
        vec = TfidfVectorizer(
            token_pattern=tp,
            ngram_range=ngram,
            min_df=min_df,
            sublinear_tf=True,
            lowercase=False,
        )
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])

        prob = np.zeros((len(te), 10))
        sent = np.zeros((len(te), 10), dtype=int)

        for a in range(10):
            # Presence Model (có fallback an toàn)
            y_p = P[tr, a]
            u_p = np.unique(y_p)
            if len(u_p) < 2:
                prob[:, a] = float(u_p[0]) if len(u_p) else 0.0
            else:
                cp = LogisticRegression(
                    C=C, class_weight="balanced", max_iter=1000, random_state=42
                ).fit(Xtr, y_p)
                idx_1 = np.nonzero(cp.classes_ == 1)[0][0]
                prob[:, a] = cp.predict_proba(Xte)[:, idx_1]

            # Sentiment Model (chỉ huấn luyện trên mẫu xuất hiện khía cạnh)
            m = S[tr, a] != IGNORE_INDEX
            y_s = S[tr, a][m]
            u_s = np.unique(y_s)
            if len(u_s) < 2:
                sent[:, a] = int(u_s[0]) if len(u_s) else 0
            else:
                cs = LogisticRegression(
                    C=C, class_weight="balanced", max_iter=1000, random_state=42
                ).fit(Xtr[m], y_s)
                sent[:, a] = cs.predict(Xte)

        r = evaluate([TR[i] for i in te], (prob >= 0.5).astype(int), sent)
        res.append((r["joint_f1"], r["aspect_macro_f1"], r["sentiment_macro_f1_given_gold_aspect"]))
        oof_prob[te] = prob
        oof_sent[te] = sent

    res = np.array(res)
    out = (res.mean(0), res.std(0))
    return (out, oof_prob, oof_sent) if ret_oof else out


def show(name, o):
    m, s = o[0] if isinstance(o, tuple) and len(o) == 3 and isinstance(o[1], np.ndarray) else o
    print(
        f"{name:34s} joint={m[0]:.4f}±{s[0]:.4f}  asp_macro={m[1]:.4f}  sent|gold={m[2]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    """Ablation tiền xử lý bằng 5-fold CV trên TRAIN (không dùng Dev/Test)."""
    N = lambda **k: [normalize(x, **k) for x in RAW]

    show("V0 token mặc định, chỉ NFC/lower", cv(N(tone=False, teencode=False), tp=DEF))
    show("V1 + sửa token pattern", cv(N(tone=False, teencode=False)))
    show("V2 + chuẩn dấu thanh", cv(N(tone=True, teencode=False)))
    show("V3 + teencode/k->không", cv(N()))
    show("V4 + đánh dấu phủ định scope 2", cv([mark_negation(x, 2) for x in N()]))

    base = [mark_negation(x, 2) for x in N()]
    show("V4 + C=10", cv(base, C=10.0))
    show("V4 + unigram only", cv(base, ngram=(1, 1)))
    show("V4 + ngram (1,3)", cv(base, ngram=(1, 3)))