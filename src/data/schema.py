import re
from typing import Dict, List, Tuple

ASPECTS: List[str] = ["SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE", "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC"]
POLARITES: List[str] = ["Positive", "Negative", "Neutral"]
ASPECT2ID = {a : i for i, a in enumerate(ASPECTS)}
POL2ID = {p: i for i, p in enumerate(POLARITES)}
IGNORE_INDEX = -100 # khia canh vang
OTHER_TOKEN = "OTHERS"
_TOKEN = re.compile(r"\{([^}]*)\}") 

def parse_label(raw : str) -> Tuple[Dict[str, str], bool]:
    """Trả về ({aspect: polarity}, has_others). Báo lỗi nếu gặp token lạ."""
    pairs: Dict [str, str] = {}
    has_others = False
    for tok in _TOKEN.findall(str(raw)):
        if tok == OTHER_TOKEN:
            has_others =True
            continue
        aspect, sep, pol  = tok.partition("#")
        if not sep or aspect not in ASPECT2ID or pol not in POL2ID:
            raise ValueError(f"Token nhãn không hợp lệ: {tok!r} trong {raw!r}")
        if aspect in pairs:
            raise ValueError(f"Khía cạnh lặp trong một phản hồi: {raw!r}")
        pairs[aspect] = pol
    return pairs, has_others

def to_vectors(pairs: Dict[str, str]) -> Tuple[List[int], List[int]]:  
    """presence[10] in {0, 1}; sentiment[10] in {0,1,2} hoặc -100 nếu văng. Khía cạnh văng không được gán neutral."""
    presence = [0] * len(ASPECTS)
    sentiment = [IGNORE_INDEX] * len(ASPECTS)
    for a, p in pairs.items():
        presence[ASPECT2ID[a]] = 1
        sentiment[ASPECT2ID[a]] = POL2ID[p]
    return presence, sentiment

def from_vector(presence: List[int], sentiment: List[int]) -> List[Dict[str, str]]:
    """Chuyển ngược vector về dạng [{aspect, sentiment}] chỉ cho khía cạnh có mặt."""
    results = []
    
    for i, flag in enumerate(presence):
        # Chỉ xử lý khi khía cạnh có xuất hiện và chỉ số cảm xúc hợp lệ (0, 1, 2)
        if flag and 0 <= sentiment[i] < len(POLARITES):
            results.append({
                "aspect": ASPECTS[i],
                "sentiment": POLARITES[sentiment[i]]
            })
            
    return results