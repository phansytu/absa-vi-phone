import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.data.schema import IGNORE_INDEX

POOLINGS = ("mean", "cls", "aspect_attn")


class MultiTaskABSA(nn.Module):
    """Encoder + 2 nhom dau ra: presence (co/khong tung aspect) va sentiment (3 lop / aspect).

    pooling:
      - "mean" / "cls": mot vector cau chung cho moi aspect (nhu truoc, checkpoint cu van doc duoc).
      - "aspect_attn": moi aspect co 1 vector truy van rieng, "nhin" vao vung cau cua minh
        -> huu ich khi mot cau co nhieu aspect trai chieu ("man hinh dep nhung pin yeu").
        Khoi tao gan dong deu nen luc dau tuong duong mean pooling.
    """

    def __init__(self, encoder_name, n_aspects=10, n_pol=3, dropout=0.1, pooling="mean"):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling phai thuoc {POOLINGS}, nhan {pooling!r}")
        self.encoder = AutoModel.from_pretrained(encoder_name)
        H = self.encoder.config.hidden_size
        self.A, self.P, self.pooling = n_aspects, n_pol, pooling
        self.drop = nn.Dropout(dropout)
        if pooling == "aspect_attn":
            self.queries = nn.Parameter(torch.randn(n_aspects, H) * 0.02)
            self.pres_w = nn.Parameter(torch.randn(n_aspects, H) * 0.02)
            self.pres_b = nn.Parameter(torch.zeros(n_aspects))
            self.sent_w = nn.Parameter(torch.randn(n_aspects, H, n_pol) * 0.02)
            self.sent_b = nn.Parameter(torch.zeros(n_aspects, n_pol))
        else:
            self.presence_head = nn.Linear(H, n_aspects)
            self.sentiment_head = nn.Linear(H, n_aspects * n_pol)

    def _pool(self, hs, mask):
        if self.pooling == "cls":
            return hs[:, 0]
        m = mask.unsqueeze(-1).type_as(hs)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, **_):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "aspect_attn":
            scores = torch.einsum("bth,ah->bat", hs, self.queries) / hs.size(-1) ** 0.5   # (B, A, T)
            scores = scores.masked_fill(attention_mask[:, None, :] == 0, torch.finfo(scores.dtype).min)
            pooled = self.drop(torch.einsum("bat,bth->bah", scores.softmax(-1), hs))      # (B, A, H)
            pres = torch.einsum("bah,ah->ba", pooled, self.pres_w) + self.pres_b          # (B, A)
            sent = torch.einsum("bah,ahp->bap", pooled, self.sent_w) + self.sent_b        # (B, A, P)
            return pres, sent
        h = self.drop(self._pool(hs, attention_mask))
        pres = self.presence_head(h)                                   # (B, A)
        sent = self.sentiment_head(h).view(-1, self.A, self.P)         # (B, A, 3)
        return pres, sent


def multitask_loss(pres_logits, sent_logits, pres_y, sent_y,
                   lam=1.0, pos_weight=None, sent_weight=None):
    l_pres = F.binary_cross_entropy_with_logits(pres_logits, pres_y, pos_weight=pos_weight)
    mask = sent_y != IGNORE_INDEX                      # chi khia canh co mat
    if mask.any():
        l_sent = F.cross_entropy(sent_logits[mask], sent_y[mask], weight=sent_weight)
    else:                                              # tranh NaN khi ca batch khong co khia canh
        l_sent = pres_logits.sum() * 0.0
    return l_pres + lam * l_sent, l_pres.detach(), l_sent.detach()
