import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.data.schema import IGNORE_INDEX


class MultiTaskABSA(nn.Module):
    def __init__(self, encoder_name, n_aspects=10, n_pol=3, dropout=0.1, pooling="mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        H = self.encoder.config.hidden_size
        self.A, self.P, self.pooling = n_aspects, n_pol, pooling
        self.drop = nn.Dropout(dropout)
        self.presence_head = nn.Linear(H, n_aspects)
        self.sentiment_head = nn.Linear(H, n_aspects * n_pol)

    def _pool(self, hs, mask):
        if self.pooling == "cls":
            return hs[:, 0]
        m = mask.unsqueeze(-1).type_as(hs)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, **_):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.drop(self._pool(hs, attention_mask))
        pres = self.presence_head(h)                                   # (B, A)
        sent = self.sentiment_head(h).view(-1, self.A, self.P)         # (B, A, 3)
        return pres, sent


def multitask_loss(pres_logits, sent_logits, pres_y, sent_y,
                   lam=1.0, pos_weight=None, sent_weight=None):
    l_pres = F.binary_cross_entropy_with_logits(pres_logits, pres_y, pos_weight=pos_weight)
    mask = sent_y != IGNORE_INDEX                      # chỉ khía cạnh có mặt
    if mask.any():
        l_sent = F.cross_entropy(sent_logits[mask], sent_y[mask], weight=sent_weight)
    else:                                              # tránh NaN khi cả batch không có khía cạnh
        l_sent = pres_logits.sum() * 0.0
    return l_pres + lam * l_sent, l_pres.detach(), l_sent.detach()