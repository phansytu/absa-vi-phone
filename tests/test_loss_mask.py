"""Unit test kiểm tra hàm loss mask của mô hình MultiTask (Mục 6.2)."""

import torch
import pytest
from src.data.schema import IGNORE_INDEX
from src.models.multitask import multitask_loss


def test_multitask_loss_normal_batch():
    batch_size = 4
    n_aspects = 10
    n_pol = 3

    # Giả lập logits
    pres_logits = torch.randn(batch_size, n_aspects, requires_grad=True)
    sent_logits = torch.randn(batch_size, n_aspects, n_pol, requires_grad=True)

    # Giả lập targets: một số có mặt (1), một số vắng (0)
    pres_y = torch.randint(0, 2, (batch_size, n_aspects)).float()
    sent_y = torch.full((batch_size, n_aspects), IGNORE_INDEX, dtype=torch.long)

    # Gán nhãn cho các khía cạnh có mặt
    mask_present = pres_y == 1
    sent_y[mask_present] = torch.randint(0, n_pol, (int(mask_present.sum()),))

    loss, lp, ls = multitask_loss(pres_logits, sent_logits, pres_y, sent_y)

    assert not torch.isnan(loss).any()
    assert not torch.isnan(lp).any()
    assert not torch.isnan(ls).any()
    assert loss.item() > 0

    # Kiểm tra backward
    loss.backward()
    assert pres_logits.grad is not None
    assert sent_logits.grad is not None


def test_multitask_loss_completely_empty_batch():
    """Kiểm tra cảnh báo Mục 6.2: nếu cả batch không có khía cạnh nào hiện diện,
    loss sentiment không được phép ra NaN!
    """
    batch_size = 2
    n_aspects = 10
    n_pol = 3

    pres_logits = torch.randn(batch_size, n_aspects, requires_grad=True)
    sent_logits = torch.randn(batch_size, n_aspects, n_pol, requires_grad=True)

    # Tất cả khía cạnh đều vắng (chỉ có OTHERS hoặc không có gì)
    pres_y = torch.zeros(batch_size, n_aspects, dtype=torch.float)
    sent_y = torch.full((batch_size, n_aspects), IGNORE_INDEX, dtype=torch.long)

    loss, lp, ls = multitask_loss(pres_logits, sent_logits, pres_y, sent_y)

    # Đảm bảo KHÔNG bị NaN
    assert not torch.isnan(loss).any(), "Loss không được phép ra NaN khi batch rỗng!"
    assert not torch.isnan(ls).any()
    assert ls.item() == 0.0

    # Gradient vẫn backward được
    loss.backward()
    assert pres_logits.grad is not None
