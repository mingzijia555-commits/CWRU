# -*- coding: utf-8 -*-
"""模型测试：前向、反向、保存与加载。"""
import os

import numpy as np
import torch

from cwru.models.models import build_model, count_parameters


def _batch(in_channels: int = 1, n: int = 4):
    x = torch.randn(n, in_channels, 1024)
    y_cls = torch.randint(0, 4, (n,))
    y_reg = torch.rand(n) * 4.0
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    return x, y_cls, y_reg, mask


def test_forward_shapes():
    for name in ("Cnn1d", "CnnGru"):
        for ch in (1, 2):
            model = build_model(name, ch)
            out = model(torch.randn(2, ch, 1024))
            assert out["logits"].shape == (2, 4)
            assert out["diameter"].shape == (2,)


def test_backward():
    from cwru.training.trainer import total_loss_from
    for name in ("Cnn1d", "CnnGru"):
        model = build_model(name, 2)
        x, y_cls, y_reg, mask = _batch(2)
        out = model(x)
        loss, cls_l, reg_l = total_loss_from(out, y_cls, y_reg, mask,
                                             torch.ones(4))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


def test_normal_excluded_from_regression_loss():
    from cwru.training.trainer import total_loss_from
    model = build_model("Cnn1d", 1)
    x, y_cls, y_reg, mask = _batch(1)
    out = model(x)
    _, _, reg_with = total_loss_from(out, y_cls, y_reg, mask, torch.ones(4))
    _, _, reg_zero = total_loss_from(out, y_cls, y_reg, torch.zeros_like(mask), torch.ones(4))
    assert float(reg_zero) == 0.0
    assert float(reg_with) > 0.0


def test_save_and_load(tmp_path):
    model = build_model("CnnGru", 2)
    path = str(tmp_path / "m.pt")
    torch.save(model.state_dict(), path)
    model2 = build_model("CnnGru", 2)
    model2.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(1, 2, 1024)
    model.eval()
    model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(x)["logits"], model2(x)["logits"])


def test_param_counts_sane():
    # 两个模型都应较小（< 2M 参数）
    assert count_parameters(build_model("Cnn1d", 1)) < 2_000_000
    assert count_parameters(build_model("CnnGru", 1)) < 2_000_000
