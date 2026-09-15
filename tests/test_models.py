# -*- coding: utf-8 -*-
"""模型测试：前向、反向、保存与加载；三个模型结构对齐。"""
import torch

from cwru.config import MODELS
from cwru.models.models import build_model, count_parameters


def _batch(in_channels: int = 1, n: int = 4):
    x = torch.randn(n, in_channels, 1024)
    y_cls = torch.randint(0, 4, (n,))
    y_reg = torch.rand(n) * 4.0
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    return x, y_cls, y_reg, mask


def test_three_models_defined():
    assert MODELS == ["Cnn1d", "CnnGru", "CnnLstm"]


def test_forward_shapes():
    for name in MODELS:
        for ch in (1, 2):
            model = build_model(name, ch)
            out = model(torch.randn(2, ch, 1024))
            assert out["logits"].shape == (2, 4)
            assert out["diameter"].shape == (2,)


def test_backward():
    from cwru.training.trainer import total_loss_from
    for name in MODELS:
        model = build_model(name, 2)
        x, y_cls, y_reg, mask = _batch(2)
        out = model(x)
        loss, cls_l, reg_l = total_loss_from(out, y_cls, y_reg, mask, torch.ones(4))
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


def test_save_and_load():
    for name in MODELS:
        model = build_model(name, 2)
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        model2 = build_model(name, 2)
        model2.load_state_dict(sd)
        x = torch.randn(1, 2, 1024)
        model.eval()
        model2.eval()
        with torch.no_grad():
            assert torch.allclose(model(x)["logits"], model2(x)["logits"], atol=1e-6)


def test_gru_lstm_structure_aligned():
    """CnnGru 与 CnnLstm 结构对齐：同一 CNN 主干与输出头，单层双向、单方向隐藏 64。

    循环单元本身不同（GRU 3 门 / LSTM 4 门），门维度必然不同，因此只校验
    共享部分一致，以及隐藏规模、层数、双向性与共享特征维度一致。
    """
    gru = build_model("CnnGru", 1)
    lstm = build_model("CnnLstm", 1)
    g_named = dict(gru.named_parameters())
    l_named = dict(lstm.named_parameters())

    # 主干与输出头：同名同形状
    g_shared = {n: p for n, p in g_named.items() if not n.startswith("gru.")}
    l_shared = {n: p for n, p in l_named.items() if not n.startswith("lstm.")}
    assert set(g_shared) == set(l_shared)
    for n, p in g_shared.items():
        assert p.shape == l_shared[n].shape, n

    # 循环单元超参对齐
    assert gru.gru.hidden_size == lstm.lstm.hidden_size == 64
    assert gru.gru.num_layers == lstm.lstm.num_layers == 1
    assert gru.gru.bidirectional and lstm.lstm.bidirectional
    assert gru.gru.input_size == lstm.lstm.input_size == 128

    # 共享特征维度：双向拼接 128
    x = torch.randn(2, 1, 1024)
    assert gru.extract(x).shape == lstm.extract(x).shape == (2, 128)
    assert gru(torch.randn(2, 1, 1024))["logits"].shape == (2, 4)

    # 门维度差异符合预期（GRU 3 门 = 192，LSTM 4 门 = 256）
    assert g_named["gru.weight_ih_l0"].shape == (192, 128)
    assert l_named["lstm.weight_ih_l0"].shape == (256, 128)


def test_param_counts_sane():
    for name in MODELS:
        assert count_parameters(build_model(name, 1)) < 2_000_000
