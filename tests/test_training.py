# -*- coding: utf-8 -*-
"""训练器测试：合成小数据上的训练、断点续训、最佳权重推理。"""
import numpy as np
import torch

from cwru.config import EXPERIMENTS
from cwru.training.trainer import _evaluate_val, train_model


def _synthetic_arrays(n_per_class: int = 40, in_channels: int = 1):
    """构建带可学信号的合成数据：不同类别不同频率，直径=类别序号/7。"""
    rng = np.random.default_rng(0)
    t = np.arange(1024) / 12000.0
    xs, ys_cls, ys_reg, masks, file_idx, files = [], [], [], [], [], []
    n_files = 0
    for split, frac in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        for c in range(4):
            for k in range(max(1, int(n_per_class * frac))):
                freq = 500.0 + c * 900.0
                sig = np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(1024)
                xs.append(sig.astype(np.float32))
                ys_cls.append(c)
                ys_reg.append(c / 7.0)
                masks.append(c != 0)
                file_idx.append(n_files)
                files.append({"index": n_files, "filename": f"syn_{split}_{c}_{k}.mat",
                              "split": split, "fault": ["Normal", "IR", "OR", "B"][c],
                              "end": "Drive", "diameter_mil": 0.0 if c == 0 else c * 7.0,
                              "load": 0, "or_clock": None, "channels": ["DE"],
                              "n_windows": 1})
                n_files += 1
    X = np.stack(xs)[:, None, :]
    return {
        "X": X, "y_cls": np.array(ys_cls, dtype=np.int64),
        "y_reg": np.array(ys_reg, dtype=np.float32),
        "reg_mask": np.array(masks, dtype=bool),
        "file_idx": np.array(file_idx, dtype=np.int32),
        "files": files,
        "norm": [{"mean": float(X.mean()), "std": float(X.std())}],
        "class_weights": [1.0, 1.0, 1.0, 1.0],
        "train_class_counts": [40, 40, 40, 40],
    }


def test_train_and_resume(tmp_path):
    arrays = _synthetic_arrays()
    exp_cfg = EXPERIMENTS["101DE"]
    out_dir = train_model("SYNTH", exp_cfg, arrays, "Cnn1d", epochs=2, seed=42)
    import os
    ckpt = torch.load(os.path.join(out_dir, "last_training.ckpt"),
                      map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2
    best_path = os.path.join(out_dir, "best_inference.pt")
    assert os.path.exists(best_path)

    # 断点续训 1 个 epoch
    train_model("SYNTH", exp_cfg, arrays, "Cnn1d", epochs=3, resume=True, seed=42)
    ckpt2 = torch.load(os.path.join(out_dir, "last_training.ckpt"),
                       map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 3
    assert ckpt2["history"][0]["epoch"] == 1


def test_best_weight_inference():
    arrays = _synthetic_arrays()
    exp_cfg = EXPERIMENTS["101DE"]
    out_dir = train_model("SYNTH2", exp_cfg, arrays, "Cnn1d", epochs=1, seed=42)
    import os
    ckpt = torch.load(os.path.join(out_dir, "best_inference.pt"),
                      map_location="cpu", weights_only=False)
    model = build_model_for_test("Cnn1d", 1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(arrays["X"][:8]))
    probs = torch.softmax(out["logits"], dim=1)
    assert probs.shape == (8, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-4)


def build_model_for_test(name, ch):
    from cwru.models.models import build_model
    return build_model(name, ch)


def test_evaluate_val_metrics():
    arrays = _synthetic_arrays(4)
    model = build_model_for_test("CnnGru", 1)
    from torch.utils.data import DataLoader
    from cwru.data.dataset import CwruDataset
    loader = DataLoader(CwruDataset(arrays, None), batch_size=16)
    m = _evaluate_val(model, loader,
                      torch.tensor(arrays["class_weights"]),
                      torch.device("cpu"))
    for key in ("val_loss", "val_acc", "val_mae", "val_rmse"):
        assert key in m
