# -*- coding: utf-8 -*-
"""训练规则：损失、优化器与训练循环。

- 分类：带逆平方根类别权重的交叉熵；
- 回归：均方误差，仅故障样本参与（Normal 掩码）；
- 总损失 = 分类损失 + 回归损失，作为早停和最佳权重保存的监控指标。
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cwru.config import (BATCH_SIZE, ES_PATIENCE, LR, MAX_EPOCHS, SEED,
                         TRAIN_SEED, WEIGHT_DECAY)
from cwru.data.dataset import CwruDataset
from cwru.models.models import build_model


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def total_loss_from(outputs: dict, y_cls: torch.Tensor, y_reg: torch.Tensor,
                    reg_mask: torch.Tensor, class_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (总损失, 分类损失, 回归损失)。"""
    cls_loss = F.cross_entropy(outputs["logits"], y_cls, weight=class_weights)
    if reg_mask.any():
        reg_loss = F.mse_loss(outputs["diameter"][reg_mask], y_reg[reg_mask])
    else:
        reg_loss = torch.zeros((), device=y_cls.device)
    return cls_loss + reg_loss, cls_loss, reg_loss


def macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    """不依赖 sklearn 的四分类 Macro-F1。"""
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def _evaluate_val(model, loader, class_weights, device) -> dict:
    model.eval()
    totals = {"loss": 0.0, "cls": 0.0, "reg": 0.0}
    n_all, n_reg = 0, 0
    y_true_all, y_pred_all = [], []
    abs_err, sq_err = [], []
    for x, y_cls, y_reg, mask in loader:
        x, y_cls = x.to(device), y_cls.to(device)
        y_reg, mask = y_reg.to(device), mask.to(device)
        out = model(x)
        loss, cls_l, reg_l = total_loss_from(out, y_cls, y_reg, mask, class_weights)
        bs = len(y_cls)
        totals["loss"] += float(loss) * bs
        totals["cls"] += float(cls_l) * bs
        totals["reg"] += float(reg_l) * bs
        n_all += bs
        n_reg += int(mask.sum())
        pred = out["logits"].argmax(dim=1)
        y_true_all.append(y_cls.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())
        if mask.any():
            err = out["diameter"][mask] - y_reg[mask]
            abs_err.append(err.abs().cpu().numpy())
            sq_err.append((err ** 2).cpu().numpy())
    mae = float(np.concatenate(abs_err).mean()) if abs_err else float("nan")
    rmse = float(np.sqrt(np.concatenate(sq_err).mean())) if sq_err else float("nan")
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return {
        "val_loss": totals["loss"] / n_all,
        "val_cls_loss": totals["cls"] / n_all,
        "val_reg_loss": totals["reg"] / n_all,
        "val_acc": float((y_true == y_pred).mean()),
        "val_f1": macro_f1_np(y_true, y_pred),
        "val_mae": mae,
        "val_rmse": rmse,
    }


def train_model(experiment: str, exp_cfg: dict, arrays: dict, model_name: str,
                out_dir: str, epochs: int = MAX_EPOCHS,
                seed: int = TRAIN_SEED, verbose: bool = True,
                split_name: str = "", window_plan: str = "") -> str:
    """训练指定实验与模型，返回运行目录。"""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    in_channels = arrays["X"].shape[1]
    model = build_model(model_name, in_channels).to(device)

    class_weights = torch.as_tensor(arrays["class_weights"], dtype=torch.float32, device=device)
    train_ds = CwruDataset(arrays, "train")
    val_ds = CwruDataset(arrays, "val")
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    config = {
        "split_set": split_name,
        "window_plan": window_plan,
        "window_len": int(arrays.get("window_len", 1024)),
        "stride": int(arrays.get("stride", 1024)),
        "experiment": experiment,
        "display": exp_cfg.get("display", ""),
        "model": model_name,
        "optimizer": "Adam",
        "regression_loss": "MSE",
        "lr_scheduler": "none",
        "in_channels": in_channels,
        "batch_size": BATCH_SIZE,
        "max_epochs": epochs,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "es_patience": ES_PATIENCE,
        "seed": seed,
        "class_weights": arrays["class_weights"],
        "norm": arrays["norm"],
        "train_class_counts": arrays["train_class_counts"],
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "reg_scale": 7.0,
    }

    history = []
    best_val_loss = float("inf")
    best_path = os.path.join(out_dir, "best_inference.pt")
    history_path = os.path.join(out_dir, "history.json")

    if verbose:
        print(f"[{experiment}/{model_name}] 训练开始: train={len(train_ds)} val={len(val_ds)} "
              f"in_channels={in_channels} device={device.type}")

    no_improve = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "cls": 0.0, "reg": 0.0}
        n_all = 0
        for x, y_cls, y_reg, mask in train_loader:
            x, y_cls = x.to(device), y_cls.to(device)
            y_reg, mask = y_reg.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, cls_l, reg_l = total_loss_from(out, y_cls, y_reg, mask, class_weights)
            loss.backward()
            optimizer.step()
            bs = len(y_cls)
            totals["loss"] += float(loss.detach()) * bs
            totals["cls"] += float(cls_l.detach()) * bs
            totals["reg"] += float(reg_l.detach()) * bs
            n_all += bs

        val_metrics = _evaluate_val(model, val_loader, class_weights, device)
        epoch_loss = totals["loss"] / n_all
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_cls_loss": totals["cls"] / n_all,
            "train_reg_loss": totals["reg"] / n_all,
            "lr": lr_now,
            **val_metrics,
        })
        improved = val_metrics["val_loss"] < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_metrics["val_loss"]
            no_improve = 0
            torch.save({
                "split_set": split_name,
                "window_plan": window_plan,
                "experiment": experiment,
                "model": model_name,
                "in_channels": in_channels,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "model_state": model.state_dict(),
                "config": config,
            }, best_path)
        else:
            no_improve += 1

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"  epoch {epoch:3d} train_loss={epoch_loss:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_acc']:.4f} "
                  f"val_mae={val_metrics['val_mae']:.4f} lr={lr_now:.2e} "
                  f"{'*best*' if improved else ''}")

        if no_improve >= ES_PATIENCE:
            if verbose:
                print(f"  早停触发（{ES_PATIENCE} 个 epoch 无改善）")
            break

    if verbose:
        print(f"[{experiment}/{model_name}] 训练结束: 最佳 val_loss={best_val_loss:.4f}, "
              f"用时 {time.time() - t0:.0f}s")

    return out_dir
