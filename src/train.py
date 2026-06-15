# training loop და wandb-ზე ლოგირება
# wandb სტრუქტურა იგივეა რაც MLflow-ზე გვქონდა:
#   project: ერთი ყველაფერზე
#   group: arch (იგივე experiment, მაგ SmallCNN)
#   run: კონკრეტული ჰიპერპარამეტრების კონფიგი
# თითო run-ში: config, epoch-ობრივ loss/acc, overfit_gap, gradient-ები, confusion matrix

import numpy as np
import torch
import torch.nn as nn
import wandb

from .models import get_model
from .data import get_dataloaders, EMOTIONS
from .utils import set_seed, count_params


def _build_optimizer(model, cfg):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                               nesterov=True, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def _build_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        if return_preds:
            all_true.extend(y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
    metrics = (running_loss / total, correct / total)
    if return_preds:
        return metrics + (np.array(all_true), np.array(all_pred))
    return metrics


def train_model(cfg, device=None, log_wandb=True):
    # ერთი run-ის სრული ტრენინგი
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_weights = get_dataloaders(cfg)
    model = get_model(cfg).to(device)

    weight = class_weights.to(device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    if log_wandb:
        wandb.init(project=cfg.project, group=cfg.arch, name=cfg.run_name,
                   config=cfg.to_dict(), notes=cfg.notes, reinit=True)
        # log="all": gradient-ებსაც ლოგავს, ანუ backward-ს ვადევნებ თვალყურს
        wandb.watch(model, log="all", log_freq=100)
        wandb.summary["num_params"] = count_params(model)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss, "train/acc": tr_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "overfit_gap": tr_acc - val_acc,   # overfit:underfit-ის მთავარი მაჩვენებელი
                "lr": optimizer.param_groups[0]["lr"],
            })
        print(f"epoch {epoch:3d}: train {tr_loss:.3f}/{tr_acc:.3f}, "
              f"val {val_loss:.3f}/{val_acc:.3f}, gap {tr_acc - val_acc:+.3f}")

    # ბოლოს confusion matrix val-ზე
    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device, return_preds=True)
    if log_wandb:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["final_overfit_gap"] = history["train_acc"][-1] - history["val_acc"][-1]
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true, preds=y_pred,
            class_names=[EMOTIONS[i] for i in range(cfg.num_classes)])})
        wandb.finish()

    return model, history, best_val_acc, (y_true, y_pred)


@torch.no_grad()
def make_submission(model, cfg, out_path="submission.csv", device=None):
    # test.csv-ზე პროგნოზი და Kaggle-ის submission ფაილი
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = get_dataloaders(cfg)
    assert test_loader is not None, "test.csv ვერ ვიპოვე data_dir-ში"
    model.eval()
    preds = []
    for x in test_loader:
        x = x.to(device)
        preds.extend(model(x).argmax(1).cpu().numpy())
    import pandas as pd
    df = pd.DataFrame({"id": np.arange(len(preds)), "emotion": preds})
    df.to_csv(out_path, index=False)
    print(f"submission შენახულია: {out_path}, {len(preds)} row")
    return out_path
