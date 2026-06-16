# დამხმარე ფუნქციები: seed, sanity check-ები, გრაფიკები
# sanity check-ები ლექციაზე გავიარეთ (forward და backward შემოწმება)

import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .data import EMOTIONS


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_initial_loss(model, loader, device, num_classes=7):
    # forward-ის შემოწმება: random მოდელის loss უნდა იყოს ~ ln(კლასები)
    # 7 კლასზე ln(7) = 1.946. თუ შედეგი ძალიან სცდება, რაღაც პრობლემაა
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        loss = criterion(model(x), y).item()
    expected = math.log(num_classes)
    print(f"initial loss = {loss:.4f}, expected ~ ln({num_classes}) = {expected:.4f}")
    return loss, expected


def overfit_small_batch(model, loader, device, n=20, steps=200, lr=1e-3):
    # backward-ის შემოწმება: პატარა batch-ზე ~100% train acc უნდა მივიღოთ
    # თუ ვერ მიიღწევა, training loop-ში ან backward-ში ხარვეზია
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x, y = next(iter(loader))
    x, y = x[:n].to(device), y[:n].to(device)

    for step in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"step {step:3d}: loss {loss.item():.4f}, acc {acc:.3f}")
    final_acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"final acc {n} მაგალითზე: {final_acc:.3f} (უნდა იყოს ~1.0)")
    return final_acc


def plot_history(history, title="training", save_path=None):
    # train და val loss/acc, overfit ან underfit აქედან ჩანს
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_title("loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val")
    ax[1].set_title("accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def plot_confusion(y_true, y_pred, save_path=None):
    # აჩვენებს, რომელ კლასებს ურევს მოდელი ერთმანეთში
    cm = confusion_matrix(y_true, y_pred)
    labels = [EMOTIONS[i] for i in range(len(EMOTIONS))]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion matrix")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig
