# FER2013 მონაცემების წამოღება
# train.csv: ორი სვეტი, emotion (0:6) და pixels (2304 რიცხვი space-ით)
# test.csv: მარტო pixels, ეს Kaggle-ის submission-ისთვისაა
# val-ს train.csv-დან ვაჭრი stratified split-ით

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

EMOTIONS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral",
}
IMG_SIZE = 48
# FER2013-ის mean/std normalize-ისთვის
FER_MEAN = 0.5077
FER_STD = 0.2550


def _pixels_to_array(pixel_strings):
    # 'rიცხვები space-ით' სტრიქონებს ვაქცევ (N, 48, 48) მასივად
    arr = np.array(
        [np.fromstring(p, sep=" ", dtype=np.float32) for p in pixel_strings]
    )
    return arr.reshape(-1, IMG_SIZE, IMG_SIZE)


class FERDataset(Dataset):
    # images: (N,48,48), labels: (N,) ან None თუ test-ია
    def __init__(self, images, labels=None, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def _transform(self, img):
        # 0:255 -> 0:1
        img = img / 255.0
        if self.augment:
            # მარტივი augmentation, CNN-ისთვის სავსებით საკმარისი
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()        # მარცხნივ:მარჯვნივ ასახვა
            # პატარა random shift, padding-ით
            pad = 4
            padded = np.pad(img, pad, mode="edge")
            x = np.random.randint(0, 2 * pad)
            y = np.random.randint(0, 2 * pad)
            img = padded[x:x + IMG_SIZE, y:y + IMG_SIZE]
        img = (img - FER_MEAN) / FER_STD
        return torch.from_numpy(img).float().unsqueeze(0)  # (1,48,48)

    def __getitem__(self, idx):
        img = self._transform(self.images[idx])
        if self.labels is None:
            return img
        return img, int(self.labels[idx])


def load_dataframes(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_path = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    return train_df, test_df


def get_dataloaders(cfg):
    # აბრუნებს train/val/test loader-ებს და class weights-ს
    train_df, test_df = load_dataframes(cfg.data_dir)

    images = _pixels_to_array(train_df["pixels"].values)
    labels = train_df["emotion"].values.astype(np.int64)

    # stratify: რომ ყველა emotion პროპორციულად მოხვდეს val-ში
    x_tr, x_val, y_tr, y_val = train_test_split(
        images, labels,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        stratify=labels,
    )

    train_ds = FERDataset(x_tr, y_tr, augment=cfg.augment)
    val_ds = FERDataset(x_val, y_val, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    test_loader = None
    if test_df is not None:
        test_images = _pixels_to_array(test_df["pixels"].values)
        test_ds = FERDataset(test_images, labels=None, augment=False)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )

    # class weights: disgust ცოტაა, ამიტომ მეტ წონას ვაძლევ
    counts = np.bincount(y_tr, minlength=cfg.num_classes)
    class_weights = counts.sum() / (cfg.num_classes * np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights


def prepare_data(data_dir="data"):
    # ვრწმუნდები რომ data/train.csv და data/test.csv ადგილზეა
    # kaggle ზოგჯერ პირდაპირ csv-ებს გვაძლევს, ზოგჯერ tar.gz-ს ან fer2013.csv-ს
    import glob, tarfile
    tr = os.path.join(data_dir, "train.csv")
    te = os.path.join(data_dir, "test.csv")
    if os.path.exists(tr) and os.path.exists(te):
        print("train.csv და test.csv უკვე ადგილზეა")
        return

    # tar.gz თუ არის, ჯერ ამოვშალოთ
    for tg in glob.glob(os.path.join(data_dir, "*.tar.gz")):
        with tarfile.open(tg) as t:
            t.extractall(data_dir)

    # fer2013.csv-დან (emotion, pixels, Usage) ავაწყოთ train/test თუ ცალკე არ მოგვცეს
    fer = os.path.join(data_dir, "fer2013.csv")
    if not os.path.exists(fer):
        hits = glob.glob(os.path.join(data_dir, "**", "fer2013.csv"), recursive=True)
        if hits:
            fer = hits[0]
    if os.path.exists(fer):
        df = pd.read_csv(fer)
        df[df["Usage"] == "Training"][["emotion", "pixels"]].to_csv(tr, index=False)
        df[df["Usage"] != "Training"][["pixels"]].to_csv(te, index=False)

    print("data:", os.listdir(data_dir))
