# არქიტექტურები, მარტივიდან რთულისკენ
# 1: BaselineMLP, სუსტი მოდელი
# 2: SmallCNN, 2 conv block
# 3: DeeperCNN, 4 conv block VGG-ის სტილში

import torch.nn as nn


class BaselineMLP(nn.Module):
    # ყველაზე მარტივი მოდელი: სურათს ბრტყელ ვექტორად ვაქცევ
    # spatial ინფორმაცია იკარგება, ამიტომ ბევრს ვერ ისწავლის
    def __init__(self, num_classes=7, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def conv_block(in_c, out_c, dropout=0.0):
    # ორი conv, შემდეგ BatchNorm და ReLU, ბოლოს pooling და dropout
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class SmallCNN(nn.Module):
    # 2 conv block, სივრცითი ზომა: 48 -> 24 -> 12
    def __init__(self, num_classes=7, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, dropout),
            conv_block(32, 64, dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeeperCNN(nn.Module):
    # 4 conv block, სივრცითი ზომა: 48 -> 24 -> 12 -> 6 -> 3
    # dropout=0 overfit-ის საჩვენებლად, dropout>0 რეგულარიზებული ვერსიისთვის
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 64, dropout * 0.5),
            conv_block(64, 128, dropout * 0.5),
            conv_block(128, 256, dropout),
            conv_block(256, 256, dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(cfg):
    # config-ის arch ველის მიხედვით ვირჩევ მოდელს
    if cfg.arch == "BaselineMLP":
        return BaselineMLP(cfg.num_classes, cfg.dropout)
    if cfg.arch == "SmallCNN":
        return SmallCNN(cfg.num_classes, cfg.dropout)
    if cfg.arch == "DeeperCNN":
        return DeeperCNN(cfg.num_classes, cfg.dropout)
    raise ValueError(f"უცნობი არქიტექტურა: {cfg.arch}")
