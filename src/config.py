# ყველა ჰიპერპარამეტრი აქ ერთ ადგილას
# wandb-ზე config-ად მიდის და run-ებს ამით ვადარებთ

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # wandb
    project: str = "fer2013-emotion-recognition"
    arch: str = "SmallCNN"          # wandb group (იგივე experiment)
    run_name: str = "baseline"      # run-ის სახელი
    notes: str = ""

    # data
    data_dir: str = "data"
    val_split: float = 0.1          # train.csv-დან რამდენი წავიდეს val-ში
    batch_size: int = 128
    augment: bool = False
    num_workers: int = 2

    # model
    dropout: float = 0.0
    num_classes: int = 7

    # optimizer
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"         # adam ან sgd
    scheduler: str = "none"         # none, cosine ან step
    label_smoothing: float = 0.0
    use_class_weights: bool = False  # disgust ცოტაა, balance-ისთვის

    # სხვა
    seed: int = 42

    def to_dict(self):
        return asdict(self)


# preset-ები რომ ხელით ყოველ ჯერზე არ ავაწყო კონფიგი
# იტერაციულად ვამატებ ახალს
PRESETS = {
    # 1: სუსტი მოდელი, underfit-ის საჩვენებლად
    "mlp_baseline": Config(
        arch="BaselineMLP", run_name="mlp_baseline",
        epochs=25, lr=1e-3, dropout=0.0,
        notes="flatten + 2 FC, ბაზისური floor",
    ),
    # 2: პირველი CNN
    "smallcnn": Config(
        arch="SmallCNN", run_name="smallcnn_base",
        epochs=30, lr=1e-3, dropout=0.25,
        notes="2 conv block + BatchNorm",
    ),
    # 3: ღრმა CNN რეგულარიზაციის გარეშე, აქ overfit-ი გამოჩნდება
    "deepcnn_overfit": Config(
        arch="DeeperCNN", run_name="deepcnn_no_reg",
        epochs=40, lr=1e-3, dropout=0.0, weight_decay=0.0, augment=False,
        notes="4 conv block, განზრახ regularization-ის გარეშე",
    ),
    # 4: იგივე ღრმა CNN, ოღონდ რეგულარიზებული
    "deepcnn_regularized": Config(
        arch="DeeperCNN", run_name="deepcnn_reg_aug",
        epochs=60, lr=1e-3, dropout=0.4, weight_decay=1e-4,
        augment=True, scheduler="cosine", use_class_weights=True,
        notes="dropout + weight decay + augmentation + cosine",
    ),
}
