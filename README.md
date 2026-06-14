# FER2013: სახის ემოციების ამოცნობა

Kaggle competition: Challenges in Representation Learning, Facial Expression Recognition Challenge.

ამ დავალებაში ვაგროვებ პრაქტიკულ გამოცდილებას ნეირონულ ქსელებთან PyTorch-ში.
ვტესტავ რამდენიმე არქიტექტურას და ჰიპერპარამეტრს, ყველა ექსპერიმენტს ვლოგავ
wandb-ზე. მთავარი აქცენტი მაღალ accuracy-ზე კი არა, არამედ overfit/underfit მოდელების
ჩვენებასა და ანალიზზეა (ისე როგორც ინსტრუქციაში ეწერა).

## მონაცემები

48x48 grayscale სურათები სახეებით, 7 კლასი:
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.

train.csv: ორი სვეტი, emotion (0:6) და pixels (2304 რიცხვი space-ით).
test.csv: მარტო pixels, ეს Kaggle submission-ისთვისაა.
val-ს train.csv-დან ვაჭრი stratified split-ით (10%), რომ ყველა emotion
პროპორციულად მოხვდეს. disgust კლასი ბევრად ცოტაა, ანუ მონაცემები imbalanced-ია.

## სტრუქტურა

```
ML-Assignment-4/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py     # ჰიპერპარამეტრების Config და preset-ები
│   ├── data.py       # ჩატვირთვა, split, augmentation, DataLoader
│   ├── models.py     # BaselineMLP, SmallCNN, DeeperCNN
│   ├── utils.py      # sanity check-ები, გრაფიკები
│   └── train.py      # training loop, wandb ლოგირება, submission
├── notebooks/
│   └── fer2013_colab.ipynb
└── reports/          # wandb report-ის ექსპორტი, გრაფიკები
```

## გარემო (Colab)

```bash
# 1: რეპოს კლონირება
!git clone https://github.com/GigiSichinava/ML-Assignment-4.git
%cd ML-Assignment-4
!pip install -q wandb kaggle

# 2: kaggle მონაცემები (kaggle.json ატვირთე)
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
!unzip -q challenges-*.zip -d data/

# 3: wandb login
import wandb; wandb.login()
```

## sanity check-ები (forward და backward)

სრულ ტრენინგამდე ვამოწმებ, რომ მოდელი და loop გამართულია. ლექციაზე ეს გავიარეთ.

1. საწყისი loss (forward): random მოდელის loss უნდა იყოს ~ ln(7) = 1.946.
   თუ ძალიან სცდება, softmax/loss-ში რაღაც გატეხილია.
2. პატარა batch-ის overfit (backward): ~20 მაგალითზე მოდელმა ~100% train acc
   უნდა გავიდეს. თუ ვერ გადის, training loop-ში ან backprop-ში ბაგია.

ფუნქციები: check_initial_loss, overfit_small_batch (src/utils.py).

## არქიტექტურები

პატარა მოდელით ვიწყებ და იტერაციულად ვამატებ სირთულეს. თითო ნაბიჯის მიზეზი ქვემოთ.

| # | მოდელი | რა არის | რას ვაჩვენებ |
|---|---|---|---|
| 1 | BaselineMLP | flatten + 2 FC | underfit, spatial info იკარგება |
| 2 | SmallCNN | 2 conv block + BatchNorm | პირველი CNN, აშკარა გაუმჯობესება |
| 3 | DeeperCNN (no reg) | 4 conv block, dropout/aug გარეშე | overfit, train >> val |
| 4 | DeeperCNN (regularized) | + dropout + weight decay + augmentation | overfit-ის გასწორება |

თითო მათგანი ცალკე wandb run-ია და ერთმანეთს ვადარებ. მარტო საუკეთესო მოდელის
ჩვენება არ მინდა, მთელი პროცესი მინდა ჩანდეს.

## wandb ლოგირების სტრუქტურა

იგივე ლოგიკა, რაც წინა დავალებაში MLflow-ზე:

project: fer2013-emotion-recognition (ერთი, ყველაფერზე).
group: arch (იგივე experiment), მაგ SmallCNN, DeeperCNN.
run: კონკრეტული ჰიპერპარამეტრების კონფიგი (იგივე run).

თითო run-ში ვლოგავ:
config (ყველა ჰიპერპარამეტრი), train/loss, train/acc, val/loss, val/acc epoch-ობრივ,
overfit_gap (train_acc minus val_acc), gradient-ების ჰისტოგრამები (wandb.watch),
confusion matrix val-ზე, summary-ში best_val_acc და num_params.

## შედეგები

| მოდელი | run | best val acc | overfit gap | კომენტარი |
|---|---|---|---|---|
| BaselineMLP | mlp_baseline | TBD | TBD | underfit |
| SmallCNN | smallcnn_base | TBD | TBD | |
| DeeperCNN | deepcnn_no_reg | TBD | TBD | overfit |
| DeeperCNN | deepcnn_reg_aug | TBD | TBD | best |

(ცხრილს ვავსებ ექსპერიმენტების შემდეგ რეალური რიცხვებით.)

## overfit/underfit ანალიზი

(აქ შევაჯამებ: რამ გამოიწვია underfit BaselineMLP-ში, რამ გამოიწვია overfit
DeeperCNN-ში რეგულარიზაციის გარეშე, და რომელმა ცვლილებამ რა შეცვალა. wandb-ის
გრაფიკებზე დაყრდნობით ვწერ.)

## გაშვება

```python
from src.config import PRESETS
from src.train import train_model

cfg = PRESETS["smallcnn"]
model, history, best_acc, (y_true, y_pred) = train_model(cfg)
```
