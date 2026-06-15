# FER2013: სახის ემოციების ამოცნობა

## პროექტის მიმოხილვა:

მოცემული პროექტი ეფუძნება Kaggle-ის პლატფორმაზე არსებულ კონკურსს, რომელიც სახის ემოციების კლასიფიკაციას ისახავს მიზნად. მოდელი 48x48 პიქსელის grayscale სურათს იღებს და 7 კლასიდან ერთ-ერთს ანიჭებს: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. ერთ-ერთი მთავარი გამოწვევაა კლასების დისბალანსი: Disgust კლასი დანარჩენებთან შედარებით ძალიან ნაკლები მაგალითით არის წარმოდგენილი. მთავარი აქცენტი გაკეთებულია overfitting/underfitting-ის ჩვენებასა და ანალიზზე და არა მხოლოდ მაღალ accuracy-ზე.

---

## ჩემი მიდგომა პრობლემის გადასაჭრელად:

1. **Sanity Check-ები (Forward და Backward):** სრული ტრენინგის დაწყებამდე ვამოწმებ, რომ random მოდელის loss ~ ln(7) = 1.946 (forward) და 20 მაგალითზე მოდელი ~100% accuracy-ს აღწევს (backward).
2. **იტერაციული არქიტექტურა:** ვიწყებ BaselineMLP-ით და თანდათანობით ვამატებ სირთულეს, თითოეული ნაბიჯის მიზეზი დასაბუთებულია.
3. **Overfitting-ის დემონსტრაცია:** DeeperCNN-ს გარეგულარიზებულ ვერსიას ვუშვებ, რომ ნათლად ჩანდეს train/val gap-ის ზრდა.
4. **Regularization:** dropout, weight decay, augmentation და cosine LR scheduler-ის კომბინაციით overfit-ს ვამცირებ.
5. **Hyperparameter Sweep:** თითოეული არქიტექტურისთვის lr-ისა და dropout-ის სხვადასხვა მნიშვნელობებს ვტესტავ.

---

## რეპოზიტორიის სტრუქტურა:

```
ML-Assignment-4/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py          Config dataclass და experiment preset-ები
│   ├── data.py            FER2013 ჩატვირთვა, split, augmentation, DataLoader
│   ├── models.py          BaselineMLP, SmallCNN, DeeperCNN
│   ├── utils.py           Sanity check-ები (forward/backward), გრაფიკები
│   └── train.py           Training loop, wandb ლოგირება, Kaggle submission
├── notebooks/
│   └── fer2013_colab.ipynb
└── reports/
```

---

## მონაცემები:

**`train.csv:`** სასწავლო მონაცემები. სვეტები: emotion (0-6) და pixels (2304 რიცხვი space-ით). ~28,709 მაგალითი. val-ი 10%-ად ვაჭრი stratified split-ით.

**`test.csv:`** სატესტო მონაცემები. მხოლოდ pixels სვეტი. Kaggle submission-ისთვის.

**კლასების დისბალანსი:** Disgust კლასი (~450 მაგალითი) Happy-სთან (~7000) შედარებით ძალიან ნაკლებია. საუკეთესო კონფიგურაციაში use_class_weights=True ვიყენებ.

---

## Sanity Check-ები (Forward და Backward):

სრულ ტრენინგამდე ვამოწმებ, რომ კოდი სწორად მუშაობს.

**Forward Check:** random-ად ინიციალიზებული მოდელის loss-ი უნდა იყოს ~ ln(7) = 1.946. თუ ძალიან სცდება, loss ფუნქციაში ან softmax-ში ბაგია.

**შედეგი:** initial loss = 1.9443, expected = 1.9459. ✓

**Backward Check:** 20 მაგალითზე მოდელი ~100% train accuracy-ს უნდა მიაღწევდეს. თუ ვერ აღწევს, training loop-ში ან backprop-ში ბაგია.

**შედეგი:** final acc = 1.000 on 20 examples. ✓

---

## Training:

### მოდელების წვრთნა და შეფასება:

ყველა მოდელი ვაწვრთნი train/val split-ით (90/10, stratified). ყველა run wandb-ში ვლოგავ overfit_gap მეტრიკით (train_acc - val_acc).

---

### 1. BaselineMLP: Overfitting-ის დემონსტრაცია

სურათს flatten-ვაქცევ ვექტორად და 2 FC layer-ს ვიყენებ. spatial სტრუქტურა მთლიანად იკარგება.

| run | epochs | dropout | best val acc | overfit gap |
|---|---|---|---|---|
| mlp_baseline | 25 | 0.0 | 0.455 | 0.483 |

**დასკვნა:** train acc 92%-მდე იზრდება, val acc კი 45%-ზე ჩერდება. gap = 0.483, კლასიკური overfit. MLP-ი train set-ს ზეპირად ისწავლის, ახალ მაგალითებზე ვერ განაზოგადებს. spatial feature-ების დასაჭერად CNN გვჭირდება.

---

### 2. SmallCNN: პირველი CNN

2 conv block + BatchNorm. 48x48 → 24x24 → 12x12. Convolution spatial სტრუქტურას ინარჩუნებს, ამიტომ MLP-ზე მკვეთრი გაუმჯობესება ველოდებოდი.

| run | epochs | dropout | best val acc | overfit gap |
|---|---|---|---|---|
| smallcnn_base | 30 | 0.25 | 0.596 | 0.039 |

**დასკვნა:** val acc 45%-დან 59.6%-მდე გაიზარდა. gap = 0.039, train და val loss ერთად ეშვება. BatchNorm-ი სწავლებას აჩქარებს და regularization-ის ეფექტიც ჰქონდა.

---

### 3. DeeperCNN (no regularization): Overfitting-ის დემონსტრაცია

4 conv block, dropout=0, augmentation გამორთული. განზრახ ვტოვებ regularization-ის გარეშე, რომ overfit-ი ნათლად ჩანდეს.

| run | epochs | dropout | best val acc | overfit gap |
|---|---|---|---|---|
| deepcnn_no_reg | 40 | 0.0 | 0.650 | 0.379 |

**დასკვნა:** epoch 8-დან train loss ნულისკენ მიდის (0.009), val loss კი 2.7-მდე იზრდება. train acc = 99.7%, val acc = 61.8%. gap = 0.379, severe overfitting. მოდელს ბევრი capacity აქვს, regularization-ის გარეშე train set-ს ზეპირად ითვისებს.

---

### 4. DeeperCNN (regularized): Overfitting-ის გასწორება

იგივე 4 conv block, მაგრამ + dropout=0.4 + weight_decay=1e-4 + augmentation + cosine LR scheduler + use_class_weights=True.

| run | epochs | dropout | best val acc | overfit gap |
|---|---|---|---|---|
| deepcnn_reg_aug | 60 | 0.4 | 0.692 | 0.077 |

**დასკვნა:** gap 0.379-დან 0.077-მდე ჩამოვიდა. val loss სტაბილურია და არ იზრდება. cosine LR scheduler-მა სწავლება გაამართა, augmentation-მა კი მოდელი უფრო robust გახადა. confusion matrix-ზე Disgust კლასი ჯერ კიდევ სუსტია, მაგრამ class weights-მა ოდნავ გააუმჯობესა.

---

### Hyperparameter Sweep:

#### SmallCNN: Learning Rate Sweep

| lr | epochs | best val acc | დასკვნა |
|---|---|---|---|
| 0.01 | 25 | (wandb-ზე) | ძალიან სწრაფი, instable training |
| 0.001 | 25 | 0.596 | ოპტიმალური |
| 0.0001 | 25 | 0.599 | ნელა სწავლობს, 25 epoch-ში ვერ კონვერგირდება |

**დასკვნა:** lr=0.001 ოპტიმალურია SmallCNN-ისთვის. lr=0.0001 ოდნავ მაღალ val acc-ს (0.599) იძლევა, მაგრამ epoch-ების გაზრდა დასჭირდება.

#### DeeperCNN: Dropout Sweep (augment=True, cosine LR, weight_decay=1e-4)

| dropout | epochs | best val acc | overfit gap | დასკვნა |
|---|---|---|---|---|
| 0.0 | 40 | 0.705 | 0.256 | overfit, train acc 96% |
| 0.3 | 40 | 0.698 | 0.120 | კარგი კომპრომისი |
| 0.5 | 40 | 0.691 | 0.070 | gap მინიმალური, ოდნავ underfit |

**დასკვნა:** dropout=0.0 ყველაზე მაღალ val acc-ს (70.5%) იძლევა, მაგრამ gap=0.256 დიდია. dropout=0.5 gap-ს 0.070-მდე ამცირებს, acc კი ოდნავ ეცემა. dropout=0.3 კარგი კომპრომისია.

---

### Model Comparison: საბოლოო შედარება:

| მოდელი | best val acc | overfit gap | შეფასება |
|---|---|---|---|
| **DeeperCNN (reg)** | **0.692** | 0.077 | **საუკეთესო ბალანსი** |
| DeeperCNN (no reg) | 0.650 | 0.379 | severe overfitting |
| SmallCNN | 0.596 | 0.039 | კარგი ბალანსი, ნაკლები capacity |
| BaselineMLP | 0.455 | 0.483 | spatial info იკარგება |

---

## Wandb Tracking:

იმის ნაცვლად, რომ სხვადასხვა hyperparameter-ით მიღებული შედეგები ხელით ჩაგვეწერა, გამოვიყენე Weights & Biases. ყველა run ავტომატურად ლოგირდება. ეს საშუალებას გვაძლევს ერთიან სივრცეში შევადაროთ ყველა ვარიანტი.

project: fer2013-emotion-recognition (ერთი, ყველაფრისთვის).
group: arch (= MLflow-ის experiment), მაგ SmallCNN, DeeperCNN.
run: კონკრეტული ჰიპერპარამეტრების კონფიგი (= MLflow-ის run).

### ჩაწერილი მეტრიკების აღწერა:

- **train/acc:** accuracy სასწავლო სეტზე.
- **val/acc:** accuracy ვალიდაციის სეტზე, ეს ყველაზე მნიშვნელოვანი მეტრიკაა.
- **overfit_gap:** train_acc - val_acc. პირდაპირ ზომავს overfitting-ის სიძლიერეს.
- **gradient histograms:** wandb.watch(log="all")-ით ვლოგავ backward pass-ის მონიტორინგისთვის.
- **confusion_matrix:** val set-ზე, კლასების შეცდომების სადემონსტრაციოდ.

### საუკეთესო მოდელის შედეგები:

- **Model:** DeeperCNN (regularized)
- **Config:** dropout=0.4, weight_decay=1e-4, augment=True, scheduler=cosine, epochs=60
- **Best Val Acc:** 0.692
- **Overfit Gap:** 0.077
- **Num Params:** 3,512,007

---

## ბმულები:

- **Wandb Project:** https://wandb.ai/gsich23-free-university-of-tbilisi-/fer2013-emotion-recognition
- **Kaggle Dataset:** https://www.kaggle.com/datasets/deadskull7/fer2013
- **Kaggle Competition:** https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge