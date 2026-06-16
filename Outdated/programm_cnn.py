import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchsummary import summary


#############################
# 1. Reproduzierbarkeit     #
#############################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


#############################
# 2. Einstellungen          #
#############################

CSV_PATH = "combined_dataset_opt.csv"

BATCH_SIZE = 32
EPOCHS = 150
TEST_SIZE = 0.2

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 12
PIN_MEMORY = True

IMAGE_SIZE = (256, 256)

# MixUp / CutMix bewusst weniger aggressiv
USE_MIXUP = False
USE_CUTMIX = False
MIX_PROB = 0.00
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0

# Label smoothing reduziert
USE_LABEL_SMOOTHING = False
LABEL_SMOOTHING = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################
# 3. Transforms             #
#############################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
        ratio=(0.9, 1.1),
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


#############################
# 4. Dataset                #
#############################

class CarBrandDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden von Bild: {img_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label


#############################
# 5. Hilfsfunktionen        #
#############################

def print_original_distribution(df, label_encoder, title):
    print(f"\n{title}")
    counts = df["label"].value_counts().sort_index()
    for label_idx, count in counts.items():
        brand_name = label_encoder.inverse_transform([label_idx])[0]
        print(f"{brand_name:20s} -> {count}")


#############################
# 6. Modell                 #
#############################

class CarBrandClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CarBrandClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            # zusätzliche Kapazität
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


#############################
# 7. Early Stopping         #
#############################

class EarlyStopping:
    def __init__(self, patience=18, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_state_dict = None

    def step(self, metric, model):
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.counter = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


#############################
# 8. MixUp / CutMix         #
#############################

def rand_bbox(size, lam):
    H = size[2]
    W = size[3]

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    y_a, y_b = y, y[index]

    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#############################
# 9. Training / Evaluation  #
#############################

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        use_mixed = np.random.rand() < MIX_PROB

        if use_mixed and (USE_MIXUP or USE_CUTMIX):
            if USE_MIXUP and USE_CUTMIX:
                use_cutmix = np.random.rand() < 0.5
            else:
                use_cutmix = USE_CUTMIX

            if use_cutmix:
                images, y_a, y_b, lam = cutmix_data(images, labels, alpha=CUTMIX_ALPHA)
            else:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)

            outputs = model(images)
            loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


#############################
# 10. Hauptprogramm         #
#############################

def main():
    print(f"Training auf: {DEVICE}")

    df = pd.read_csv(CSV_PATH)

    label_encoder = preprocessing.LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["brand"])

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    num_classes = len(label_encoder.classes_)
    print(f"Anzahl Klassen: {num_classes}")

    print_original_distribution(train_df, label_encoder, "Trainingsset (original):")
    print_original_distribution(val_df, label_encoder, "Validierungsset:")

    train_dataset = CarBrandDataset(train_df, transform=train_transform)
    val_dataset = CarBrandDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = CarBrandClassifier(num_classes).to(DEVICE)

    print("\nModell-Architektur:")
    summary(model, input_size=(3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

    # gewichtete Loss statt Sampler
    class_counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    if USE_LABEL_SMOOTHING:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    early_stopping = EarlyStopping(patience=18, min_delta=0.0)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f"Lernrate reduziert: {old_lr:.6f} -> {new_lr:.6f}")

        early_stopping.step(val_acc, model)

        if early_stopping.should_stop:
            print(f"Early Stopping in Epoch {epoch+1} (beste Val-Acc: {early_stopping.best_score:.2f}%)")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    val_loss, val_acc, all_preds, all_labels = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE
    )

    print(f"\nFinale Validierung:")
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Confusion Matrix (alle Klassen)")
    plt.xlabel("Vorhersage")
    plt.ylabel("Wahrheit")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_dataset_opt.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    epochs_range = range(1, len(train_loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs_range, train_loss_history, label="Train Loss", color="tab:blue")
    ax1.plot(epochs_range, val_loss_history, label="Val Loss", color="tab:cyan", linestyle="--")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:orange")
    ax2.plot(epochs_range, train_acc_history, label="Train Acc", color="tab:orange")
    ax2.plot(epochs_range, val_acc_history, label="Val Acc", color="tab:red", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    plt.title("Training & Validation Loss/Accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves_dataset_opt.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    torch.save(model.state_dict(), "car_brand_classifier_weightedloss_512cnn_dataset_opt.pth")
    print("\nBestes Modell als 'car_brand_classifier_weightedloss_512cnn.pth' gespeichert!")
    print(f"Plots gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()