import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


#############################
# 1. Einstellungen
#############################

MODEL_PATH = "cnn_full_train_100_percent.pth"
TRAIN_CSV_PATH = "combined_dataset_opt.csv"

TEST_PATHS = [
    Path("/srv/groups/group2/data/Data_opt/Dataset_opt/Car Brand Classification Dataset opt/test"),
    Path("/srv/groups/group2/data/Data_opt/Dataset_opt/training-car opt/test_set"),
]

BATCH_SIZE = 32
NUM_WORKERS = 12
PIN_MEMORY = True
IMAGE_SIZE = (256, 256)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################
# 2. Transforms
#############################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


#############################
# 3. Modell
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
# 4. Hilfsfunktionen
#############################

def create_dataframe_from_test_folders(root_path, valid_brands):
    data = []

    if not root_path.exists():
        print(f"Warnung: Pfad nicht gefunden: {root_path}")
        return pd.DataFrame(columns=["brand", "image_file", "path", "dataset_source"])

    for brand_folder in os.listdir(root_path):
        folder_path = root_path / brand_folder

        if folder_path.is_dir():
            if brand_folder not in valid_brands:
                print(f"Überspringe unbekannte Marke im Testset: {brand_folder}")
                continue

            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    data.append({
                        "brand": brand_folder,
                        "image_file": image_file,
                        "path": str(folder_path / image_file),
                        "dataset_source": str(root_path)
                    })

    return pd.DataFrame(data)


class CarBrandDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
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


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


#############################
# 5. Hauptprogramm
#############################

def main():
    print(f"Test auf: {DEVICE}")

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    required_cols = {"path", "brand"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError(f"Trainings-CSV muss diese Spalten enthalten: {required_cols}")

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_df["brand"])

    valid_brands = set(label_encoder.classes_)
    num_classes = len(label_encoder.classes_)

    print(f"Anzahl Klassen aus Trainings-CSV: {num_classes}")

    test_dfs = []
    for test_path in TEST_PATHS:
        df_part = create_dataframe_from_test_folders(test_path, valid_brands)
        print(f"{test_path} -> {len(df_part)} Bilder")
        test_dfs.append(df_part)

    test_df = pd.concat(test_dfs, ignore_index=True)

    if len(test_df) == 0:
        raise ValueError("Kein Testbild gefunden. Bitte Testpfade prüfen.")

    test_df["label"] = label_encoder.transform(test_df["brand"])

    print(f"\nGesamtes Testset: {len(test_df)} Bilder")
    print("Klassenverteilung im Testset:")
    print(test_df["brand"].value_counts().sort_index())

    test_dataset = CarBrandDataset(test_df, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0
    )

    model = CarBrandClassifier(num_classes).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"\nModell geladen: {MODEL_PATH}")

    all_preds, all_labels = evaluate(model, test_loader, DEVICE)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_
    ))

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
    plt.title("Confusion Matrix (Testset)")
    plt.xlabel("Vorhersage")
    plt.ylabel("Wahrheit")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_testset.png", dpi=300, bbox_inches="tight")
    plt.close()

    results_df = test_df.copy()
    results_df["true_label"] = all_labels
    results_df["pred_label"] = all_preds
    results_df["true_brand"] = label_encoder.inverse_transform(all_labels)
    results_df["pred_brand"] = label_encoder.inverse_transform(all_preds)
    results_df["correct"] = results_df["true_label"] == results_df["pred_label"]
    results_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\nConfusion Matrix gespeichert unter: {output_dir / 'confusion_matrix_testset.png'}")
    print(f"Vorhersagen gespeichert unter: {output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()