import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sklearn import preprocessing

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchsummary import summary


# Ignore certains avertissements venant de PIL lors du chargement des images
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


#############################
# 1. Reproductibilité
#############################

def set_seed(seed=42):
    
    #Fixe les graines aléatoires pour obtenir des résultats aussi reproductibles que possible entre plusieurs lancements du programme.

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


#############################
# 2. Paramètres généraux
#############################

CSV_PATH = "combined_dataset_opt.csv"

BATCH_SIZE = 32
EPOCHS = 150

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 12
PIN_MEMORY = True

IMAGE_SIZE = (256, 256)

# Utilise le GPU si disponible, sinon le CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################
# 3. Transformations des images
#############################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transformations appliquées aux images pendant l'entraînement
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    # Recadrage aléatoire pour rendre le modèle plus robuste
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
        ratio=(0.9, 1.1),
        interpolation=InterpolationMode.BILINEAR
    ),

    # Miroir horizontal aléatoire
    transforms.RandomHorizontalFlip(p=0.5),

    # Petite rotation aléatoire
    transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),

    # Variation légère des couleurs
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),

    # Conversion en tenseur PyTorch
    transforms.ToTensor(),

    # Normalisation avec les valeurs ImageNet
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


#############################
# 4. Dataset personnalisé
#############################

class CarBrandDataset(Dataset):
    
    #Dataset PyTorch pour charger les images de voitures et leurs labels.
    #Le CSV doit contenir au minimum les colonnes 'path' et 'brand'.
    

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
            raise RuntimeError(f"Erreur lors du chargement de l'image : {img_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label


#############################
# 5. Fonction d'affichage
#############################

def print_original_distribution(df, label_encoder, title):
    
    #Affiche le nombre d'images disponibles pour chaque marque.
    
    print(f"\n{title}")

    counts = df["label"].value_counts().sort_index()

    for label_idx, count in counts.items():
        brand_name = label_encoder.inverse_transform([label_idx])[0]
        print(f"{brand_name:20s} -> {count}")


#############################
# 6. Modèle CNN
#############################

class CarBrandClassifier(nn.Module):
    
    #Réseau de neurones convolutif pour classifier les marques de voitures.
    

    def __init__(self, num_classes):
        super(CarBrandClassifier, self).__init__()

        # Partie convolutionnelle : extraction des caractéristiques de l'image
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

        # Réduit chaque carte de caractéristiques à une taille 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Partie classification finale
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
# 7. Entraînement d'une époque
#############################

def train_one_epoch(model, loader, optimizer, criterion, device):
    
    #Entraîne le modèle pendant une époque complète.
    #Retourne la loss moyenne et la précision.
    
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

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


#############################
# 8. Programme principal
#############################

def main():
    print(f"Entraînement sur : {DEVICE}")

    # Chargement du fichier CSV
    df = pd.read_csv(CSV_PATH)

    # Vérifie que les colonnes nécessaires sont présentes
    required_cols = {"path", "brand"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Le CSV doit contenir ces colonnes : {required_cols}")

    # Transforme les noms de marques en labels numériques
    label_encoder = preprocessing.LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["brand"])

    train_df = df.copy().reset_index(drop=True)

    num_classes = len(label_encoder.classes_)
    print(f"Nombre de classes : {num_classes}")

    print_original_distribution(train_df, label_encoder, "Jeu d'entraînement (100 %) :")

    # Création du dataset et du dataloader
    train_dataset = CarBrandDataset(train_df, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0
    )

    # Création du modèle
    model = CarBrandClassifier(num_classes).to(DEVICE)

    print("\nArchitecture du modèle :")
    summary(model, input_size=(3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

    # Calcul des poids de classes pour compenser les déséquilibres du dataset
    class_counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    train_loss_history = []
    train_acc_history = []

    best_train_loss = float("inf")
    best_state_dict = None

    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        print(
            f"Époque {epoch + 1}/{EPOCHS} : "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%"
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Taux d'apprentissage actuel : {current_lr:.8f}")

        # Sauvegarde en mémoire du meilleur modèle selon la loss d'entraînement
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    # Recharge le meilleur modèle trouvé pendant l'entraînement
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Création du dossier de sortie
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Création du graphique loss / accuracy
    epochs_range = range(1, len(train_loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs_range, train_loss_history, label="Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:orange")
    ax2.plot(epochs_range, train_acc_history, label="Train Acc", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    plt.title("Loss / Accuracy pendant l'entraînement")
    fig.tight_layout()

    fig.savefig(
        output_dir / "training_curves_full_train_dataset_opt.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Sauvegarde finale du modèle entraîné
    torch.save(model.state_dict(), "cnn_full_train_100_percent.pth")

    print("\nMeilleur modèle sauvegardé sous : 'cnn_full_train_100_percent.pth'")
    print(f"Meilleure Train Loss : {best_train_loss:.4f}")
    print(f"Graphique sauvegardé dans : {output_dir}")


if __name__ == "__main__":
    main()