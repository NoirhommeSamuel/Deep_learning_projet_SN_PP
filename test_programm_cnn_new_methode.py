import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchsummary import summary


#############################
# 1. Transforms definieren  #
#############################

image_size = (264, 264)  # wie in deinem Fruit-Notebook

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        fill=(255, 255, 255)
    ),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.8, 1.2)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])


###################################
# 2. Dataset (wie vorher)         #
###################################

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

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


#########################################
# 3. Hilfsfunktionen                    #
#########################################

def print_original_distribution(df, label_encoder, title):
    print(f"\n{title}")
    counts = df["label"].value_counts().sort_index()
    for label_idx, count in counts.items():
        brand_name = label_encoder.inverse_transform([label_idx])[0]
        print(f"{brand_name:20s} -> {count}")


def count_classes_in_loader(data_loader, label_encoder):
    counter = Counter()
    for _, labels in data_loader:
        counter.update(labels.tolist())

    print("\nVerteilung nach intelligentem Sampling (eine Epoche):")
    for label_idx, count in sorted(counter.items()):
        brand_name = label_encoder.inverse_transform([label_idx])[0]
        print(f"{brand_name:20s} -> {count}")


def save_augmented_samples(dataset, label_encoder, output_dir, num_images=5, seed=42):
    output_dir.mkdir(exist_ok=True)
    generator = torch.Generator()
    generator.manual_seed(seed)

    if hasattr(dataset, 'df') and 'label' in dataset.df.columns:
        # Verwende das gleiche Sampling-Gewicht wie beim Training, um intelligente Auswahl zu demonstrieren
        class_counts = dataset.df['label'].value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = torch.tensor(dataset.df['label'].map(class_weights).to_numpy(copy=True), dtype=torch.double)
        indices = torch.multinomial(sample_weights, num_samples=num_images, replacement=True, generator=generator)
    else:
        indices = torch.randperm(len(dataset), generator=generator)[:num_images]

    to_pil = transforms.ToPILImage()
    for i, idx in enumerate(indices.tolist(), start=1):
        image, label = dataset[idx]
        if isinstance(image, torch.Tensor):
            image = to_pil(image.cpu())

        brand_name = label_encoder.inverse_transform([int(label)])[0]
        image.save(output_dir / f"augmented_image_{i}_{brand_name}.png")

    print(f"{num_images} intelligent angepasste Bilder gespeichert in {output_dir}")


#########################################
# 4. CNN-Modell (angepasst ans Fruit-Notebook) #
#########################################

class CarBrandClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CarBrandClassifier, self).__init__()
        
        # CNN-Layer (ähnlich FruitClassifier)
        #self.conv1 = nn.Conv2d(3, 8, kernel_size=3)      # Input: 3x100x100 -> 8x98x98
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout1 = nn.Dropout(0.1)
        
        #self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)     # -> 64x47x47
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout2 = nn.Dropout(0.1)
        
        #self.conv3 = nn.Conv2d(16, 32, kernel_size=3)    # -> 128x21x21  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)  
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.1)
        
        # Adaptive Pooling: unabhängig von Input-Größe immer 256x4x4 Output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
      
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 400)         # 256x4x4 nach letztem Pool (4096 Features)
        self.dropout_fc1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(400, 400)
        self.dropout5 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(400, num_classes)           # Output: num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        #x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = self.adaptive_pool(x)  # Macht unabhängig von Input-Größe

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)  # Raw Logits
        return x


#########################################
# 5. Hauptprogramm                      #
#########################################

def main():
    csv_path = "combined_dataset_2.csv"
    batch_size = 64
    epochs = 200
    test_size = 0.2
    learning_rate = 0.001

    num_workers = 12  # Windows-sicher
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf: {device}")

    ###############################################
    # Daten laden                                 #
    ###############################################

    df = pd.read_csv(csv_path)
    label_encoder = preprocessing.LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["brand"])

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    num_classes = len(label_encoder.classes_)
    print(f"Anzahl Klassen: {num_classes}")

    # Sampling für Trainings-Set
    class_counts = train_df["label"].value_counts().sort_index()
    max_count = class_counts.max()
    n_classes = len(class_counts)
    num_samples_per_epoch = int(max_count * n_classes)

    class_weights = 1.0 / class_counts
    sample_weights = train_df["label"].map(class_weights).to_numpy(copy=True)
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples_per_epoch,
        replacement=True
    )

    # DataLoader
    train_dataset = CarBrandDataset(train_df, transform=train_transform)
    val_dataset = CarBrandDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    save_augmented_samples(train_dataset, label_encoder, output_dir, num_images=5)

    print_original_distribution(train_df, label_encoder, "Trainingsset vor Sampling:")
    count_classes_in_loader(train_loader, label_encoder)

    #########################################
    # 6. Modell, Loss, Optimizer            #
    #########################################

    model = CarBrandClassifier(num_classes).to(device)
    print("\nModell-Architektur:")
    summary(model, input_size=(3, 100, 100))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #########################################
    # 7. Training                           #
    #########################################

    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

    #########################################
    # 8. Evaluation                         #
    #########################################

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * val_correct / val_total

    print(f"\nValidierung:")
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    # Output-Ordner erstellen
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_[:15],  
                yticklabels=label_encoder.classes_[:15])
    plt.title("Confusion Matrix (erste 15 Klassen)")
    plt.xlabel("Vorhersage")
    plt.ylabel("Wahrheit")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    val_loss_history = [val_loss] * len(loss_history)  # Val-Loss pro Epoch (konstant, da nur am Ende gemessen)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=label_encoder.classes_))

        # Training Curves: Train/Val Loss + Train Accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Train & Val Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(loss_history, label='Train Loss', color='tab:blue', linewidth=2)
    ax1.plot(val_loss_history, label='Val Loss', color='tab:red', linewidth=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Train & Val Loss')
    
    # Plot 2: Train Accuracy
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)', color='tab:orange')
    ax2.plot(accuracy_history, label='Train Accuracy', color='tab:orange', linewidth=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Train Accuracy')
    
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves_complete.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # Modell speichern
    torch.save(model.state_dict(), "car_brand_classifier.pth")
    print("\nModell als 'car_brand_classifier.pth' gespeichert!")


if __name__ == "__main__":
    main()