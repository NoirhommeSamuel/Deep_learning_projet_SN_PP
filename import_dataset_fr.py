# Importe pandas pour manipuler des tableaux de données, ici des DataFrames.
import pandas as pd

# Importe os pour parcourir les dossiers et fichiers du système.
import os

# Importe fuzzywuzzy pour comparer des chaînes de caractères approximativement.
# process permet de chercher la meilleure correspondance.
# fuzz contient les méthodes de comparaison.
from fuzzywuzzy import process, fuzz

# Importe Path pour gérer les chemins de fichiers de manière plus propre.
from pathlib import Path


# ===== 1. CHARGER LES TROIS DATASETS =====

# Chemin vers le premier dataset, ici le dossier train.
dataset1_path = Path("/srv/groups/group2/data/Data_opt/Dataset_opt/Car Brand Classification Dataset opt/train")

# Chemin vers le deuxième dataset, ici le dossier val.
dataset2_path = Path("/srv/groups/group2/data/Data_opt/Dataset_opt/Car Brand Classification Dataset opt/val")

# Chemin vers le troisième dataset.
dataset3_path = Path("/srv/groups/group2/data/Data_opt/Dataset_opt/training-car opt/training_set")


def create_dataframe_from_folders(root_path, source_name='dataset1'):
    """
    Crée un DataFrame à partir d'une structure de dossiers.

    Structure attendue :
    root_path/
        BMW/
            image1.jpg
            image2.jpg
        Audi/
            image3.jpg

    Chaque dossier représente une marque.
    Chaque image dans ce dossier est associée à cette marque.
    """

    # Liste qui contiendra les informations de toutes les images.
    data = []

    # Vérifie si le chemin existe.
    # Si le chemin n'existe pas, on arrête le programme avec une erreur claire.
    if not root_path.exists():
        raise FileNotFoundError(f"Pfad nicht gefunden: {root_path}")

    # Parcourt tous les éléments dans le dossier principal.
    # Chaque élément devrait être un dossier de marque.
    for brand_folder in os.listdir(root_path):

        # Crée le chemin complet vers le dossier de la marque.
        folder_path = root_path / brand_folder

        # Vérifie que l'élément est bien un dossier.
        if folder_path.is_dir():

            # Parcourt tous les fichiers dans le dossier de cette marque.
            for image_file in os.listdir(folder_path):

                # Vérifie si le fichier est une image avec une extension autorisée.
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):

                    # Ajoute une ligne au futur DataFrame.
                    data.append({
                        # Nom de la marque, pris depuis le nom du dossier.
                        'brand': brand_folder,

                        # Nom du fichier image.
                        'image_file': image_file,

                        # Chemin complet vers l'image.
                        'full_path': str(folder_path / image_file),

                        # Source du dataset, par exemple dataset1, dataset2 ou dataset3.
                        'dataset_source': source_name
                    })

    # Convertit la liste de dictionnaires en DataFrame pandas.
    return pd.DataFrame(data)


def create_dataframe_from_folders_v2(root_path, source_name='dataset'):
    """
    Version alternative pour créer un DataFrame depuis une structure de dossiers.

    Dans ce code, cette fonction fait pratiquement la même chose que
    create_dataframe_from_folders.
    """

    # Liste qui contiendra toutes les lignes du DataFrame.
    data = []

    # Vérifie si le chemin existe.
    if not root_path.exists():
        raise FileNotFoundError(f"Pfad nicht gefunden: {root_path}")

    # Parcourt les dossiers de marques.
    for brand_folder in os.listdir(root_path):

        # Chemin complet vers le dossier de marque.
        folder_path = root_path / brand_folder

        # Vérifie que c'est un dossier.
        if folder_path.is_dir():

            # Parcourt les fichiers images du dossier.
            for image_file in os.listdir(folder_path):

                # Ne garde que les images jpg, jpeg ou png.
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):

                    # Ajoute les informations de l'image.
                    data.append({
                        'brand': brand_folder,
                        'image_file': image_file,
                        'full_path': str(folder_path / image_file),
                        'dataset_source': source_name
                    })

    # Retourne le DataFrame final.
    return pd.DataFrame(data)


# Affiche un message indiquant que les chemins vont être vérifiés.
print("Prüfe Pfade...")

# Vérifie pour chaque dataset si le chemin existe et si c'est bien un dossier.
for p in [dataset1_path, dataset2_path, dataset3_path]:
    print(p, "->", p.exists(), p.is_dir())

# Ligne vide pour rendre l'affichage plus lisible.
print()


# Charge le premier dataset.
print("Lade Dataset 1...")
df1 = create_dataframe_from_folders(dataset1_path, source_name='dataset1')

# Affiche le nombre d'images trouvées dans le dataset 1.
print(f"Dataset 1: {len(df1)} Bilder")

# Affiche les marques différentes trouvées dans le dataset 1.
print(f"Marken: {df1['brand'].unique()}\n")


# Charge le deuxième dataset.
print("Lade Dataset 2...")
df2 = create_dataframe_from_folders_v2(dataset2_path, source_name='dataset2')

# Affiche le nombre d'images trouvées.
print(f"Dataset 2: {len(df2)} Bilder")

# Affiche les marques du dataset 2.
print(f"Marken: {df2['brand'].unique()}\n")


# Charge le troisième dataset.
print("Lade Dataset 3...")
df3 = create_dataframe_from_folders_v2(dataset3_path, source_name='dataset3')

# Affiche le nombre d'images trouvées.
print(f"Dataset 3: {len(df3)} Bilder")

# Affiche les marques du dataset 3.
print(f"Marken: {df3['brand'].unique()}")


# ===== 2. FUSIONNER LES DATASETS =====

# Combine les trois DataFrames en un seul.
# ignore_index=True recrée un index propre de 0 à n-1.
df_combined = pd.concat([df1, df2, df3], ignore_index=True)

# Affiche le nombre total d'images après fusion.
print(f"Kombinierte Datensätze: {len(df_combined)} Bilder")

# Affiche le nombre d'images par marque avant nettoyage.
print(f"Alle Marken VOR Bereinigung:\n{df_combined['brand'].value_counts()}\n")


# ===== 3. NORMALISER LES NOMS DES MARQUES =====

def normalize_brand(name):
    """
    Uniformise les noms des marques.

    Exemple :
    'Mercedes-Benz', 'mercedes benz' et 'benz'
    deviennent tous 'mercedes'.
    """

    # Si le nom est vide ou manquant, on le retourne tel quel.
    if pd.isna(name):
        return name

    # Convertit le nom en texte, le met en minuscules
    # et enlève les espaces au début et à la fin.
    name = str(name).lower().strip()

    # Dictionnaire des variantes connues.
    # La clé est une variante possible, la valeur est le nom standard.
    replacements = {
        "mercedes benz": "mercedes",
        "mercedes-benz": "mercedes",
        "benz": "mercedes",
        "bmw": "bmw",
        "volkswagen": "volkswagen",
        "vw": "volkswagen",
        "audi": "audi",
        "porsche": "porsche",
        "volvo": "volvo",
        "ford": "ford",
        "chevrolet": "chevrolet",
        "tesla": "tesla",
        "toyota": "toyota",
        "honda": "honda",
        "hyundai": "hyundai",
        "kia": "kia",
    }

    # Parcourt toutes les variantes connues.
    for alt, standard in replacements.items():

        # Si la variante apparaît dans le nom,
        # on retourne le nom standardisé.
        if alt in name:
            return standard

    # Si aucune variante n'est trouvée,
    # on retourne le nom nettoyé.
    return name


# Applique la fonction normalize_brand à chaque marque.
# Le résultat est stocké dans une nouvelle colonne.
df_combined['brand_normalized'] = df_combined['brand'].apply(normalize_brand)

# Affiche les noms avant et après normalisation.
print("Markennamen VOR und NACH Normalisierung:")
print(df_combined[['brand', 'brand_normalized']].drop_duplicates().sort_values('brand'))


# ===== 4. FUZZY MATCHING POUR UNIFIER ENCORE PLUS LES NOMS =====

# Récupère la liste des marques normalisées uniques.
unique_brands = df_combined['brand_normalized'].dropna().unique().tolist()


def fuzzy_match_brands(brand, choices=unique_brands, threshold=85):
    """
    Compare un nom de marque avec les autres noms existants.

    Si un nom ressemble fortement à un autre nom,
    il est remplacé par la meilleure correspondance.

    threshold=85 signifie qu'il faut au moins 85 % de similarité.
    """

    # Si la marque est vide, on la retourne telle quelle.
    if pd.isna(brand):
        return brand

    # Cherche la meilleure correspondance entre brand et choices.
    # fuzz.token_set_ratio est une méthode qui compare les mots.
    match, score = process.extractOne(brand, choices, scorer=fuzz.token_set_ratio)

    # Si le score est assez élevé, on utilise la correspondance trouvée.
    # Sinon, on garde le nom original.
    return match if score >= threshold else brand


# Applique le fuzzy matching à toutes les marques normalisées.
df_combined['brand_matched'] = df_combined['brand_normalized'].apply(fuzzy_match_brands)

# Affiche le nombre d'images par marque après fuzzy matching.
print("Nach Fuzzy Matching:")
print(df_combined['brand_matched'].value_counts())


# ===== 5. DÉTECTER ET ÉVENTUELLEMENT SUPPRIMER LES DOUBLONS =====

# Affiche le nombre de lignes avant suppression des doublons.
print(f"VORHER: {len(df_combined)} Zeilen")

# Calcule le nombre de lignes exactement identiques.
exact_dupes = df_combined.duplicated().sum()
print(f"Exakte Duplikate: {exact_dupes}")

# Calcule les doublons basés sur la marque et le nom du fichier.
# Cela détecte potentiellement la même image présente dans plusieurs dossiers.
dupes_by_file = df_combined.duplicated(
    subset=['brand_matched', 'image_file'],
    keep='first'
).sum()

print(f"Duplikate (same file in different folders): {dupes_by_file}")

# Ces deux lignes sont commentées.
# Donc les doublons sont seulement détectés, mais pas supprimés.

# Supprimerait les lignes entièrement identiques.
# df_combined = df_combined.drop_duplicates()

# Supprimerait les doublons ayant la même marque et le même nom de fichier.
# df_combined = df_combined.drop_duplicates(subset=['brand_matched', 'image_file'], keep='first')

# Comme la suppression est commentée, le nombre de lignes reste identique.
print(f"NACHHER: {len(df_combined)} Zeilen")

# Affiche la statistique finale par marque.
print(f"\nFINALE STATISTIK PRO MARKE:")
print(df_combined['brand_matched'].value_counts().sort_values(ascending=False))


# ===== 6. SAUVEGARDER ET AFFICHER LE RÉSULTAT FINAL =====

# Prépare un DataFrame final avec uniquement les colonnes utiles.
df_final = df_combined[['brand_matched', 'image_file', 'full_path', 'dataset_source']].copy()

# Renomme les colonnes pour les adapter au code d'entraînement CNN.
df_final.columns = ['brand', 'image_file', 'path', 'source']

# Définit le chemin de sortie du fichier CSV final.
output_path = Path("/srv/groups/group2/projects/combined_dataset_opt.csv")

# Sauvegarde le DataFrame final en CSV.
# index=False évite d'ajouter une colonne d'index inutile.
df_final.to_csv(output_path, index=False)

# Confirme que le fichier a été sauvegardé.
print(f"✓ Gespeichert: {output_path}\n")

# Affiche un résumé du dataset final.
print("FINALE DATASET ÜBERSICHT:")

# Nombre total d'images.
print(f"Gesamte Bilder: {len(df_final)}")

# Nombre de marques différentes.
print(f"Anzahl Marken: {df_final['brand'].nunique()}")

# Nombre d'images par marque.
print(f"\nBilder pro Marke:")
print(df_final['brand'].value_counts())

# Affiche les 10 premières lignes du DataFrame final.
print(f"\nErste 10 Zeilen:")
print(df_final.head(10))