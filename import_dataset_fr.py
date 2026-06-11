import pandas as pd
import os
from fuzzywuzzy import process, fuzz
from pathlib import Path

# ===== 1. CHARGER LES DEUX JEUX DE DONNÉES =====

# Chemin vers le premier dataset
dataset1_path = r"C:\Users\pparr\Documents\Henallux\Henallux PP 2025_2026\Semester 2\Systèmes_intelligents\Dataset\Car Brand Classification Dataset\train"

# Chemin vers le deuxième dataset
dataset2_path = r"C:\Users\pparr\Documents\Henallux\Henallux PP 2025_2026\Semester 2\Systèmes_intelligents\Dataset\training-car\training_set"


def create_dataframe_from_folders(root_path):
    
    #Crée un DataFrame à partir d'une structure de dossiers.

    #Principe :
    #- chaque dossier représente une marque de voiture ;
    #- chaque fichier image dans ce dossier appartient à cette marque.

    data = []

    # Parcourir tous les dossiers présents dans le chemin principal
    for brand_folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, brand_folder)

        # Vérifier que l'élément est bien un dossier
        if os.path.isdir(folder_path):

            # Parcourir tous les fichiers dans le dossier de la marque
            for image_file in os.listdir(folder_path):

                # Garder uniquement les fichiers images
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):

                    # Ajouter les informations de l'image dans la liste
                    data.append({
                        'brand': brand_folder,
                        'image_file': image_file,
                        'full_path': os.path.join(folder_path, image_file),
                        'dataset_source': 'dataset1'
                    })

    # Convertir la liste de dictionnaires en DataFrame pandas
    return pd.DataFrame(data)


def create_dataframe_from_folders_v2(root_path, source_name='dataset'):
    
    #Crée un DataFrame à partir d'une structure de dossiers.

    #Cette version permet de choisir le nom de la source du dataset.
    

    data = []

    # Parcourir tous les dossiers présents dans le chemin principal
    for brand_folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, brand_folder)

        # Vérifier que l'élément est bien un dossier
        if os.path.isdir(folder_path):

            # Parcourir tous les fichiers dans le dossier de la marque
            for image_file in os.listdir(folder_path):

                # Garder uniquement les fichiers images
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):

                    # Ajouter les informations de l'image dans la liste
                    data.append({
                        'brand': brand_folder,
                        'image_file': image_file,
                        'full_path': os.path.join(folder_path, image_file),
                        'dataset_source': source_name
                    })

    # Convertir la liste de dictionnaires en DataFrame pandas
    return pd.DataFrame(data)


# Charger le premier dataset
print("Chargement du Dataset 1...")
df1 = create_dataframe_from_folders(dataset1_path)
print(f"Dataset 1 : {len(df1)} images")
print(f"Marques : {df1['brand'].unique()}\n")

# Charger le deuxième dataset
print("Chargement du Dataset 2...")
df2 = create_dataframe_from_folders_v2(dataset2_path, source_name='dataset2')
print(f"Dataset 2 : {len(df2)} images")
print(f"Marques : {df2['brand'].unique()}")


# ===== 2. FUSIONNER LES DEUX DATASETS =====

# Combiner les deux DataFrames en un seul
df_combined = pd.concat([df1, df2], ignore_index=True)

print(f"Datasets combinés : {len(df_combined)} images")
print(f"Toutes les marques AVANT nettoyage :\n{df_combined['brand'].value_counts()}\n")


# ===== 3. NORMALISER LES NOMS DES MARQUES =====

def normalize_brand(name):
    
    #Uniformise les noms des marques.

    #Exemple :
    #- 'Mercedes Benz' devient 'mercedes'
    #- 'VW' devient 'volkswagen'
    

    # Si le nom est vide ou manquant, on le retourne tel quel
    if pd.isna(name):
        return name

    # Convertir le nom en minuscules et supprimer les espaces inutiles
    name = str(name).lower().strip()

    # Dictionnaire des variantes fréquentes à remplacer
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

    # Vérifier si une variante connue est présente dans le nom
    for alt, standard in replacements.items():
        if alt in name:
            return standard

    # Si aucune variante n'est trouvée, retourner le nom nettoyé
    return name


# Appliquer la normalisation à toutes les marques
df_combined['brand_normalized'] = df_combined['brand'].apply(normalize_brand)

print("Noms des marques AVANT et APRÈS normalisation :")
print(df_combined[['brand', 'brand_normalized']].drop_duplicates().sort_values('brand'))


# ===== 4. UTILISER LE FUZZY MATCHING POUR UNE MEILLEURE UNIFORMISATION =====

# Récupérer la liste des marques uniques après normalisation
unique_brands = df_combined['brand_normalized'].dropna().unique().tolist()


def fuzzy_match_brands(brand, choices=unique_brands, threshold=85):
    
    #Compare les noms de marques de manière approximative.

    #Cela permet de corriger certains noms proches ou contenant des fautes.
    #Exemple :
    #- 'mercedez' peut être associé à 'mercedes'
    

    # Si la marque est vide ou manquante, on la retourne telle quelle
    if pd.isna(brand):
        return brand

    # Trouver la marque la plus proche dans la liste des marques existantes
    match, score = process.extractOne(
        brand,
        choices,
        scorer=fuzz.token_set_ratio
    )

    # Si le score de similarité est assez élevé, utiliser la marque trouvée
    if score >= threshold:
        return match

    # Sinon, garder le nom original
    return brand


# Appliquer le fuzzy matching aux marques normalisées
df_combined['brand_matched'] = df_combined['brand_normalized'].apply(fuzzy_match_brands)

print("Après fuzzy matching :")
print(df_combined['brand_matched'].value_counts())


# ===== 5. DÉTECTER ET ÉVENTUELLEMENT SUPPRIMER LES DOUBLONS =====

print(f"AVANT : {len(df_combined)} lignes")

# Compter les lignes complètement identiques
exact_dupes = df_combined.duplicated().sum()
print(f"Doublons exacts : {exact_dupes}")

# Compter les doublons ayant la même marque et le même nom de fichier
dupes_by_file = df_combined.duplicated(
    subset=['brand_matched', 'image_file'],
    keep='first'
).sum()

print(f"Doublons avec même fichier dans différents dossiers : {dupes_by_file}")

# Supprimer les doublons exacts si nécessaire
# df_combined = df_combined.drop_duplicates()

# Supprimer les doublons basés sur la marque et le nom du fichier si nécessaire
# df_combined = df_combined.drop_duplicates(
#     subset=['brand_matched', 'image_file'],
#     keep='first'
# )

print(f"APRÈS : {len(df_combined)} lignes")
print(f"\nSTATISTIQUE FINALE PAR MARQUE :")
print(df_combined['brand_matched'].value_counts().sort_values(ascending=False))


# ===== 6. PRÉPARER, SAUVEGARDER ET AFFICHER LES RÉSULTATS =====

# Créer un DataFrame final avec uniquement les colonnes utiles
df_final = df_combined[
    ['brand_matched', 'image_file', 'full_path', 'dataset_source']
].copy()

# Renommer les colonnes pour obtenir des noms plus simples
df_final.columns = ['brand', 'image_file', 'path', 'source']

# Chemin de sortie du fichier CSV final
output_path = r"C:\Users\pparr\Documents\Henallux\Henallux PP 2025_2026\Semester 2\Systèmes_intelligents\Deep_learning_projet_SN_PP\combined_dataset.csv"

# Sauvegarder le DataFrame final dans un fichier CSV
df_final.to_csv(output_path, index=False)

print(f"✓ Sauvegardé : {output_path}\n")

# Afficher un résumé du dataset final
print("APERÇU FINAL DU DATASET :")
print(f"Nombre total d'images : {len(df_final)}")
print(f"Nombre de marques : {df_final['brand'].nunique()}")

print(f"\nImages par marque :")
print(df_final['brand'].value_counts())

# Afficher les 10 premières lignes du DataFrame final
print(f"\n10 premières lignes :")
print(df_final.head(10))