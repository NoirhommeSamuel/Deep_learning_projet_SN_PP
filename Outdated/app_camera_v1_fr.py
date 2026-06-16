import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from nicegui import ui, events


#############################
# 1. Paramètres de l'application
#############################

# Chemin vers le fichier du modèle entraîné
MODEL_PATH = Path('cnn_full_train_100_percent.pth')

# Utilise le GPU si disponible, sinon le CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Taille des images utilisée par le modèle
IMAGE_SIZE = (256, 256)

# Seuil minimum de confiance pour considérer une prédiction comme fiable
UNCERTAINTY_THRESHOLD = 0.45

# Valeurs utilisées pour normaliser les images comme pendant l'entraînement
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Liste des marques que le modèle peut reconnaître
TRAIN_CLASSES = [
    'acura',
    'aston martin',
    'audi',
    'bentley',
    'bmw',
    'buick',
    'cadillac',
    'chevrolet',
    'chrysler',
    'dodge',
    'ferrari',
    'fiat',
    'ford',
    'gmc',
    'honda',
    'hyundai',
    'infiniti',
    'jaguar',
    'jeep',
    'kia',
    'lamborghini',
    'land rover',
    'lexus',
    'lincoln',
    'mazda',
    'mercedes',
    'mini',
    'mitsubishi',
    'nissan',
    'porsche',
    'ram',
    'subaru',
    'toyota',
    'volkswagen',
    'volvo',
]

# Associe chaque indice numérique à une marque
IDX_TO_BRAND = {idx: brand for idx, brand in enumerate(TRAIN_CLASSES)}


#############################
# 2. Modèle CNN
#############################

class CarBrandClassifier(nn.Module):
    
    #Réseau de neurones convolutif utilisé pour reconnaître la marque d'une voiture à partir d'une image.
    

    def __init__(self, num_classes):
        super().__init__()

        # Partie convolutionnelle : extrait les caractéristiques visuelles de l'image
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

        # Réduit les caractéristiques à une taille fixe de 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Partie finale : transforme les caractéristiques en prédiction de marque
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        
        #Définit comment l'image traverse le modèle.
        
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


#############################
# 3. Prétraitement des images
#############################

# Transformations appliquées à l'image avant la prédiction
infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def pretty_brand_name(name: str) -> str:
    
    #Rend le nom de la marque plus lisible.
    #Exemple : 'aston martin' devient 'Aston Martin'.
    
    return ' '.join(part.capitalize() for part in name.split())


def pil_to_data_url(image: Image.Image) -> str:
    
    #Convertit une image PIL en texte base64.
    #Cela permet d'afficher directement l'image dans l'interface web.
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded}'


def load_model():
    
    #Charge le modèle entraîné depuis le fichier .pth.
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Fichier du modèle introuvable : {MODEL_PATH}')

    # Crée le modèle avec le bon nombre de classes
    model = CarBrandClassifier(num_classes=len(TRAIN_CLASSES)).to(DEVICE)

    # Charge les poids sauvegardés
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Charge les poids dans le modèle
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print('Clés manquantes :', missing)
    print('Clés inattendues :', unexpected)

    if missing or unexpected:
        print('ATTENTION : le modèle et le fichier sauvegardé ne correspondent pas parfaitement.')
    else:
        print('Modèle chargé avec succès.')

    # Met le modèle en mode évaluation
    model.eval()

    return model


# Charge le modèle une seule fois au lancement de l'application
MODEL = load_model()


@torch.inference_mode()
def predict_image(pil_image: Image.Image):
    
    #Prédit les trois marques les plus probables pour une image donnée.
    
    # Convertit l'image en RGB
    image = pil_image.convert('RGB')

    # Applique les transformations nécessaires et ajoute une dimension batch
    x = infer_transform(image).unsqueeze(0).to(DEVICE)

    # Calcule les sorties du modèle
    logits = MODEL(x)

    # Convertit les sorties en probabilités
    probs = torch.softmax(logits, dim=1)[0]

    # Récupère les trois meilleures prédictions
    top_probs, top_indices = torch.topk(probs, k=3)

    results = []

    # Transforme les résultats en liste lisible
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({
            'brand': IDX_TO_BRAND[idx],
            'confidence': prob,
        })

    return results


#############################
# 4. Style de l'interface
#############################

# Ajoute du CSS personnalisé pour améliorer l'apparence de l'application
ui.add_head_html('''
<style>
body {
    background:
        radial-gradient(circle at top left, rgba(1, 105, 111, 0.10), transparent 28%),
        linear-gradient(180deg, #f4f7f8 0%, #edf2f4 100%);
    color: #102a43;
}
.main-shell { max-width: 1180px; width: 100%; }
.glass-card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 24px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #102a43;
}
.hero-subtitle {
    color: #52606d;
    line-height: 1.7;
    max-width: 800px;
}
.preview-frame {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px dashed #cbd5e1;
    border-radius: 20px;
    min-height: 340px;
}
.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 18px;
    padding: 16px;
}
.metric-label { font-size: 0.9rem; color: #64748b; margin-bottom: 6px; }
.metric-value { font-size: 1.35rem; font-weight: 800; color: #0f172a; }
.metric-sub { font-size: 0.9rem; color: #64748b; margin-top: 4px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.92rem;
}
.status-safe { background: #eaf8ef; color: #166534; border: 1px solid #bbf7d0; }
.status-warn { background: #fff4e5; color: #b45309; border: 1px solid #fed7aa; }
.result-box {
    border-radius: 20px;
    padding: 18px;
    border: 1px solid rgba(15, 23, 42, 0.08);
}
.result-safe { background: linear-gradient(180deg, #ecfdf5 0%, #f8fffb 100%); }
.result-warn { background: linear-gradient(180deg, #fff7ed 0%, #fffdfa 100%); }
.result-title { font-size: 0.95rem; color: #64748b; margin-bottom: 6px; }
.result-brand { font-size: 1.8rem; font-weight: 800; color: #0f172a; }
.result-text { color: #475569; margin-top: 8px; line-height: 1.6; }
.info-box {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 18px;
    padding: 16px;
    color: #334155;
    line-height: 1.6;
}
.section-title { font-size: 1.15rem; font-weight: 700; color: #0f172a; }
.small-muted { color: #64748b; font-size: 0.92rem; }
.upload-note { color: #64748b; line-height: 1.6; }
.top-card {
    border-radius: 18px;
    padding: 14px 16px;
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: rgba(255,255,255,0.8);
}
.rank-number {
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: #e6f4f4;
    color: #01696f;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}
.brand-name { font-weight: 700; color: #0f172a; }
.brand-prob { color: #64748b; font-size: 0.92rem; }
</style>
''')


#############################
# 5. Interface utilisateur
#############################

# Définit le titre de la page dans le navigateur
ui.page_title('Car Brand Classifier')

# Conteneur principal de l'application
with ui.column().classes('w-full items-center px-4 py-8'):
    with ui.column().classes('main-shell gap-6'):

        # Carte d'introduction
        with ui.card().classes('glass-card w-full p-8 gap-4'):
            ui.label('Car Brand Classifier').classes('hero-title')
            ui.label(
                'Téléverse une photo de voiture ou prends directement une photo avec ton smartphone. '
                'L’application affiche la marque la plus probable, les 3 meilleurs résultats, '
                'la confiance du modèle et des informations pour interpréter la prédiction.'
            ).classes('hero-subtitle')

            # Informations générales sur le modèle
            with ui.row().classes('w-full gap-3 items-center'):
                ui.label(f'Modèle exécuté sur : {DEVICE}').classes('small-muted')
                ui.label(f'Nombre de marques : {len(TRAIN_CLASSES)}').classes('small-muted')
                ui.label(f'Incertain sous : {UNCERTAINTY_THRESHOLD * 100:.0f}%').classes('small-muted')

        # Grille principale : upload à gauche, analyse à droite
        with ui.grid(columns=2).classes('w-full gap-6 max-[900px]:grid-cols-1'):

            # Partie gauche : choix et prévisualisation de l'image
            with ui.card().classes('glass-card w-full p-6 gap-5'):
                ui.label('Téléverser une image').classes('section-title')

                # Zone d'affichage de l'image choisie
                with ui.element('div').classes('preview-frame w-full flex items-center justify-center overflow-hidden'):
                    preview = ui.image('').classes('w-full max-h-[420px] object-contain rounded-xl')
                    preview.set_visibility(False)
                    preview_hint = ui.label('Aucune image sélectionnée').classes('text-slate-400 text-base')

                # Animation de chargement pendant l'analyse
                spinner = ui.spinner(size='lg')
                spinner.set_visibility(False)

                ui.label(
                    'Tu peux choisir une image existante ou prendre une photo directement avec ton téléphone.'
                ).classes('upload-note')

                async def handle_upload(e: events.UploadEventArguments):
                    """
                    Fonction appelée automatiquement quand l'utilisateur envoie une image.
                    Elle lit l'image, l'affiche, lance la prédiction et met à jour les résultats.
                    """
                    try:
                        spinner.set_visibility(True)

                        # Lit le fichier envoyé par l'utilisateur
                        file_bytes = e.content.read()
                        pil_image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

                        # Affiche l'image dans l'interface
                        preview.set_source(pil_to_data_url(pil_image))
                        preview.set_visibility(True)
                        preview_hint.set_visibility(False)

                        # Lance la prédiction avec le modèle
                        results = predict_image(pil_image)

                        # Récupère la meilleure prédiction
                        best = results[0]
                        best_conf = best['confidence']

                        # Calcule l'écart entre la première et la deuxième prédiction
                        gap_top12 = results[0]['confidence'] - results[1]['confidence']

                        # Vérifie si la prédiction est considérée comme fiable
                        is_safe = best_conf >= UNCERTAINTY_THRESHOLD

                        # Met à jour la confiance affichée
                        confidence_value.set_text(f'{best_conf * 100:.2f}%')
                        confidence_bar.set_value(best_conf)

                        if is_safe:
                            status_badge.content = '<div class="status-pill status-safe">Prédiction plutôt fiable</div>'
                            confidence_hint.set_text(
                                f'La prédiction est au-dessus du seuil de {UNCERTAINTY_THRESHOLD * 100:.0f}%.'
                            )
                            result_box.content = f'''
                                <div class="result-box result-safe">
                                    <div class="result-title">Marque reconnue</div>
                                    <div class="result-brand">{pretty_brand_name(best["brand"])}</div>
                                    <div class="result-text">
                                        Le modèle considère cette marque comme la prédiction la plus probable.
                                    </div>
                                </div>
                            '''
                        else:
                            status_badge.content = '<div class="status-pill status-warn">Prédiction incertaine</div>'
                            confidence_hint.set_text(
                                f'La prédiction est sous le seuil de {UNCERTAINTY_THRESHOLD * 100:.0f}%.'
                            )
                            result_box.content = f'''
                                <div class="result-box result-warn">
                                    <div class="result-title">Meilleure estimation</div>
                                    <div class="result-brand">{pretty_brand_name(best["brand"])}</div>
                                    <div class="result-text">
                                        Le modèle n’est pas assez sûr. La marque affichée est seulement la meilleure estimation actuelle.
                                    </div>
                                </div>
                            '''

                        # Met à jour les cartes Top 1, Top 2 et Top 3
                        top1_card.content = f'''
                            <div class="metric-card">
                                <div class="metric-label">Top 1</div>
                                <div class="metric-value">{pretty_brand_name(results[0]["brand"])}</div>
                                <div class="metric-sub">{results[0]["confidence"] * 100:.2f}%</div>
                            </div>
                        '''
                        top2_card.content = f'''
                            <div class="metric-card">
                                <div class="metric-label">Top 2</div>
                                <div class="metric-value">{pretty_brand_name(results[1]["brand"])}</div>
                                <div class="metric-sub">{results[1]["confidence"] * 100:.2f}%</div>
                            </div>
                        '''
                        top3_card.content = f'''
                            <div class="metric-card">
                                <div class="metric-label">Top 3</div>
                                <div class="metric-value">{pretty_brand_name(results[2]["brand"])}</div>
                                <div class="metric-sub">{results[2]["confidence"] * 100:.2f}%</div>
                            </div>
                        '''
                        gap_card.content = f'''
                            <div class="metric-card">
                                <div class="metric-label">Écart Top 1 - Top 2</div>
                                <div class="metric-value">{gap_top12 * 100:.2f} points</div>
                                <div class="metric-sub">Un écart plus grand indique souvent une décision plus claire</div>
                            </div>
                        '''

                        # Met à jour le classement détaillé
                        ranking_1.content = f'''
                            <div class="top-card">
                                <div class="flex items-center gap-3">
                                    <div class="rank-number">1</div>
                                    <div>
                                        <div class="brand-name">{pretty_brand_name(results[0]["brand"])}</div>
                                        <div class="brand-prob">{results[0]["confidence"] * 100:.2f}%</div>
                                    </div>
                                </div>
                            </div>
                        '''
                        ranking_2.content = f'''
                            <div class="top-card">
                                <div class="flex items-center gap-3">
                                    <div class="rank-number">2</div>
                                    <div>
                                        <div class="brand-name">{pretty_brand_name(results[1]["brand"])}</div>
                                        <div class="brand-prob">{results[1]["confidence"] * 100:.2f}%</div>
                                    </div>
                                </div>
                            </div>
                        '''
                        ranking_3.content = f'''
                            <div class="top-card">
                                <div class="flex items-center gap-3">
                                    <div class="rank-number">3</div>
                                    <div>
                                        <div class="brand-name">{pretty_brand_name(results[2]["brand"])}</div>
                                        <div class="brand-prob">{results[2]["confidence"] * 100:.2f}%</div>
                                    </div>
                                </div>
                            </div>
                        '''

                        # Met à jour les informations d'interprétation
                        info_1.content = (
                            '<div class="info-box"><strong>Interprétation :</strong> '
                            f'Le modèle pense actuellement que la marque la plus probable est '
                            f'<strong>{pretty_brand_name(best["brand"])}</strong>.</div>'
                        )
                        info_2.content = (
                            '<div class="info-box"><strong>Règle d’incertitude :</strong> '
                            f'Sous {UNCERTAINTY_THRESHOLD * 100:.0f}%, le résultat est marqué comme incertain. '
                            f'Valeur actuelle : {best_conf * 100:.2f}%.</div>'
                        )
                        info_3.content = (
                            '<div class="info-box"><strong>Comparaison :</strong> '
                            f'L’écart entre le Top 1 et le Top 2 est de {gap_top12 * 100:.2f} points.</div>'
                        )
                        info_4.content = (
                            '<div class="info-box"><strong>Conseil pratique :</strong> '
                            'Une image bien éclairée avec une vue claire de l’avant ou du côté de la voiture améliore souvent la prédiction.</div>'
                        )

                    except Exception as ex:
                        # Affiche une erreur si l'image ne peut pas être analysée
                        print('Erreur pendant l’analyse :', repr(ex))
                        status_badge.content = '<div class="status-pill status-warn">Erreur</div>'
                        result_box.content = f'''
                            <div class="result-box result-warn">
                                <div class="result-title">Erreur</div>
                                <div class="result-brand">Analyse impossible</div>
                                <div class="result-text">{ex}</div>
                            </div>
                        '''
                        confidence_value.set_text('-')
                        confidence_bar.set_value(0)
                        confidence_hint.set_text('Une erreur est survenue pendant le traitement de l’image.')

                    finally:
                        # Cache le chargement après l'analyse
                        spinner.set_visibility(False)

                # Bouton d'upload de l'image
                upload = ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                    max_files=1,
                    label='Choisir une photo ou prendre une photo avec la caméra',
                ).props('accept="image/*" capture="environment"').classes('w-full')

                ui.label(
                    'Conseil : la voiture doit être bien visible, nette et pas trop petite dans l’image.'
                ).classes('small-muted')

            # Partie droite : résultat de l'analyse
            with ui.card().classes('glass-card w-full p-6 gap-5'):
                ui.label('Analyse').classes('section-title')

                status_badge = ui.html('<div class="status-pill status-warn">En attente d’une image</div>')

                result_box = ui.html('''
                    <div class="result-box result-warn">
                        <div class="result-title">Statut</div>
                        <div class="result-brand">Aucune analyse</div>
                        <div class="result-text">Après l’upload, la marque reconnue et la fiabilité apparaîtront ici.</div>
                    </div>
                ''')

                ui.label('Confiance').classes('small-muted')
                confidence_value = ui.label('-').classes('text-2xl font-extrabold text-slate-800')
                confidence_bar = ui.linear_progress(value=0).classes('w-full')
                confidence_hint = ui.label('Aucun résultat pour le moment.').classes('small-muted')

                with ui.grid(columns=2).classes('w-full gap-4 max-[700px]:grid-cols-1'):
                    top1_card = ui.html('<div class="metric-card"><div class="metric-label">Top 1</div><div class="metric-value">-</div><div class="metric-sub">-</div></div>')
                    top2_card = ui.html('<div class="metric-card"><div class="metric-label">Top 2</div><div class="metric-value">-</div><div class="metric-sub">-</div></div>')
                    top3_card = ui.html('<div class="metric-card"><div class="metric-label">Top 3</div><div class="metric-value">-</div><div class="metric-sub">-</div></div>')
                    gap_card = ui.html('<div class="metric-card"><div class="metric-label">Écart Top 1 - Top 2</div><div class="metric-value">-</div><div class="metric-sub">Un écart plus grand indique souvent une décision plus claire</div></div>')

        # Partie du bas : classement et informations utiles
        with ui.grid(columns=2).classes('w-full gap-6 max-[900px]:grid-cols-1'):

            with ui.card().classes('glass-card w-full p-6 gap-4'):
                ui.label('Classement Top 3').classes('section-title')
                ranking_1 = ui.html('<div class="top-card"><div class="flex items-center gap-3"><div class="rank-number">1</div><div><div class="brand-name">-</div><div class="brand-prob">-</div></div></div></div>')
                ranking_2 = ui.html('<div class="top-card"><div class="flex items-center gap-3"><div class="rank-number">2</div><div><div class="brand-name">-</div><div class="brand-prob">-</div></div></div></div>')
                ranking_3 = ui.html('<div class="top-card"><div class="flex items-center gap-3"><div class="rank-number">3</div><div><div class="brand-name">-</div><div class="brand-prob">-</div></div></div></div>')

            with ui.card().classes('glass-card w-full p-6 gap-4'):
                ui.label('Informations utiles').classes('section-title')
                info_1 = ui.html('<div class="info-box"><strong>Interprétation :</strong> La confiance est la probabilité préférée par le modèle pour la meilleure marque.</div>')
                info_2 = ui.html(f'<div class="info-box"><strong>Règle d’incertitude :</strong> Sous {UNCERTAINTY_THRESHOLD * 100:.0f}%, la prédiction est marquée comme incertaine.</div>')
                info_3 = ui.html('<div class="info-box"><strong>Qualité de l’image :</strong> Une bonne lumière, une vue claire de l’avant ou du côté et peu de fond aident la reconnaissance.</div>')
                info_4 = ui.html('<div class="info-box"><strong>Limites :</strong> Les voitures cachées, les images de nuit, le tuning ou les angles inhabituels peuvent provoquer des erreurs.</div>')


# Lance l'application web sur le port 8080
ui.run(host='0.0.0.0', port=8080, title='Car Brand Classifier')