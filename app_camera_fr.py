import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from nicegui import ui


#############################
# 1. Paramètres
#############################

MODEL_PATH = Path('cnn_full_train_100_percent.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = (256, 256)

# Sous 55 %, le modèle est considéré comme incertain
UNCERTAINTY_THRESHOLD = 0.55

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

IDX_TO_BRAND = {idx: brand for idx, brand in enumerate(TRAIN_CLASSES)}

MODEL = None


#############################
# 2. Modèle
#############################

class CarBrandClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

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
# 3. Prétraitement
#############################

infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def pretty_brand_name(name: str) -> str:
    return ' '.join(part.capitalize() for part in name.split())


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded}'


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Fichier du modèle introuvable : {MODEL_PATH}')

    model = CarBrandClassifier(num_classes=len(TRAIN_CLASSES)).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print('Missing keys:', missing)
    print('Unexpected keys:', unexpected)

    if missing or unexpected:
        print('ATTENTION : le modèle et le checkpoint ne correspondent pas parfaitement.')
    else:
        print('Modèle chargé avec succès.')

    model.eval()
    return model


@torch.inference_mode()
def predict_image(pil_image: Image.Image):
    if MODEL is None:
        raise RuntimeError('Le modèle n’est pas chargé.')

    image = pil_image.convert('RGB')
    x = infer_transform(image).unsqueeze(0).to(DEVICE)

    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=3)

    results = []

    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({
            'brand': IDX_TO_BRAND[idx],
            'confidence': prob,
        })

    print('Top-3 :', results)
    return results


async def lire_fichier_upload(e):
    """
    Lit le fichier uploadé.
    Cette fonction est compatible avec plusieurs versions de NiceGUI.
    """
    if hasattr(e, 'content'):
        return e.content.read()

    if hasattr(e, 'file'):
        result = e.file.read()

        if hasattr(result, '__await__'):
            return await result

        return result

    raise AttributeError("Impossible de lire le fichier uploadé : ni 'content' ni 'file' trouvés.")


#############################
# 4. Application
#############################

def main():
    global MODEL

    MODEL = load_model()

    ui.page_title('Car Brand Classifier')

    with ui.column().classes('w-full items-center'):
        ui.label('Car Brand Classifier').classes('text-3xl font-bold mt-6')
        ui.label('Téléverse une photo d’une voiture ou prends directement une photo avec ton téléphone.')
        ui.label(f'Modèle exécuté sur : {DEVICE}').classes('text-sm text-gray-600 mb-4')

        with ui.card().classes('w-full max-w-2xl p-6'):
            preview = ui.image('').classes('w-full max-h-96 object-contain rounded')
            preview.set_visibility(False)

            result_label = ui.label('Aucune image téléversée.').classes('text-lg mt-4')
            confidence_label = ui.label('').classes('text-base text-orange-600')
            top1_label = ui.label('').classes('text-2xl font-semibold text-primary')
            top2_label = ui.label('').classes('text-base')
            top3_label = ui.label('').classes('text-base')

            spinner = ui.spinner(size='lg')
            spinner.set_visibility(False)

            async def handle_upload(e):
                try:
                    spinner.set_visibility(True)

                    result_label.set_text('Image en cours de traitement...')
                    confidence_label.set_text('')
                    top1_label.set_text('')
                    top2_label.set_text('')
                    top3_label.set_text('')

                    file_bytes = await lire_fichier_upload(e)

                    pil_image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

                    preview.set_source(pil_to_data_url(pil_image))
                    preview.set_visibility(True)

                    results = predict_image(pil_image)

                    best = results[0]
                    best_conf = best['confidence']

                    result_label.set_text('Prédiction terminée.')

                    if best_conf < UNCERTAINTY_THRESHOLD:
                        top1_label.set_text(
                            f"Incertain : meilleure estimation : "
                            f"{pretty_brand_name(best['brand'])} ({best_conf * 100:.2f}%)"
                        )
                        confidence_label.set_text(
                            f"Sous {UNCERTAINTY_THRESHOLD * 100:.0f} %, la prédiction est considérée comme incertaine."
                        )
                    else:
                        top1_label.set_text(
                            f"Top 1 : {pretty_brand_name(best['brand'])} ({best_conf * 100:.2f}%)"
                        )
                        confidence_label.set_text('La prédiction est considérée comme suffisamment fiable.')

                    if len(results) > 1:
                        top2_label.set_text(
                            f"Top 2 : {pretty_brand_name(results[1]['brand'])} "
                            f"({results[1]['confidence'] * 100:.2f}%)"
                        )

                    if len(results) > 2:
                        top3_label.set_text(
                            f"Top 3 : {pretty_brand_name(results[2]['brand'])} "
                            f"({results[2]['confidence'] * 100:.2f}%)"
                        )

                except Exception as ex:
                    print('Erreur upload :', repr(ex))

                    result_label.set_text(f'Erreur : {ex}')
                    confidence_label.set_text('')
                    top1_label.set_text('')
                    top2_label.set_text('')
                    top3_label.set_text('')

                finally:
                    spinner.set_visibility(False)

            ui.upload(
                on_upload=handle_upload,
                auto_upload=True,
                max_files=1,
                label='Choisir une photo ou prendre une photo avec la caméra',
            ).props('accept="image/*" capture="environment"').classes('w-full mt-4')

            ui.label(
                'Sur téléphone, le navigateur peut proposer la caméra ou la galerie.'
            ).classes('text-sm text-gray-600 mt-4')

    ui.run(
        host='0.0.0.0',
        port=8080,
        title='Car Brand Classifier',
        reload=False
    )


if __name__ == '__main__':
    main()