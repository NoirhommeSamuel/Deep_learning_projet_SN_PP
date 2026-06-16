import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from nicegui import ui, events

#############################
# 1. Einstellungen
#############################

MODEL_PATH = Path('cnn_full_train_100_percent.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (256, 256)
UNCERTAINTY_THRESHOLD = 0.45

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

#############################
# 2. Modell
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
# 3. Preprocessing
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
        raise FileNotFoundError(f'Modell-Datei nicht gefunden: {MODEL_PATH}')

    model = CarBrandClassifier(num_classes=len(TRAIN_CLASSES)).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print('Missing keys:', missing)
    print('Unexpected keys:', unexpected)

    if missing or unexpected:
        print('WARNUNG: Modell und Checkpoint passen nicht 100% zusammen.')
    else:
        print('Modell erfolgreich geladen.')

    model.eval()
    return model

MODEL = load_model()

@torch.inference_mode()
def predict_image(pil_image: Image.Image):
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

    return results

#############################
# 4. Styling
#############################

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
# 5. UI
#############################

ui.page_title('Car Brand Classifier')

with ui.column().classes('w-full items-center px-4 py-8'):
    with ui.column().classes('main-shell gap-6'):

        with ui.card().classes('glass-card w-full p-8 gap-4'):
            ui.label('Car Brand Classifier').classes('hero-title')
            ui.label(
                'Lade ein Fahrzeugfoto hoch oder nimm mit dem Smartphone direkt ein Bild auf. '
                'Die Anwendung zeigt die wahrscheinlichste Marke, die Top-3-Ergebnisse, die Confidence '
                'und hilfreiche Hinweise zur Einordnung der Vorhersage.'
            ).classes('hero-subtitle')

            with ui.row().classes('w-full gap-3 items-center'):
                ui.label(f'Modell läuft auf: {DEVICE}').classes('small-muted')
                ui.label(f'Anzahl Marken: {len(TRAIN_CLASSES)}').classes('small-muted')
                ui.label(f'Unsicher unter: {UNCERTAINTY_THRESHOLD * 100:.0f}%').classes('small-muted')

        with ui.grid(columns=2).classes('w-full gap-6 max-[900px]:grid-cols-1'):

            with ui.card().classes('glass-card w-full p-6 gap-5'):
                ui.label('Bild hochladen').classes('section-title')

                with ui.element('div').classes('preview-frame w-full flex items-center justify-center overflow-hidden'):
                    preview = ui.image('').classes('w-full max-h-[420px] object-contain rounded-xl')
                    preview.set_visibility(False)
                    preview_hint = ui.label('Noch kein Bild ausgewählt').classes('text-slate-400 text-base')

                spinner = ui.spinner(size='lg')
                spinner.set_visibility(False)

                ui.label(
                    'Du kannst ein vorhandenes Bild wählen oder auf dem Handy direkt mit der Kamera fotografieren.'
                ).classes('upload-note')

                async def handle_upload(e: events.UploadEventArguments):
                    try:
                        spinner.set_visibility(True)

                        file_bytes = e.content.read()
                        pil_image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

                        preview.set_source(pil_to_data_url(pil_image))
                        preview.set_visibility(True)
                        preview_hint.set_visibility(False)

                        results = predict_image(pil_image)
                        best = results[0]
                        best_conf = best['confidence']
                        gap_top12 = results[0]['confidence'] - results[1]['confidence']
                        is_safe = best_conf >= UNCERTAINTY_THRESHOLD

                        confidence_value.set_text(f'{best_conf * 100:.2f}%')
                        confidence_bar.set_value(best_conf)

                        if is_safe:
                            status_badge.content = '<div class="status-pill status-safe">Vorhersage eher sicher</div>'
                            confidence_hint.set_text(
                                f'Die Vorhersage liegt über dem Schwellenwert von {UNCERTAINTY_THRESHOLD * 100:.0f}%.'
                            )
                            result_box.content = f'''
                                <div class="result-box result-safe">
                                    <div class="result-title">Erkannte Marke</div>
                                    <div class="result-brand">{pretty_brand_name(best["brand"])}</div>
                                    <div class="result-text">
                                        Das Modell bewertet diese Marke aktuell als wahrscheinlichste Vorhersage.
                                    </div>
                                </div>
                            '''
                        else:
                            status_badge.content = '<div class="status-pill status-warn">Vorhersage unsicher</div>'
                            confidence_hint.set_text(
                                f'Die Vorhersage liegt unter dem Schwellenwert von {UNCERTAINTY_THRESHOLD * 100:.0f}%.'
                            )
                            result_box.content = f'''
                                <div class="result-box result-warn">
                                    <div class="result-title">Beste Vermutung</div>
                                    <div class="result-brand">{pretty_brand_name(best["brand"])}</div>
                                    <div class="result-text">
                                        Das Modell ist sich nicht sicher. Die Marke wird nur als beste aktuelle Vermutung angezeigt.
                                    </div>
                                </div>
                            '''

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
                                <div class="metric-label">Abstand Top 1 zu Top 2</div>
                                <div class="metric-value">{gap_top12 * 100:.2f} Prozentpunkte</div>
                                <div class="metric-sub">Mehr Abstand bedeutet meist klarere Entscheidung</div>
                            </div>
                        '''

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

                        info_1.content = (
                            '<div class="info-box"><strong>Interpretation:</strong> '
                            f'Das Modell hält <strong>{pretty_brand_name(best["brand"])}</strong> im Moment für die wahrscheinlichste Marke.</div>'
                        )
                        info_2.content = (
                            '<div class="info-box"><strong>Unsicherheitsregel:</strong> '
                            f'Unter {UNCERTAINTY_THRESHOLD * 100:.0f}% wird das Ergebnis als unsicher markiert. '
                            f'Aktueller Wert: {best_conf * 100:.2f}%.</div>'
                        )
                        info_3.content = (
                            '<div class="info-box"><strong>Vergleich:</strong> '
                            f'Der Abstand zwischen Top 1 und Top 2 beträgt {gap_top12 * 100:.2f} Prozentpunkte.</div>'
                        )
                        info_4.content = (
                            '<div class="info-box"><strong>Praktischer Tipp:</strong> '
                            'Ein gut beleuchtetes Bild mit deutlich sichtbarer Front oder Seitenansicht verbessert meist die Vorhersage.</div>'
                        )

                    except Exception as ex:
                        print('Fehler bei der Analyse:', repr(ex))
                        status_badge.content = '<div class="status-pill status-warn">Fehler</div>'
                        result_box.content = f'''
                            <div class="result-box result-warn">
                                <div class="result-title">Fehler</div>
                                <div class="result-brand">Analyse nicht möglich</div>
                                <div class="result-text">{ex}</div>
                            </div>
                        '''
                        confidence_value.set_text('-')
                        confidence_bar.set_value(0)
                        confidence_hint.set_text('Es ist ein Fehler beim Verarbeiten des Bildes aufgetreten.')

                    finally:
                        spinner.set_visibility(False)

                upload = ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                    max_files=1,
                    label='Foto auswählen oder mit Kamera aufnehmen',
                ).props('accept="image/*" capture="environment"').classes('w-full')

                ui.label(
                    'Tipp: Das Auto sollte möglichst gut sichtbar, scharf und nicht zu klein im Bild sein.'
                ).classes('small-muted')

            with ui.card().classes('glass-card w-full p-6 gap-5'):
                ui.label('Analyse').classes('section-title')

                status_badge = ui.html('<div class="status-pill status-warn">Warte auf Bild</div>')

                result_box = ui.html('''
                    <div class="result-box result-warn">
                        <div class="result-title">Status</div>
                        <div class="result-brand">Noch keine Analyse</div>
                        <div class="result-text">Nach dem Upload erscheinen hier die erkannte Marke und die Bewertung der Sicherheit.</div>
                    </div>
                ''')

                ui.label('Confidence').classes('small-muted')
                confidence_value = ui.label('-').classes('text-2xl font-extrabold text-slate-800')
                confidence_bar = ui.linear_progress(value=0).classes('w-full')
                confidence_hint = ui.label('Noch kein Ergebnis vorhanden.').classes('small-muted')

                with ui.grid(columns=2).classes('w-full gap-4 max-[700px]:grid-cols-1'):
                    top1_card = ui.html('<div class="metric-card"><div class="metric-label">Top 1</div><div class="metric-value">-</div><div class="metric-sub">-</div></div>')
                    top2_card = ui.html('<div class="metric-card"><div class="metric-label">Top 2</div><div class="metric-value">-</div><div class="metric-sub">-</div></div>')
                    top3_card = ui.html('<div class="metric-card"><div class="metric-label">Top 3</div><div class="metric-value">-</div><div class="metric-sub">-</div></div>')
                    gap_card = ui.html('<div class="metric-card"><div class="metric-label">Abstand Top 1 zu Top 2</div><div class="metric-value">-</div><div class="metric-sub">Mehr Abstand bedeutet meist klarere Entscheidung</div></div>')

        with ui.grid(columns=2).classes('w-full gap-6 max-[900px]:grid-cols-1'):

            with ui.card().classes('glass-card w-full p-6 gap-4'):
                ui.label('Top-3 Ranking').classes('section-title')
                ranking_1 = ui.html('<div class="top-card"><div class="flex items-center gap-3"><div class="rank-number">1</div><div><div class="brand-name">-</div><div class="brand-prob">-</div></div></div></div>')
                ranking_2 = ui.html('<div class="top-card"><div class="flex items-center gap-3"><div class="rank-number">2</div><div><div class="brand-name">-</div><div class="brand-prob">-</div></div></div></div>')
                ranking_3 = ui.html('<div class="top-card"><div class="flex items-center gap-3"><div class="rank-number">3</div><div><div class="brand-name">-</div><div class="brand-prob">-</div></div></div></div>')

            with ui.card().classes('glass-card w-full p-6 gap-4'):
                ui.label('Nützliche Informationen').classes('section-title')
                info_1 = ui.html('<div class="info-box"><strong>Interpretation:</strong> Die Confidence ist die vom Modell bevorzugte Wahrscheinlichkeit für die beste Marke.</div>')
                info_2 = ui.html(f'<div class="info-box"><strong>Unsicherheitsregel:</strong> Unter {UNCERTAINTY_THRESHOLD * 100:.0f}% wird die Vorhersage als unsicher markiert.</div>')
                info_3 = ui.html('<div class="info-box"><strong>Bildqualität:</strong> Gute Beleuchtung, klare Sicht auf Front oder Seite und wenig Hintergrund helfen der Erkennung.</div>')
                info_4 = ui.html('<div class="info-box"><strong>Grenzen:</strong> Verdeckte Fahrzeuge, Nachtbilder, Tuning oder sehr ungewöhnliche Perspektiven können zu Fehlern führen.</div>')

ui.run(host='0.0.0.0', port=8080, title='Car Brand Classifier')