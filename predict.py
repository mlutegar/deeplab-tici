import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# CONFIGURAÇÕES
# ============================================
CONFIG = {
    'model_path': 'models/deeplab_best.pth',  # Modelo treinado
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_size': 512,
    'threshold': 0.5,  # Quanto de certeza precisa (0-1)
}


# ============================================
# CARREGAR O MODELO TREINADO
# ============================================
def load_trained_model(model_path, device):
    """Carrega o modelo que você treinou"""
    print(f"📥 Carregando modelo de {model_path}...")

    # Criar a mesma arquitetura
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,  # Não usa pré-treino, usa seus pesos
        in_channels=3,
        classes=1,
        activation=None,
    )

    # Carregar os pesos treinados
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Modo de avaliação (desliga dropout, etc)

    print("✅ Modelo carregado com sucesso!")
    return model


# ============================================
# PREPROCESSAR IMAGEM
# ============================================
def preprocess_image(image_path, image_size=512):
    """Prepara a imagem para o modelo"""

    # Ler imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não consegui ler a imagem: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    # Mesmas transformações do treino (sem augmentation)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Aplicar transformação
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # Adiciona batch dimension

    return image_tensor, image, original_size


# ============================================
# FAZER PREDIÇÃO
# ============================================
@torch.no_grad()  # Não calcular gradientes (mais rápido)
def predict(model, image_tensor, device, threshold=0.5):
    """
    Faz a predição!

    Returns:
        pred_mask: Máscara binária (0 ou 255)
        confidence: Mapa de confiança (0 a 1)
    """

    image_tensor = image_tensor.to(device)

    # MÁGICA ACONTECE AQUI! 🎩✨
    output = model(image_tensor)

    # Converter logits para probabilidades (0 a 1)
    confidence = torch.sigmoid(output).cpu().numpy()[0, 0]

    # Aplicar threshold
    pred_mask = (confidence > threshold).astype(np.uint8) * 255

    return pred_mask, confidence


# ============================================
# VISUALIZAR RESULTADO
# ============================================
def visualize_prediction(original_image, pred_mask, confidence, save_path=None):
    """Mostra o resultado de forma bonita"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # 1. Imagem original
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('🖼️ Imagem Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Máscara predita
    axes[0, 1].imshow(pred_mask, cmap='gray')
    axes[0, 1].set_title('🏗️ Edifícios Detectados', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Mapa de confiança (heatmap)
    im = axes[1, 0].imshow(confidence, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('🔥 Mapa de Confiança', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 4. Overlay (edifícios em vermelho sobre a imagem)
    overlay = original_image.copy()

    # Redimensionar máscara para tamanho original
    pred_mask_resized = cv2.resize(pred_mask,
                                   (original_image.shape[1], original_image.shape[0]))

    # Colorir edifícios de vermelho semi-transparente
    red_mask = np.zeros_like(original_image)
    red_mask[pred_mask_resized > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(original_image, 0.7, red_mask, 0.3, 0)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('🎯 Overlay (Vermelho = Edifícios)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Resultado salvo em: {save_path}")

    plt.show()


# ============================================
# CALCULAR ESTATÍSTICAS
# ============================================
def calculate_stats(pred_mask, confidence):
    """Calcula estatísticas da predição"""

    total_pixels = pred_mask.size
    building_pixels = np.sum(pred_mask > 0)
    building_percentage = (building_pixels / total_pixels) * 100

    avg_confidence = np.mean(confidence[pred_mask > 0]) if building_pixels > 0 else 0

    print("\n" + "=" * 50)
    print("📊 ESTATÍSTICAS DA PREDIÇÃO")
    print("=" * 50)
    print(f"🏗️  Área com edifícios: {building_percentage:.2f}%")
    print(f"📏 Pixels de edifícios: {building_pixels:,} de {total_pixels:,}")
    print(f"🎯 Confiança média: {avg_confidence:.2%}")
    print("=" * 50 + "\n")


# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def main(image_path, save_result=True):
    """
    Pipeline completo de predição

    Args:
        image_path: Caminho para a imagem
        save_result: Se True, salva o resultado
    """

    print("🚀 Iniciando predição...")
    print(f"📸 Imagem: {image_path}\n")

    # 1. Carregar modelo
    model = load_trained_model(CONFIG['model_path'], CONFIG['device'])

    # 2. Preprocessar imagem
    image_tensor, original_image, original_size = preprocess_image(
        image_path,
        CONFIG['image_size']
    )

    # 3. Fazer predição
    print("🔮 Fazendo predição...")
    pred_mask, confidence = predict(
        model,
        image_tensor,
        CONFIG['device'],
        CONFIG['threshold']
    )

    # 4. Calcular estatísticas
    calculate_stats(pred_mask, confidence)

    # 5. Visualizar
    output_path = f"resultado_{Path(image_path).stem}.png" if save_result else None
    visualize_prediction(original_image, pred_mask, confidence, output_path)

    print("✨ Predição concluída!")

    return pred_mask, confidence


# ============================================
# PROCESSAR PASTA INTEIRA
# ============================================
def predict_folder(folder_path, output_folder='resultados'):
    """Processa todas as imagens de uma pasta"""

    Path(output_folder).mkdir(exist_ok=True)

    # Carregar modelo uma vez
    model = load_trained_model(CONFIG['model_path'], CONFIG['device'])

    # Listar todas as imagens
    image_files = list(Path(folder_path).glob('*.jpg')) + \
                  list(Path(folder_path).glob('*.png'))

    print(f"📂 Encontradas {len(image_files)} imagens em {folder_path}\n")

    for img_path in image_files:
        print(f"Processando: {img_path.name}...")

        # Preprocessar
        image_tensor, original_image, _ = preprocess_image(str(img_path))

        # Predizer
        pred_mask, confidence = predict(model, image_tensor, CONFIG['device'])

        # Salvar resultado
        output_path = Path(output_folder) / f"pred_{img_path.stem}.png"
        cv2.imwrite(str(output_path), pred_mask)

        print(f"✅ Salvo em: {output_path}\n")

    print(f"🎉 Processamento completo! Resultados em '{output_folder}/'")


# ============================================
# EXECUTAR
# ============================================
if __name__ == '__main__':
    import sys

    # USO 1: Predizer uma imagem
    # python predict.py minha_foto.jpg

    # USO 2: Predizer pasta inteira
    # python predict.py --folder minhas_fotos/

    if len(sys.argv) < 2:
        print("❌ Uso:")
        print("  python predict.py <imagem.jpg>")
        print("  python predict.py --folder <pasta/>")
        sys.exit(1)

    if sys.argv[1] == '--folder':
        predict_folder(sys.argv[2])
    else:
        main(sys.argv[1])