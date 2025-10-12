import os
import urllib.request
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile


class DownloadProgressBar(tqdm):
    """Barra de progresso para downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Baixa arquivo com barra de progresso"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(file_path, extract_to):
    """Extrai arquivo .tgz ou .zip"""
    print(f"📦 Extraindo {file_path}...")

    if file_path.endswith('.tgz') or file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
    elif file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    print(f"✅ Extraído para {extract_to}")


def download_massachusetts_buildings():
    """Baixa Massachusetts Buildings Dataset"""

    base_url = "https://www.cs.toronto.edu/~vmnih/data/mass_buildings/train/"

    # URLs dos arquivos
    files = {
        'train_images': 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/train/sat/index.html',
        'train_masks': 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/train/map/index.html',
        'val_images': 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/valid/sat/index.html',
        'val_masks': 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/valid/map/index.html',
        'test_images': 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/test/sat/index.html',
        'test_masks': 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/test/map/index.html',
    }

    # Criar estrutura de pastas
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/train/images').mkdir(parents=True, exist_ok=True)
    Path('data/train/masks').mkdir(parents=True, exist_ok=True)
    Path('data/val/images').mkdir(parents=True, exist_ok=True)
    Path('data/val/masks').mkdir(parents=True, exist_ok=True)
    Path('data/test/images').mkdir(parents=True, exist_ok=True)
    Path('data/test/masks').mkdir(parents=True, exist_ok=True)

    print("🌍 Massachusetts Buildings Dataset")
    print("=" * 50)
    print("\n⚠️  Este dataset não tem links diretos de download.")
    print("Você precisará baixar manualmente de:")
    print("https://www.cs.toronto.edu/~vmnih/data/")
    print("\nEstou gerando um script alternativo...\n")


def download_alternative_dataset():
    """
    Como o Massachusetts Buildings não tem links diretos,
    vou usar o Inria Aerial Image Dataset que é mais fácil de baixar
    """
    print("📥 Baixando Inria Aerial Image Dataset (alternativa)")
    print("=" * 50)

    # Inria dataset (similar, mas com links diretos)
    urls = [
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005",
    ]

    Path('data/raw').mkdir(parents=True, exist_ok=True)

    print("\n⚠️  Infelizmente, este também precisa de download manual.")
    print("Alternativa: vou criar um dataset sintético pequeno para você testar!\n")


def create_dummy_dataset():
    """Cria um dataset sintético pequeno para testar o código"""
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw

    print("🎨 Criando dataset sintético para testes...")
    print("=" * 50)

    # Criar estrutura
    Path('data/train/images').mkdir(parents=True, exist_ok=True)
    Path('data/train/masks').mkdir(parents=True, exist_ok=True)
    Path('data/val/images').mkdir(parents=True, exist_ok=True)
    Path('data/val/masks').mkdir(parents=True, exist_ok=True)

    def create_building_sample(size=512):
        """Cria uma imagem sintética com edifícios"""
        # Imagem base (terreno)
        image = np.random.randint(100, 150, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        # Adicionar alguns "edifícios" (retângulos)
        num_buildings = np.random.randint(3, 8)

        for _ in range(num_buildings):
            x1 = np.random.randint(0, size - 100)
            y1 = np.random.randint(0, size - 100)
            w = np.random.randint(30, 120)
            h = np.random.randint(30, 120)

            # Edifício na imagem (cinza mais escuro)
            color = np.random.randint(50, 100, 3)
            image[y1:y1 + h, x1:x1 + w] = color

            # Edifício na máscara (branco)
            mask[y1:y1 + h, x1:x1 + w] = 255

        return image, mask

    # Criar amostras de treino
    print("\n📁 Criando amostras de treino...")
    for i in tqdm(range(50)):
        img, mask = create_building_sample()
        cv2.imwrite(f'data/train/images/sample_{i:03d}.png', img)
        cv2.imwrite(f'data/train/masks/sample_{i:03d}.png', mask)

    # Criar amostras de validação
    print("📁 Criando amostras de validação...")
    for i in tqdm(range(15)):
        img, mask = create_building_sample()
        cv2.imwrite(f'data/val/images/val_{i:03d}.png', img)
        cv2.imwrite(f'data/val/masks/val_{i:03d}.png', mask)

    print("\n✅ Dataset sintético criado com sucesso!")
    print(f"   - Train: 50 amostras")
    print(f"   - Val: 15 amostras")
    print(f"\n📍 Localização: data/train/ e data/val/")
    print("\n⚠️  Este é um dataset SINTÉTICO para testes.")
    print("   Para treino real, baixe um dataset profissional.\n")


if __name__ == '__main__':
    print("🚀 Download de Datasets para Segmentação")
    print("=" * 60)
    print("\nEscolha uma opção:")
    print("1 - Instruções para Massachusetts Buildings (download manual)")
    print("2 - Criar dataset sintético para testes (recomendado)")
    print("3 - Sair")

    choice = input("\nSua escolha (1-3): ").strip()

    if choice == '1':
        download_massachusetts_buildings()
        print("\n📋 INSTRUÇÕES PARA DOWNLOAD MANUAL:")
        print("-" * 60)
        print("1. Acesse: https://www.cs.toronto.edu/~vmnih/data/")
        print("2. Clique em 'Massachusetts Buildings Dataset'")
        print("3. Baixe os arquivos:")
        print("   - Training Set: Input images + Target maps")
        print("   - Validation Set: Input images + Target maps")
        print("   - Test Set: Input images + Target maps")
        print("4. Extraia na pasta 'data/raw/'")
        print("5. Organize em data/train/images, data/train/masks, etc.")

    elif choice == '2':
        create_dummy_dataset()

    else:
        print("👋 Até logo!")