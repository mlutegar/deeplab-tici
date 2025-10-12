import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================
# CONFIGURAÇÕES (AJUSTE PARA SUA GPU)
# ============================================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 2,  # 3050 Ti aguenta 2-4
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'image_size': 512,  # Não tente 1024 com 4GB VRAM
    'num_classes': 1,  # Segmentação binária (edifício sim/não)
}

# ============================================
# DATASET CUSTOMIZADO
# ============================================
class BuildingDataset(Dataset):
    """
    Dataset para imagens e máscaras.
    Estrutura esperada:
    data/
      images/
        img1.jpg
        img2.jpg
      masks/
        img1.png  (valores: 0=fundo, 255=edifício)
        img2.png
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Lista todos os arquivos de imagem
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                   list(self.images_dir.glob('*.png')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar máscara (assume mesmo nome da imagem)
        mask_path = self.masks_dir / img_path.name.replace('.jpg', '.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Normalizar máscara para [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        # Aplicar transformações
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)  # Adiciona dimensão de canal

# ============================================
# TRANSFORMAÇÕES (AUGMENTATIONS)
# ============================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(CONFIG['image_size'], CONFIG['image_size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                               rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['image_size'], CONFIG['image_size']),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# ============================================
# CRIAR MODELO DEEPLAB V3
# ============================================
def create_deeplab_model():
    """
    DeepLab v3+ com encoder ResNet50
    Você pode trocar o encoder por:
    - 'resnet34' (mais leve)
    - 'resnet101' (mais pesado)
    - 'efficientnet-b0' (balanceado)
    """
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',  # Pré-treinado
        in_channels=3,
        classes=CONFIG['num_classes'],
        activation=None,  # Sigmoid aplicado na loss
    )
    return model

# ============================================
# FUNÇÃO DE TREINO
# ============================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Treinando')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

# ============================================
# FUNÇÃO DE VALIDAÇÃO
# ============================================
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    for images, masks in tqdm(dataloader, desc='Validando'):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ============================================
# MAIN - TREINAMENTO
# ============================================
def main():
    print(f"🚀 Usando device: {CONFIG['device']}")
    
    # Criar datasets
    train_dataset = BuildingDataset(
        'data/train/images',
        'data/train/masks',
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = BuildingDataset(
        'data/val/images',
        'data/val/masks',
        transform=get_transforms(is_train=False)
    )
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Criar modelo
    model = create_deeplab_model().to(CONFIG['device'])
    print(f"✅ Modelo DeepLab v3+ criado!")
    
    # Loss e otimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                   lr=CONFIG['learning_rate'],
                                   weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs']
    )
    
    # Loop de treinamento
    best_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n📈 Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, criterion, 
                                 optimizer, CONFIG['device'])
        val_loss = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Salvar melhor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/deeplab_best.pth')
            print("💾 Modelo salvo!")
    
    print("\n✨ Treinamento concluído!")

if __name__ == '__main__':
    main()