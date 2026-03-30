import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import xBDDamageDataset
from model import OrbitalDamageDetector

# ==========================================
# 1. Custom Loss Function: BCE + Dice Loss
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        # This safely handles the raw logits coming from our ResNet34 model
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # Calculate standard BCE
        bce_loss = self.bce(inputs, targets)
        
        # Calculate Dice Loss (forces model to care about building edges)
        inputs = torch.sigmoid(inputs) # Convert logits to probabilities (0 to 1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))  
        
        return bce_loss + dice_loss

# ==========================================
# 2. Training Configuration
# ==========================================
def main():
    # Ignite the RTX 4060
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing High-Performance Training on: {device}")

    # Hyperparameters
    BATCH_SIZE = 2      # Lowered to 2 to protect VRAM from 6-channel OOM crashes
    LEARNING_RATE = 1e-4
    EPOCHS = 5          
    
    # Updated Paths mapping to our new pipeline
    IMAGES_DIR = "data/raw/images"
    LABELS_DIR = "data/raw/labels"
    WEIGHTS_DIR = "data/weights"
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ==========================================
    # 3. Load Data & Architecture
    # ==========================================
    print("[*] Loading xBD Dataset...")
    train_dataset = xBDDamageDataset(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"[*] Loaded {len(train_dataset)} image pairs.")

    print("[*] Loading Model to GPU...")
    model = OrbitalDamageDetector().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEDiceLoss()

    # ==========================================
    # 4. The Training Loop
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # tqdm progress bar wrapped around the dataloader
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in loop:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed | Average Combined Loss: {avg_loss:.4f}\n")

    # ==========================================
    # 5. Save the Real Weights
    # ==========================================
    save_path = os.path.join(WEIGHTS_DIR, "orbital_detector_v2.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[+] Training Complete! Real weights saved to: {save_path}")

if __name__ == "__main__":
    main()