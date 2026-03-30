import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .model import OrbitalDamageDetector

# Hardcoded path to the real weights we just trained
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../data/weights/orbital_detector_v2.pth")

class DamageInferenceEngine:
    def __init__(self):
        # 1. Detect Hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ML Core] Initializing Inference Engine on: {self.device}")

        # 2. Load Architecture
        self.model = OrbitalDamageDetector().to(self.device)
        
        # 3. Load Real Weights
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"[ML Core] CRITICAL ERROR: Trained weights not found at {WEIGHTS_PATH}")
            
        print("[ML Core] Loading trained weights (v2)...")
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
        self.model.eval() # Set to evaluation mode (disables gradients/dropout)
        print("[ML Core] AI Engine Armed and Ready.")

    def _preprocess_image(self, image_path):
        """Loads and normalizes an image for the neural network."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 256x256 to match our training dimensions
        img = cv2.resize(img, (256, 256))
        return img.astype(np.float32) / 255.0

    def generate_heatmap(self, pre_image_path, post_image_path, output_save_path):
        """
        Executes the AI forward pass and generates a noise-filtered Magma XAI Heatmap.
        """
        # 1. Load and Stack Images -> (256, 256, 6)
        pre_img = self._preprocess_image(pre_image_path)
        post_img = self._preprocess_image(post_image_path)
        stacked = np.concatenate([pre_img, post_img], axis=-1)

        # 2. Convert to PyTorch Tensor -> (1, 6, 256, 256)
        tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 3. Execute AI Inference
        with torch.no_grad():
            raw_logits = self.model(tensor)
            probabilities = torch.sigmoid(raw_logits)

        # 4. Extract the 2D array
        prob_mask = probabilities.squeeze().cpu().numpy()

        # --- ADVANCED SAR NOISE FILTERING ---
        # A. Hard Threshold: Ignore anything the AI isn't at least 65% confident about
        prob_mask[prob_mask < 0.65] = 0.0

        # B. Morphological Opening: Erase tiny SAR speckle noise (false positives)
        kernel = np.ones((5, 5), np.uint8) 
        clean_mask = cv2.morphologyEx(prob_mask, cv2.MORPH_OPEN, kernel)

        # C. Morphological Closing: Fill in the gaps inside actual detected structures
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        # -----------------------------------------

        # 5. Render Explainable AI (XAI) Heatmap
        plt.figure(figsize=(6, 6))
        # Use the clean_mask instead of the raw prob_mask
        plt.imshow(clean_mask, cmap='magma', vmin=0.0, vmax=1.0)
        plt.axis('off')
        
        # 6. Save the heatmap
        plt.savefig(output_save_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

        # Calculate a stricter damage metric based on the cleaned data
        damage_ratio = np.sum(clean_mask > 0.65) / (256 * 256)
        return {"damage_percentage": round(float(damage_ratio * 100), 2)}

# Instantiate a global engine so FastAPI only loads the heavy model into VRAM once
engine = DamageInferenceEngine()

def run_analysis(pre_path, post_path, output_path):
    """Wrapper function for the FastAPI backend to call."""
    return engine.generate_heatmap(pre_path, post_path, output_path)