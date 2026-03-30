import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class OrbitalDamageDetector(nn.Module):
    def __init__(self):
        super(OrbitalDamageDetector, self).__init__()
        
        # Initialize a U-Net with a pre-trained ResNet34 backbone
        self.model = smp.Unet(
            encoder_name="resnet34",        # Uses ResNet34 for feature extraction
            encoder_weights="imagenet",     # Loads weights pre-trained on ImageNet
            in_channels=6,                  # 6 channels: Pre-disaster RGB (3) + Post-disaster RGB (3)
            classes=1,                      # 1 channel: Binary damage mask
            activation=None                 # Outputs raw logits; activation is handled by the Loss Function
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Tensor of shape (Batch, 6, Height, Width)
        Returns:
            torch.Tensor: Tensor of shape (Batch, 1, Height, Width) containing raw damage logits
        """
        return self.model(x)

# Local execution block to verify architecture dimensions
if __name__ == "__main__":
    print("Initializing OrbitalDamageDetector with ResNet34 backbone...")
    model = OrbitalDamageDetector()
    
    # Simulate a batch of 2 satellite image pairs (6 channels, 256x256 resolution)
    dummy_input = torch.randn(2, 6, 256, 256)
    
    print(f"Feeding simulated tensor of shape: {dummy_input.shape}")
    output = model(dummy_input)
    
    print(f"Output tensor shape: {output.shape}")
    print("Architecture verified. Ready for xBD integration.")