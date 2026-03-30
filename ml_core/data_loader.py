import os
import torch
import rasterio
import numpy as np

def load_geotiff_to_tensor(filepath):
    """
    Reads a GeoTIFF file and converts it into a PyTorch tensor.
    Handles NaN values and normalizes the data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find the file: {filepath}. Did you move it to the data/raw/ folder?")
        
    print(f"Loading {os.path.basename(filepath)}...")
    
    with rasterio.open(filepath) as src:
        # Read the first band (our scripts only exported single-band difference calculations)
        # rasterio reads data as numpy arrays
        image_data = src.read(1)
        
        # Geodata often contains extreme outlier values or NaNs at the edges of the satellite swath
        # We replace NaNs (Not a Number) with 0.0 to prevent our neural network from crashing
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert the numpy array to a PyTorch float32 tensor
        tensor_data = torch.from_numpy(image_data).float()
        
        # Add a channel dimension so it matches the expected PyTorch format: [Channels, Height, Width]
        tensor_data = tensor_data.unsqueeze(0)
        
        return tensor_data

if __name__ == "__main__":
    # Define the paths to our newly downloaded data
    sar_path = "../data/raw/sar_damage_baseline.tif"
    ndvi_path = "../data/raw/ndvi_degradation_baseline.tif"
    
    try:
        # 1. Load the SAR (Structural Damage) Tensor
        sar_tensor = load_geotiff_to_tensor(sar_path)
        print(f"SAR Tensor Shape: {sar_tensor.shape}")
        print(f"SAR Min value: {sar_tensor.min().item():.4f}, Max value: {sar_tensor.max().item():.4f}")
        
        print("-" * 30)
        
        # 2. Load the NDVI (Ecological Degradation) Tensor
        ndvi_tensor = load_geotiff_to_tensor(ndvi_path)
        print(f"NDVI Tensor Shape: {ndvi_tensor.shape}")
        print(f"NDVI Min value: {ndvi_tensor.min().item():.4f}, Max value: {ndvi_tensor.max().item():.4f}")
        
        print("\nSuccess! Data is securely loaded into PyTorch and ready for the neural network.")
        
    except Exception as e:
        print(f"\nError encountered: {e}")
