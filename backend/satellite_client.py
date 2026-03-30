import ee
import os
import requests

def initialize_engine():
    """Connects to Google Earth Engine using your local credentials."""
    try:
        # Authenticate using your specific project ID
        ee.Initialize(project='orbital-intelligence-thesis') 
        print("Orbital Uplink Established: Earth Engine API Active.")
    except Exception as e:
        print("Earth Engine not authenticated. Please run 'earthengine authenticate' in the terminal.")
        raise e

def fetch_orbital_data(lat, lon, claim_id, output_dir):
    """
    Commands Sentinel-2 to capture Before/After True Color imagery for the AI,
    and generates an NDVI Environmental map for the React UI.
    """
    print(f"[*] Uplinking to Sentinel-2 for target {claim_id}...")
    
    # 1. Define a 5km x 5km Region of Interest (ROI) around the clicked coordinate
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(2500).bounds() 

    # --- TRUE COLOR OPTICAL ALIGNMENT (THE AI FIX) ---
    rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}

    # 2. Fetch the "BEFORE" Image (Historical Baseline: 2023)
    pre_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    
    pre_image = pre_collection.median().visualize(**rgb_vis).clip(roi)

    # 3. Fetch the "AFTER" Image (Recent Status: 2024)
    post_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2024-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    
    post_image = post_collection.median().visualize(**rgb_vis).clip(roi)

    # 4. Generate NDVI Environmental Map for the React UI (using the 2024 data)
    raw_post_image = post_collection.median().clip(roi)
    ndvi_image = raw_post_image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # 5. Generate direct download URLs (PNGs for React and AI)
    pre_url = pre_image.getThumbURL({'dimensions': '512x512', 'format': 'png'})
    post_url = post_image.getThumbURL({'dimensions': '512x512', 'format': 'png'})
    
    ndvi_url = ndvi_image.getThumbURL({
        'dimensions': '512x512',
        'format': 'png',
        'min': -0.2, 'max': 0.8,
        'palette': ['ff0000', 'ffaa00', 'ffff00', 'aaff00', '00ff00'] 
    })

    # 6. Define file paths
    pre_png_path = os.path.join(output_dir, f"pre_optical_{claim_id}.png")
    post_png_path = os.path.join(output_dir, f"post_optical_{claim_id}.png")
    ndvi_png_path = os.path.join(output_dir, f"real_ndvi_{claim_id}.png")

    # 7. Download and save all 3 files
    print("[*] Downloading telemetry to local pipeline...")
    with open(pre_png_path, 'wb') as f:
        f.write(requests.get(pre_url).content)
    with open(post_png_path, 'wb') as f:
        f.write(requests.get(post_url).content)
    with open(ndvi_png_path, 'wb') as f:
        f.write(requests.get(ndvi_url).content)

    print("[+] Download complete.")
    
    # We return a 4th `None` variable to safely match the unpack logic in main.py
    return pre_png_path, post_png_path, ndvi_png_path, None

# Quick test execution
if __name__ == "__main__":
    initialize_engine()
    print("Testing connection to Sentinel Network...")
    fetch_orbital_data(27.1832, 56.2666, 999, "../data/processed")
    print("Test execution finished. Check your processed folder!")