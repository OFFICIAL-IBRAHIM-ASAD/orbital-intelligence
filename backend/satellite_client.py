import ee
import os
import requests

# Iran-Israel war began in early March 2026.
# Pre-war baseline: Jan 2025 – Feb 2026 (stable peacetime reference)
# Post-war period:  Mar 2026 – present
PRE_WAR_START  = '2025-01-01'
PRE_WAR_END    = '2026-02-28'
POST_WAR_START = '2026-03-01'
POST_WAR_END   = '2026-03-31'

def initialize_engine():
    """Connects to Google Earth Engine using your local credentials."""
    try:
        ee.Initialize(project='orbital-intelligence-thesis')
        print("Orbital Uplink Established: Earth Engine API Active.")
    except Exception as e:
        print("Earth Engine not authenticated. Please run 'earthengine authenticate' in the terminal.")
        raise e

def fetch_orbital_data(lat, lon, claim_id, output_dir):
    """
    Commands Sentinel-2 to capture Before/After True Color imagery for the AI
    and generates a pre/post NDVI delta map for the React UI.

    Returns:
        tuple: (pre_png_path, post_png_path, ndvi_delta_png_path, ndvi_pre_png_path)
    """
    print(f"[*] Uplinking to Sentinel-2 for target {claim_id}...")

    # 1. Define a 5 km × 5 km Region of Interest (ROI) around the clicked coordinate
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(2500).bounds()

    rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}

    # 2. Fetch the PRE-WAR image (peacetime baseline: Jan 2025 – Feb 2026)
    pre_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(PRE_WAR_START, PRE_WAR_END)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    )
    pre_image = pre_collection.median().visualize(**rgb_vis).clip(roi)

    # 3. Fetch the POST-WAR image (March 2026 onwards)
    post_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(POST_WAR_START, POST_WAR_END)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    )
    post_image = post_collection.median().visualize(**rgb_vis).clip(roi)

    # 4. Compute NDVI for both periods and derive a degradation delta
    ndvi_vis_params = {
        'dimensions': '512x512',
        'format': 'png',
        'min': -0.2, 'max': 0.8,
        'palette': ['ff0000', 'ffaa00', 'ffff00', 'aaff00', '00ff00'],
    }

    raw_pre  = pre_collection.median().clip(roi)
    raw_post = post_collection.median().clip(roi)

    pre_ndvi  = raw_pre.normalizedDifference(['B8', 'B4']).rename('NDVI')
    post_ndvi = raw_post.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Negative delta = vegetation loss; red palette makes losses vivid
    ndvi_delta = post_ndvi.subtract(pre_ndvi).rename('ndvi_delta')
    delta_vis_params = {
        'dimensions': '512x512',
        'format': 'png',
        'min': -0.5, 'max': 0.5,
        'palette': ['ff0000', 'ff6600', 'ffff00', '00ff00', '006600'],
    }

    # 5. Build download URLs
    pre_url        = pre_image.getThumbURL({'dimensions': '512x512', 'format': 'png'})
    post_url       = post_image.getThumbURL({'dimensions': '512x512', 'format': 'png'})
    ndvi_pre_url   = pre_ndvi.getThumbURL(ndvi_vis_params)
    ndvi_delta_url = ndvi_delta.getThumbURL(delta_vis_params)

    # 6. Define file paths
    pre_png_path        = os.path.join(output_dir, f"pre_optical_{claim_id}.png")
    post_png_path       = os.path.join(output_dir, f"post_optical_{claim_id}.png")
    ndvi_pre_png_path   = os.path.join(output_dir, f"ndvi_pre_{claim_id}.png")
    ndvi_delta_png_path = os.path.join(output_dir, f"real_ndvi_{claim_id}.png")

    # 7. Download and save all four images
    # Guard: ensure every destination path stays inside output_dir (path traversal protection)
    resolved_output_dir = os.path.realpath(output_dir)
    print("[*] Downloading telemetry to local pipeline...")
    for url, path in [
        (pre_url,        pre_png_path),
        (post_url,       post_png_path),
        (ndvi_pre_url,   ndvi_pre_png_path),
        (ndvi_delta_url, ndvi_delta_png_path),
    ]:
        resolved_path = os.path.realpath(path)
        if not resolved_path.startswith(resolved_output_dir + os.sep):
            raise ValueError(f"Unsafe output path rejected: {path}")
        with open(resolved_path, 'wb') as f:
            f.write(requests.get(url).content)

    print("[+] Download complete.")
    return pre_png_path, post_png_path, ndvi_delta_png_path, ndvi_pre_png_path

# Quick test execution
if __name__ == "__main__":
    initialize_engine()
    print("Testing connection to Sentinel Network...")
    fetch_orbital_data(27.1832, 56.2666, 999, "../data/processed")
    print("Test execution finished. Check your processed folder!")