import ee
import time

def extract_sar_damage_baseline():
    # 1. Initialize with your Project ID
    PROJECT_ID = 'orbital-intelligence-thesis'
    try:
        ee.Initialize(project=PROJECT_ID)
        print("GEE Initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 2. Define the Region of Interest (ROI)
    # Using a test bounding box near a major logistical port (Bandar Abbas / Strait of Hormuz)
    # Format: [min_longitude, min_latitude, max_longitude, max_latitude]
    roi = ee.Geometry.Rectangle([56.10, 27.05, 56.40, 27.25])
    print("Region of Interest (ROI) defined.")

    # 3. Define the Temporal Baselines
    # Pre-strike (Late 2025) and Post-strike (Early 2026 up to current)
    pre_start, pre_end = '2025-08-01', '2025-12-31'
    post_start, post_end = '2026-01-01', '2026-03-24'

    # 4. Load the Sentinel-1 SAR Collection
    # We filter for 'IW' (Interferometric Wide) and 'VV' polarization, which is best for urban structures.
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterBounds(roi)
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW'))
                     .select('VV'))

    # 5. Create Pre and Post Composites
    # We use median() to remove temporary noise (like passing ships or temporary interference)
    pre_mosaic = s1_collection.filterDate(pre_start, pre_end).median().clip(roi)
    post_mosaic = s1_collection.filterDate(post_start, post_end).median().clip(roi)

    # 6. Calculate the Structural Damage (Difference)
    # A negative value means backscatter dropped (e.g., a standing building became flat rubble)
    damage_layer = post_mosaic.subtract(pre_mosaic).rename('sar_damage')

    # 7. Export the Data to Google Drive
    # GEE prevents direct downloads of large datasets. We export to your Google Drive as a GeoTIFF.
    # Our PyTorch model will later ingest these GeoTIFFs.
    export_task = ee.batch.Export.image.toDrive(
        image=damage_layer,
        description='SAR_Damage_Baseline_Test',
        folder='Orbital_Intelligence_Data', # It will create this folder in your Drive
        fileNamePrefix='sar_damage_baseline',
        region=roi.getInfo()['coordinates'],
        scale=10, # Sentinel-1 resolution is 10 meters per pixel
        maxPixels=1e10,
        fileFormat='GeoTIFF'
    )

    print("Starting export task to Google Drive...")
    export_task.start()

    # 8. Monitor the Task
    print("Task submitted. Waiting for Earth Engine servers to process (this may take a few minutes)...")
    while export_task.active():
        print(f"Task state: {export_task.status()['state']}...")
        time.sleep(15) # Check every 15 seconds

    print(f"Task finished with state: {export_task.status()['state']}")
    if export_task.status()['state'] == 'COMPLETED':
        print("Success! The GeoTIFF is now in your Google Drive under the 'Orbital_Intelligence_Data' folder.")
    else:
        print(f"Error details: {export_task.status().get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    extract_sar_damage_baseline()
