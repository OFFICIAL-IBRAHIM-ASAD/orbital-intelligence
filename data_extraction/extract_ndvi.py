import ee
import time

def extract_ndvi_degradation_baseline():
    # 1. Initialize with your Project ID
    PROJECT_ID = 'orbital-intelligence-thesis'
    try:
        ee.Initialize(project=PROJECT_ID)
        print("GEE Initialized successfully for Optical Extraction.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 2. Define the exact same Region of Interest (ROI)
    roi = ee.Geometry.Rectangle([56.10, 27.05, 56.40, 27.25])
    print("Region of Interest (ROI) defined.")

    # 3. Define the Temporal Baselines
    pre_start, pre_end = '2025-08-01', '2025-12-31'
    post_start, post_end = '2026-01-01', '2026-03-24'

    # 4. Load the Sentinel-2 Surface Reflectance Collection
    # We must filter out heavily cloudy images since this is an optical sensor
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(roi)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

    # 5. Define the NDVI Calculation Function
    def add_ndvi(image):
        # NDVI = (NIR - Red) / (NIR + Red)
        # In Sentinel-2: B8 is NIR, B4 is Red
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    # 6. Map the function over the collection to calculate NDVI for every image
    s2_ndvi_collection = s2_collection.map(add_ndvi)

    # 7. Create Pre and Post Composites (using median to clear remaining cloud shadows)
    pre_ndvi = s2_ndvi_collection.filterDate(pre_start, pre_end).select('NDVI').median().clip(roi)
    post_ndvi = s2_ndvi_collection.filterDate(post_start, post_end).select('NDVI').median().clip(roi)

    # 8. Calculate Ecological Degradation
    # A negative value indicates vegetation loss or scorch marks
    ndvi_difference = post_ndvi.subtract(pre_ndvi).rename('ndvi_degradation')

    # 9. Export the Data to Google Drive
    export_task = ee.batch.Export.image.toDrive(
        image=ndvi_difference,
        description='NDVI_Degradation_Baseline_Test',
        folder='Orbital_Intelligence_Data', 
        fileNamePrefix='ndvi_degradation_baseline',
        region=roi.getInfo()['coordinates'],
        scale=10, # Sentinel-2 optical resolution is 10 meters
        maxPixels=1e10,
        fileFormat='GeoTIFF'
    )

    print("Starting NDVI export task to Google Drive...")
    export_task.start()

    # 10. Monitor the Task
    print("Task submitted. Waiting for Earth Engine to process the optical composite...")
    while export_task.active():
        print(f"Task state: {export_task.status()['state']}...")
        time.sleep(15)

    print(f"Task finished with state: {export_task.status()['state']}")
    if export_task.status()['state'] == 'COMPLETED':
        print("Success! The NDVI GeoTIFF is alongside your SAR data in Google Drive.")
    else:
        print(f"Error details: {export_task.status().get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    extract_ndvi_degradation_baseline()
