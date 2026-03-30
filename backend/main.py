import os
import sys
import requests
import json

# 1. FIRST: Tell Python to look in the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. SECOND: Import satellite client and our newly trained AI engine
from backend.satellite_client import fetch_orbital_data, initialize_engine  
from ml_core.inference import run_analysis 

# 3. THIRD: Standard FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
os.makedirs(processed_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=processed_dir), name="assets")

# Initialize the Earth Engine connection when the server starts
try:
    initialize_engine()
except Exception as e:
    print(f"Warning: Earth Engine init failed. {e}")

# --- DYNAMIC IN-MEMORY DATABASE ---
global_ledger = [
    {
        "id": 1,
        "claim_type": "Kinetic Strike",
        "description": "Algorithmic targeting utilized to neutralize logistics hub near Bandar Abbas, Iran.",
        "status": "Pending Verification",
        "latitude": 27.1832, 
        "longitude": 56.2666
    }
]

class TargetCoordinates(BaseModel):
    latitude: float
    longitude: float

# --- REVERSE GEOCODING ENGINE ---
def get_place_name(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
        headers = {'User-Agent': 'OrbitalIntelligenceApp/1.0'}
        response = requests.get(url, headers=headers).json()
        
        if 'name' in response:
            return response['name']
        elif 'address' in response:
            return response['address'].get('country', 'Unknown Region')
        return "Unknown Region"
    except Exception as e:
        return "Unknown Coordinates"

# --- ENDPOINTS ---
@app.get("/api/claims")
def get_all_claims():
    return global_ledger

@app.post("/api/target")
def create_new_target(target: TargetCoordinates):
    new_id = len(global_ledger) + 1
    place_name = get_place_name(target.latitude, target.longitude)
    
    new_claim = {
        "id": new_id,
        "claim_type": "Custom Orbital Inspection",
        "description": f"Targeting coordinates locked on {place_name} ({target.latitude:.4f}, {target.longitude:.4f}).",
        "status": "Pending Verification",
        "latitude": target.latitude,
        "longitude": target.longitude
    }
    
    global_ledger.append(new_claim) 
    return new_claim 

# --- THE LIVE SATELLITE PIPELINE ---
@app.get("/api/analyze/{claim_id}")
def analyze_dynamic_target(claim_id: int):
    claim = next((c for c in global_ledger if c["id"] == claim_id), None)
    if not claim:
        return {"error": "Target not found"}

    print(f"Initiating live satellite uplink for coordinates: {claim['latitude']}, {claim['longitude']}")

    try:
        # 1. Download LIVE True Color data from Space
        # This matches the new return sequence from satellite_client.py
        pre_png, post_png, ndvi_png, _ = fetch_orbital_data(
            lat=claim['latitude'], 
            lon=claim['longitude'], 
            claim_id=claim_id, 
            output_dir=processed_dir
        )
        
        # 2. Define exactly where the AI should save the resulting XAI Heatmap
        xai_filename = f"heatmap_claim_{claim_id}.png"
        xai_output_path = os.path.join(processed_dir, xai_filename)
        
        # 3. Pass the downloaded optical PNGs directly into the PyTorch AI
        analysis_results = run_analysis(
            pre_path=pre_png, 
            post_path=post_png, 
            output_path=xai_output_path
        )
        
        damage_percentage = analysis_results["damage_percentage"]
        ndvi_filename = os.path.basename(ndvi_png)

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return {"error": "Satellite imagery or AI inference failed for this region."}

    # 4. Generate the final empirical verdict based on the model's output
    verdict = "Verified" if damage_percentage > 5.0 else "Unverified"

    return {
        "verdict": verdict,
        "details": f"Live orbital scan and AI analysis complete for {claim['latitude']:.2f}, {claim['longitude']:.2f}. AI confidence shows {damage_percentage}% of sector structurally compromised.",
        "xai_heatmap_url": f"http://localhost:8000/assets/{xai_filename}",
        "ndvi_map_url": f"http://localhost:8000/assets/{ndvi_filename}"
    }