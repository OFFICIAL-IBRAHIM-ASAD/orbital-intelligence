import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import GlobeView from './components/GlobeView';

function App() {
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);

  // 1. Initial Load: Fetch the ledger data
  useEffect(() => {
    const fetchClaims = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/claims');
        setClaims(response.data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching claims from backend:", error);
        setLoading(false);
      }
    };
    fetchClaims();
  }, []);

  // 2. The Verification Trigger: Calls the Dynamic Endpoint
  const verifyClaim = async (id) => {
    try {
      // 1. Immediately update UI to show we are analyzing
      setClaims(claims.map(claim => 
        claim.id === id ? { ...claim, status: "Analyzing Orbital Data..." } : claim
      ));

      // 2. Fetch the data from the backend using the specific ID
      const response = await fetch("http://localhost:8000/api/analyze/" + id);
      const data = await response.json(); 

      // NEW SAFETY NET: If the backend returns an error, throw it to the catch block
      if (data.error) {
        throw new Error(data.error);
      }

      // 3. Update the specific claim with the newly generated unique heatmaps
      setClaims(claims.map(claim => 
        claim.id === id ? { 
          ...claim, 
          status: data.verdict || "System Offline",
          details: data.details || "No details provided.",
          xai_heatmap_url: data.xai_heatmap_url,
          ndvi_map_url: data.ndvi_map_url 
        } : claim
      ));

    } catch (error) {
      console.error("Error connecting to Orbital AI:", error);
      // Fallback if the backend crashes or loses memory
      setClaims(claims.map(claim => 
        claim.id === id ? { ...claim, status: "Target Not Found in Database" } : claim
      ));
    }
  };
  
  // 3. Captures map clicks and sends them to the backend queue
  const handleMapClick = async (lng, lat) => {
    try {
      const response = await fetch("http://localhost:8000/api/target", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ latitude: lat, longitude: lng })
      });
      
      const newTarget = await response.json();
      
      // Update our UI list with the newly created target containing the real place name
      setClaims([...claims, newTarget]);
    } catch (error) {
      console.error("Failed to designate target:", error);
    }
  };

  return (
    <div className="dashboard-container">
      <div className="globe-container">
        <GlobeView claims={claims} onMapClick={handleMapClick} />
      </div>
      
      <div className="sidebar">
        <h2>Ledger Database</h2>
        <hr style={{ borderColor: '#333', width: '100%' }} />
        {loading ? (
          <p>Connecting to backend...</p>
        ) : (
          claims.map(claim => (
            <div key={claim.id} className="claim-card">
              <h3>{claim.claim_type}</h3>
              <p style={{ fontSize: '0.9em', color: '#aaa' }}>{claim.description}</p>
              
              {/* NEW SAFETY NET: (claim.status || '') prevents React from crashing if status is undefined */}
              <p>Status: <span className={
                (claim.status || '').includes('Verified') ? 'verified' : 
                (claim.status || '').includes('Partially') ? 'partial' : 
                claim.status === 'Pending Verification' ? 'pending' : 'unverified'
              }>{claim.status}</span></p>

              {/* Display the AI's mathematical details if they exist */}
              {claim.details && (
                <p style={{ fontSize: '0.85em', color: '#ccc', marginTop: '5px' }}>
                  {claim.details}
                </p>
              )}

              {/* The Interactive Verification Button */}
              {claim.status === 'Pending Verification' && claim.latitude && (
                <button 
                  className="verify-btn" 
                  onClick={() => verifyClaim(claim.id)}
                >
                  Run AI Geospatial Verification
                </button>
              )}

              {/* XAI Heatmap (Structural Damage) */}
              {claim.xai_heatmap_url && (
                <div className="xai-container" style={{ marginTop: '15px', borderTop: '1px solid #333', paddingTop: '10px' }}>
                  <h4 style={{ fontSize: '0.9em', marginBottom: '8px' }}>AI Structural Damage (SAR)</h4>
                  <img 
                    src={`${claim.xai_heatmap_url}?t=${new Date().getTime()}`} 
                    alt="Structural Confidence Heatmap" 
                    style={{ width: '100%', borderRadius: '4px', border: '1px solid #444' }} 
                  />
                </div>
              )}

              {/* NDVI Environmental Map */}
              {claim.ndvi_map_url && (
                <div className="ndvi-container" style={{ marginTop: '15px', borderTop: '1px solid #333', paddingTop: '10px' }}>
                  <h4 style={{ fontSize: '0.9em', marginBottom: '8px' }}>Environmental Degradation (NDVI)</h4>
                  <p style={{ fontSize: '0.8em', color: '#888', marginBottom: '10px' }}>
                    Red/Orange indicates vegetation loss, cratering, or scorched earth. Green indicates healthy terrain.
                  </p>
                  <img 
                    src={`${claim.ndvi_map_url}?t=${new Date().getTime()}`} 
                    alt="NDVI Environmental Map" 
                    style={{ width: '100%', borderRadius: '4px', border: '1px solid #444' }} 
                  />
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;