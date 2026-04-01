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

      // 3. Update the specific claim with all newly generated outputs
      setClaims(claims.map(claim =>
        claim.id === id ? {
          ...claim,
          status: data.verdict || "System Offline",
          details: data.details || "No details provided.",
          xai_heatmap_url:  data.xai_heatmap_url,
          ndvi_map_url:     data.ndvi_map_url,
          ndvi_pre_url:     data.ndvi_pre_url,
          pre_optical_url:  data.pre_optical_url,
          post_optical_url: data.post_optical_url,
          cost_estimate:    data.cost_estimate,
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

              {/* Pre / Post Optical Satellite Imagery */}
              {claim.pre_optical_url && claim.post_optical_url && (
                <div style={{ marginTop: '15px', borderTop: '1px solid #333', paddingTop: '10px' }}>
                  <h4 style={{ fontSize: '0.9em', marginBottom: '8px' }}>Structural Damage – Satellite Comparison</h4>
                  <p style={{ fontSize: '0.8em', color: '#888', marginBottom: '8px' }}>
                    Pre-war (Jan 2025 – Feb 2026) vs Post-war (Mar 2026+)
                  </p>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontSize: '0.75em', color: '#aaa', textAlign: 'center', margin: '0 0 4px' }}>Before</p>
                      <img
                        src={`${claim.pre_optical_url}?t=${new Date().getTime()}`}
                        alt="Pre-war satellite imagery"
                        style={{ width: '100%', borderRadius: '4px', border: '1px solid #555' }}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontSize: '0.75em', color: '#aaa', textAlign: 'center', margin: '0 0 4px' }}>After</p>
                      <img
                        src={`${claim.post_optical_url}?t=${new Date().getTime()}`}
                        alt="Post-war satellite imagery"
                        style={{ width: '100%', borderRadius: '4px', border: '1px solid #555' }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* XAI Heatmap (Structural Damage) */}
              {claim.xai_heatmap_url && (
                <div className="xai-container" style={{ marginTop: '15px', borderTop: '1px solid #333', paddingTop: '10px' }}>
                  <h4 style={{ fontSize: '0.9em', marginBottom: '8px' }}>AI Structural Damage Heatmap</h4>
                  <img 
                    src={`${claim.xai_heatmap_url}?t=${new Date().getTime()}`} 
                    alt="Structural Confidence Heatmap" 
                    style={{ width: '100%', borderRadius: '4px', border: '1px solid #444' }} 
                  />
                </div>
              )}

              {/* NDVI Before / After Environmental Map */}
              {claim.ndvi_pre_url && claim.ndvi_map_url && (
                <div className="ndvi-container" style={{ marginTop: '15px', borderTop: '1px solid #333', paddingTop: '10px' }}>
                  <h4 style={{ fontSize: '0.9em', marginBottom: '8px' }}>Vegetation / Climate Change (NDVI)</h4>
                  <p style={{ fontSize: '0.8em', color: '#888', marginBottom: '8px' }}>
                    Green = healthy vegetation. Red/Orange = scorched earth, vegetation loss.
                  </p>
                  <div style={{ display: 'flex', gap: '6px', marginBottom: '10px' }}>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontSize: '0.75em', color: '#aaa', textAlign: 'center', margin: '0 0 4px' }}>Pre-war NDVI</p>
                      <img
                        src={`${claim.ndvi_pre_url}?t=${new Date().getTime()}`}
                        alt="Pre-war NDVI"
                        style={{ width: '100%', borderRadius: '4px', border: '1px solid #555' }}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontSize: '0.75em', color: '#aaa', textAlign: 'center', margin: '0 0 4px' }}>NDVI Change</p>
                      <img
                        src={`${claim.ndvi_map_url}?t=${new Date().getTime()}`}
                        alt="NDVI degradation delta"
                        style={{ width: '100%', borderRadius: '4px', border: '1px solid #555' }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Infrastructure Damage Cost Estimate */}
              {claim.cost_estimate && (
                <div style={{ marginTop: '15px', borderTop: '1px solid #333', paddingTop: '10px', background: '#161b22', borderRadius: '6px', padding: '12px' }}>
                  <h4 style={{ fontSize: '0.9em', marginBottom: '8px', color: '#f0a500' }}>💰 Infrastructure Damage Cost Estimate</h4>
                  <p style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff', margin: '0 0 6px' }}>
                    {claim.cost_estimate.cost_display}
                  </p>
                  <p style={{ fontSize: '0.8em', color: '#aaa', margin: '0 0 4px' }}>
                    Region: <strong style={{ color: '#ccc' }}>{claim.cost_estimate.region}</strong>
                  </p>
                  <p style={{ fontSize: '0.8em', color: '#aaa', margin: '0 0 4px' }}>
                    Estimated damaged area: <strong style={{ color: '#ccc' }}>{(claim.cost_estimate.damaged_area_m2 / 1_000_000).toFixed(3)} km²</strong>
                  </p>
                  <p style={{ fontSize: '0.75em', color: '#666', margin: '0' }}>
                    Based on {claim.cost_estimate.assumptions?.cost_per_m2_usd} USD/m², {(claim.cost_estimate.assumptions?.urban_density * 100).toFixed(0)}% urban density, ×{claim.cost_estimate.assumptions?.infra_multiplier} infrastructure multiplier.
                  </p>
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