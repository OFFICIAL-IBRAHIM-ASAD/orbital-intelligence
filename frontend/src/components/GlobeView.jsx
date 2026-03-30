import React from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import Map from 'react-map-gl/maplibre'; // Standard default import
import 'maplibre-gl/dist/maplibre-gl.css';

// Center the initial camera view over our target conflict zone
const INITIAL_VIEW_STATE = {
  longitude: 56.25, // Strait of Hormuz / Bandar Abbas
  latitude: 27.15,
  zoom: 4.5,
  pitch: 50, // Tilt the camera for a 3D orbital perspective
  bearing: 0
};

// Sleek, dark-mode base map
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

export default function GlobeView({ claims, onMapClick }) {
  
  // Transform the ledger database claims into visual 3D points
  const layers = [
    new ScatterplotLayer({
      id: 'conflict-claims-layer',
      data: claims.filter(d => d.longitude && d.latitude), // Only plot claims that have coordinates
      pickable: true,
      opacity: 0.8,
      stroked: true,
      filled: true,
      radiusScale: 5000, 
      radiusMinPixels: 6,
      radiusMaxPixels: 25,
      lineWidthMinPixels: 2,
      
      getPosition: d => [d.longitude, d.latitude],
      
      getFillColor: d => {
        if (d.status === 'Verified') return [76, 175, 80]; // Green
        if (d.status === 'Partially Supported') return [255, 152, 0]; // Orange
        if (d.status === 'Pending Verification' || d.status === 'Analyzing Orbital Data...') return [100, 100, 255]; // Blue
        return [244, 67, 54]; // Red (Unverified)
      },
      getLineColor: () => [255, 255, 255], 
    })
  ];

  return (
    <DeckGL
      initialViewState={INITIAL_VIEW_STATE}
      controller={true} // Allows user to pan, zoom, and rotate
      layers={layers}
      
      // NEW: Extract the coordinates from the click event and pass them to the backend
      onClick={(info) => {
        if (onMapClick && info.coordinate) {
          // info.coordinate is an array: [longitude, latitude]
          onMapClick(info.coordinate[0], info.coordinate[1]);
        }
      }}

      // Creates a simple hover tooltip showing the claim status
      getTooltip={({object}) => object && {
        html: `<b>${object.claim_type}</b><br/>Status: ${object.status}`,
        style: {
          backgroundColor: '#1e1e1e',
          color: '#fff',
          fontSize: '0.9em',
          padding: '10px',
          borderRadius: '4px',
          border: '1px solid #444'
        }
      }}
    >
      <Map reuseMaps mapStyle={MAP_STYLE} />
    </DeckGL>
  );
}