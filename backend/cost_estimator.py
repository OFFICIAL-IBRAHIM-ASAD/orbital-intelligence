"""
Infrastructure Damage Cost Estimator
=====================================
Converts the AI-derived damage_percentage into an approximate USD cost
based on regional construction economics for Iran and Israel.

Methodology
-----------
1.  The ROI is a 2.5 km-radius buffer (bounds ≈ 5 km × 5 km = 25 km²).
2.  Sentinel-2 has 10 m native resolution → each pixel ≈ 100 m².
3.  damaged_area_m2 = damage_percentage / 100  ×  roi_area_m2
4.  Not all land is built-up; we apply a country-specific *urban density*
    factor representing the fraction of land covered by structures.
5.  cost = damaged_area_m2 × urban_density × cost_per_m2 × infra_multiplier
    where infra_multiplier accounts for roads, utilities, and civil works
    beyond the buildings themselves (≈ 1.5×).

Regional cost assumptions (USD / m², 2025 values):
    Iran   – ~$450/m²  (mid-range emerging market construction cost)
    Israel – ~$2,800/m² (developed market, high land and labour costs)

These are conservative academic estimates; actual costs vary widely.
"""

# --------------------------------------------------------------------------- #
# Regional economic parameters                                                 #
# --------------------------------------------------------------------------- #

REGION_PARAMS = {
    "iran": {
        "cost_per_m2_usd": 450,
        "urban_density":   0.30,   # 30 % of ROI assumed built-up (typical Iranian city outskirts)
        "infra_multiplier": 1.5,
    },
    "israel": {
        "cost_per_m2_usd": 2800,
        "urban_density":   0.45,   # 45 % (dense urban corridors in central/northern Israel)
        "infra_multiplier": 1.5,
    },
    "default": {
        "cost_per_m2_usd": 800,
        "urban_density":   0.30,
        "infra_multiplier": 1.5,
    },
}

# ROI geometry (5 km × 5 km square centred on the buffered point)
ROI_AREA_M2 = 5_000 * 5_000   # 25 000 000 m²

# Geographic bounding boxes for country detection
# (lat_min, lat_max, lon_min, lon_max)
IRAN_BBOX   = (25.0, 40.0, 44.0, 64.0)
ISRAEL_BBOX = (29.0, 34.0, 34.0, 36.0)


def detect_region(lat: float, lon: float) -> str:
    """Return 'iran', 'israel', or 'default' based on coordinates."""
    if (IRAN_BBOX[0] <= lat <= IRAN_BBOX[1] and
            IRAN_BBOX[2] <= lon <= IRAN_BBOX[3]):
        return "iran"
    if (ISRAEL_BBOX[0] <= lat <= ISRAEL_BBOX[1] and
            ISRAEL_BBOX[2] <= lon <= ISRAEL_BBOX[3]):
        return "israel"
    return "default"


def estimate_damage_cost(
    damage_percentage: float,
    lat: float,
    lon: float,
    roi_area_m2: float = ROI_AREA_M2,
) -> dict:
    """
    Calculate a basic infrastructure replacement cost from AI damage output.

    Args:
        damage_percentage: Fraction of sector flagged as damaged (0–100).
        lat:               Target latitude.
        lon:               Target longitude.
        roi_area_m2:       Total area of the region of interest in m².

    Returns:
        dict with keys:
            region            – detected country / region label
            damaged_area_m2   – estimated m² of damaged infrastructure
            cost_usd          – total estimated replacement cost in USD
            cost_display      – human-readable formatted string
            assumptions       – dict of unit costs used
    """
    if not (0.0 <= damage_percentage <= 100.0):
        raise ValueError(f"damage_percentage must be 0–100; got {damage_percentage}")

    region = detect_region(lat, lon)
    params = REGION_PARAMS[region]

    damaged_fraction  = damage_percentage / 100.0
    damaged_area_m2   = damaged_fraction * roi_area_m2 * params["urban_density"]
    cost_usd          = damaged_area_m2 * params["cost_per_m2_usd"] * params["infra_multiplier"]

    # Format for display
    if cost_usd >= 1_000_000_000:
        cost_display = f"~${cost_usd / 1_000_000_000:.2f} billion USD"
    elif cost_usd >= 1_000_000:
        cost_display = f"~${cost_usd / 1_000_000:.2f} million USD"
    else:
        cost_display = f"~${cost_usd:,.0f} USD"

    return {
        "region":           region.title(),
        "damaged_area_m2":  round(damaged_area_m2, 2),
        "cost_usd":         round(cost_usd, 2),
        "cost_display":     cost_display,
        "assumptions": {
            "cost_per_m2_usd":  params["cost_per_m2_usd"],
            "urban_density":    params["urban_density"],
            "infra_multiplier": params["infra_multiplier"],
            "roi_area_m2":      roi_area_m2,
        },
    }
