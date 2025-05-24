import os
from cartopy.io import shapereader
from shapely.ops import unary_union
import datetime as dt

# suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# NOAA & other realtime sources
REAL_TIME_SOURCES = {
    "satellite_sst":     "https://coastwatch.noaa.gov/erddap/griddap/NOAA_DHW.csv",
 #   "marine_traffic":    "https://services.marinetraffic.com/api/exportvessels",
    "noaa_buoys":        "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt",
    "marine_pollution":  "https://globalplasticwatch.org/api/map-data",
    "fishing_activity":  "https://globalfishingwatch.org/api/v2/heatmap/config",
    "oil_spills":        "https://raw.githubusercontent.com/marine-pollution/MADOS/master/data/oil_spill.geojson",
    "ais_vessels": "https://ais.transparency.everywhere/api/public/vessels"
}

SAUDI_WATERS = {
    "red_sea":      {"bounds": (18.0, 38.0, 28.0, 46.0), "depth_range": (0, 2500)},
    "arabian_gulf": {"bounds": (24.0, 48.0, 30.0, 56.0), "depth_range": (0, 100)}
}

MARINE_HAZARDS = {
    "oil_spill":        {"radius": 5000, "color": [255, 0, 0, 200]},
    "illegal_fishing":  {"radius": 3000, "color": [255, 165, 0, 200]},
    "thermal_plume":    {"radius": 2000, "color": [0, 255, 255, 150]},
    "freshwater_zone": {"radius": 10000, "color": [0, 0, 255, 150]}    
}

# Build a single Shapely geometry covering all land
_land_shp    = shapereader.natural_earth(resolution='10m', category='physical', name='land')
_land_reader = shapereader.Reader(_land_shp)
LAND_UNION   = unary_union(list(_land_reader.geometries()))

WATER_QUALITY_PARAMS = {
    "ideal_ph": (6.5, 8.5),
    "max_pollution": 0.2,
    "oil_spill_radius_km": 20
}

OIL_RISK_COLORS = {
    "low": [0, 255, 0, 200],
    "medium": [255, 255, 0, 200],
    "high": [255, 0, 0, 200]
}
