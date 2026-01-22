# --- CONFIGURAZIONE GENERALE ---
CITY_NAME = "Roma"

# Bounding Box per filtrare i dati geografici (Lat/Lon)
# Usato in data_cleaning per eliminare punti fuori città
BOUNDING_BOX = {
    "min_lat": 41.6,
    "max_lat": 42.2,
    "min_lon": 12.2,
    "max_lon": 12.8
}

# --- ROI (Regions of Interest) ---
# Mappiamo coordinate GPS grezze su etichette semantiche.
# Usato in spatial_enrichment e nel frontend per visualizzazione/analisi.
ROME_ROIS = {
    "Colosseo": {"lat_min": 41.889, "lat_max": 41.891, "lon_min": 12.491, "lon_max": 12.494},
    "Vaticano": {"lat_min": 41.900, "lat_max": 41.906, "lon_min": 12.450, "lon_max": 12.460},
    "Pantheon": {"lat_min": 41.897, "lat_max": 41.899, "lon_min": 12.475, "lon_max": 12.478},
    "FontanaTrevi": {"lat_min": 41.900, "lat_max": 41.902, "lon_min": 12.482, "lon_max": 12.484},
    "Trastevere": {"lat_min": 41.886, "lat_max": 41.892, "lon_min": 12.465, "lon_max": 12.472},
    "StazioneTermini": {"lat_min": 41.900, "lat_max": 41.903, "lon_min": 12.498, "lon_max": 12.504},
    "PiazzaNavona": {"lat_min": 41.898, "lat_max": 41.900, "lon_min": 12.472, "lon_max": 12.474},
    "PiazzaSpagna": {"lat_min": 41.905, "lat_max": 41.907, "lon_min": 12.481, "lon_max": 12.484},
    "VillaBorghese": {"lat_min": 41.910, "lat_max": 41.918, "lon_min": 12.475, "lon_max": 12.495}
}

# Calcolo automatico dei centroidi per uso frontend (mappe, grafi)
# Formato: "Nome": [lon, lat] (Standard GeoJSON/PyDeck usa Lon, Lat)
ROI_COORDS = {}
for name, box in ROME_ROIS.items():
    center_lat = (box["lat_min"] + box["lat_max"]) / 2
    center_lon = (box["lon_min"] + box["lon_max"]) / 2
    # PyDeck vuole [Lon, Lat]
    ROI_COORDS[name] = [center_lon, center_lat]

# Configurazione Spark Analysis
ANALYSIS_CONFIG = {
    "APP_NAME": "FlickrFlow",
    "CITY_NAME": CITY_NAME,
    "BOUNDING_BOX": BOUNDING_BOX,
    "USE_GEOFENCE": True
}
