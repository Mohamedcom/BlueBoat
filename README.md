# Smart Surface Vessel – Marine Cognition Ultima 3.0

A Python‐based autonomous ocean intelligence platform that combines real-time data fetching, AI-driven risk detection, and an interactive Streamlit/Folium interface. The system ingests satellite, buoy, AIS and pollution data, runs anomaly and oil-spill risk models, and visualizes vessel positions and environmental layers on a live dashboard.

## Repository Structure

- **main.py**  
  Entry point: initializes and runs the Streamlit UI :contentReference[oaicite:0]{index=0}

- **interface.py**  
  `HoloMarineInterface` class: Streamlit/Folium front-end, map rendering, controls, and analytics panels :contentReference[oaicite:1]{index=1}

- **engine.py**  
  `QuantumMarineEngine`: orchestrates data fetching, model training & inference, alert management, and quality scoring :contentReference[oaicite:2]{index=2}

- **data_fetcher.py**  
  `DataFetcher`: asynchronous retrieval and parsing of real-time marine data sources (NOAA, AIS, pollution, fishing, weather) :contentReference[oaicite:3]{index=3}

- **features.py**  
  Helpers for loading historical incident data and fetching SST/weather with retry logic :contentReference[oaicite:4]{index=4}

- **models.py**  
  Keras model definitions for threat, current, and vessel-behavior prediction :contentReference[oaicite:5]{index=5}

- **alerts.py**  
  `MarineAlert` dataclass and `PriorityAlertQueue` for severity-based alert prioritization :contentReference[oaicite:6]{index=6}

- **vessel_tracker.py**  
  `SyntheticVesselTracker`: simulates vessel positions on predefined routes, ensuring separation and land-mask checks[oaicite:7]{index=7}

- **config.py**  
  Constants: data source URLs, region bounds, hazard definitions, and land-mask geometry :contentReference[oaicite:8]{index=8}
