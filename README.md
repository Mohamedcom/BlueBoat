# Smart Surface Vessel – Marine Cognition Ultima 3.0

A Python‐based autonomous ocean intelligence platform that combines real-time data fetching, AI-driven risk detection, and an interactive Streamlit/Folium interface. The system ingests satellite, buoy, AIS and pollution data, runs anomaly and oil-spill risk models, and visualizes vessel positions and environmental layers on a live dashboard.

## Repository Structure

- **main.py**  
  Entry point: initializes and runs the Streamlit UI :contentReference

- **interface.py**  
  `HoloMarineInterface` class: Streamlit/Folium front-end, map rendering, controls, and analytics panels :contentReference

- **engine.py**  
  `QuantumMarineEngine`: orchestrates data fetching, model training & inference, alert management, and quality scoring :contentReference

- **data_fetcher.py**  
  `DataFetcher`: asynchronous retrieval and parsing of real-time marine data sources (NOAA, AIS, pollution, fishing, weather) :contentReference

- **features.py**  
  Helpers for loading historical incident data and fetching SST/weather with retry logic :contentReference

- **models.py**  
  Keras model definitions for threat, current, and vessel-behavior prediction :contentReference

- **alerts.py**  
  `MarineAlert` dataclass and `PriorityAlertQueue` for severity-based alert prioritization :contentReference

- **vessel_tracker.py**  
  `SyntheticVesselTracker`: simulates vessel positions on predefined routes, ensuring separation and land-mask checks

- **config.py**  
  Constants: data source URLs, region bounds, hazard definitions, and land-mask geometry :contentReference

- **Surveyes.ipynb**  
  Jupyter notebook for interactive data exploration, survey visualizations, and preliminary analysis
