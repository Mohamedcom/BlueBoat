import asyncio
import threading
import time
import math
from datetime import datetime

import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pandas as pd
import pydeck as pdk
import plotly.express as px
import datetime as dt

from features import load_incidents, fetch_sst, fetch_weather, WEATHER_DESC
from config import REAL_TIME_SOURCES
from engine import QuantumMarineEngine
import base64

# map center for vessel‐alert distance checks
MAP_CENTER = (26.2235, 50.5826)


class HoloMarineInterface:
    """
    Streamlit-based holo UI for Autonomous Ocean Intelligence.
    """
    def __init__(self):
        # Page config
        st.set_page_config(layout="wide", page_title="Marine Cognition Ultima 3.0")
        st.title("🌊 MARINE COGNITION ULTIMA 3.0: Autonomous Ocean Intelligence System")

        # Core engine
        self.core = QuantumMarineEngine()

        # Session-state defaults
        self._init_session_state()

        # Kick off background updates
        self._start_update_loop()
        self._start_vessel_thread()

    def _init_session_state(self):
        if 'marine_env' not in st.session_state:
            st.session_state.marine_env = {
                'layers': {
                    'sst': True,
                    'vessels': True,
                    'pollution': True,
                    'fishing': True,
                    'oil_spills': True,
                    'freshwater': True
                },
                'map_style': 'nautical',
                'last_update': datetime.now()
            }

    def _start_update_loop(self):
        # Async data fetch every 5 minutes
        if 'async_loop' not in st.session_state:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
            st.session_state.async_loop = loop
            # initial load
            asyncio.run_coroutine_threadsafe(self.core.update_data(), loop)

        if 'data_task' not in st.session_state:
            st.session_state.data_task = asyncio.run_coroutine_threadsafe(
                self._data_loop(), st.session_state.async_loop
            )

    async def _data_loop(self):
        while True:
            try:
                await self.core.update_data()
                st.session_state.marine_env['last_update'] = datetime.now()
                await asyncio.sleep(300)        # 5 minutes
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Data Loop Error: {e}")
                await asyncio.sleep(60)


    def _start_vessel_thread(self):
        # Update vessel positions every 10s
        if 'vessel_thread' not in st.session_state:
            def vessel_loop():
                while True:
                    try:
                        self.core.vessel_tracker.update_positions()
                        time.sleep(10)
                       # st.experimental_rerun()
                    except:
                        time.sleep(60)
            t = threading.Thread(target=vessel_loop, daemon=True)
            t.start()
            st.session_state.vessel_thread = t

    def render(self):
        # Build UI sections
        self._render_controls()
        self._render_map()
        self._render_analytics()
        self._render_alerts()

        # On-demand features (heatmap, fish count, temp)
        if st.session_state.get("show_heatmap"):
            st.subheader("🌊 SST Heatmap")
            df = self.core.data.get("satellite_sst", pd.DataFrame())
            if not df.empty:
                fig = px.density_mapbox(
                    df, lat="latitude", lon="longitude", z="sst",
                    radius=10, zoom=5, mapbox_style="open-street-map"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No SST data available.")

        if st.session_state.get("show_vessel"):
            st.subheader("🚢 Vessel Count")
            df = pd.DataFrame(self.core.vessel_tracker.vessels)
            if not df.empty:
                st.bar_chart(df['type'].value_counts())
            else:
                st.info("No vessel data available.")

        if st.session_state.get("show_temp"):
            st.subheader("🌡️ Temperature Status")
            df_b = self.core.data.get("buoy_data", pd.DataFrame())
            if not df_b.empty:
                latest = df_b.iloc[-1]['sst']
                st.metric("Sea-Surface Temperature", f"{latest:.1f} °C")
            else:
                st.info("No buoy data available.")

    def _show_point_analysis(self, lat: float, lon: float):
        # Add risk metric to existing columns
        col5 = st.columns(5)
        with col5:
            risk = self._get_oil_risk_at_point(lat, lon)
            st.metric("Spill Risk", f"{risk:.0%}")

    def _get_oil_risk_at_point(self, lat: float, lon: float) -> float:
        """Get nearest oil spill risk prediction"""
        if 'oil_risk' in self.core.data and not self.core.data['oil_risk'].empty:
            df = self.core.data['oil_risk']
            distances = ((df['lat'] - lat)**2 + (df['lon'] - lon)**2)
            return df.iloc[distances.idxmin()]['risk']
        return 0.0

    def _render_controls(self):
        with st.sidebar:
            # Holographic style container
            st.markdown(
                """
                <div class="cyber-container" style="text-align:center; margin-bottom:16px;">
                  <h2 style="margin:0; padding:8px; color:#fff; font-weight:normal;">
                    ⚡ CONTROL MATRIX
                  </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            env = st.session_state.marine_env
            layers = env['layers']
            
            st.markdown("---")
            st.subheader("Objects")
            layers['sst']        = st.checkbox("Sea-Surface Temperature", value=layers['sst'], key="layer_sst")
            layers['vessels']    = st.checkbox("Vessels",                   value=layers['vessels'], key="layer_vessels")
            layers['pollution']  = st.checkbox("Pollution",                 value=layers['pollution'], key="layer_pollution")
            layers['fishing']    = st.checkbox("Fishing",                   value=layers['fishing'], key="layer_fishing")
            layers['oil_spills'] = st.checkbox("Oil Spills",                value=layers['oil_spills'], key="layer_oil_spills")
            layers['freshwater'] = st.checkbox("Freshwater Suitability",    value=layers['freshwater'], key="layer_freshwater")
                              
            st.markdown("---")
            st.subheader("On-Demand Features")
            st.checkbox("🔥 Show Heatmap",     key="show_heatmap")
            st.checkbox("🚢 Show Vessels Count",  key="show_vessel")
            st.checkbox("🌡️ Show Temp Status", key="show_temp")

            st.markdown("---")
            st.subheader("Map Style")
            style_opts = ['nautical','satellite','dark','outdoors']
            env['map_style'] = st.radio(
                "Choose style:", options=style_opts,
                index=style_opts.index(env['map_style']), key="map_style_radio"
            )

    def _render_map(self):
        st.subheader("LIVE OCEAN INTELLIGENCE MAP")

        env        = st.session_state.marine_env
        layers_cfg = env['layers']
        style      = env['map_style']

        view = pdk.ViewState(
            latitude=MAP_CENTER[0],
            longitude=MAP_CENTER[1],
            zoom=6.5,
            pitch=50,
            bearing=0
        )
        style_map = {
            'nautical':  'mapbox://styles/mapbox/navigation-night-v1',
            'satellite': 'mapbox://styles/mapbox/satellite-v9',
            'dark':      'mapbox://styles/mapbox/dark-v10',
            'outdoors':  'mapbox://styles/mapbox/outdoors-v11'
        }

        deck_layers = []

        # 1) Synthetic vessels
        if layers_cfg['vessels'] and self.core.vessel_tracker.vessels:
            df_synth = pd.DataFrame(self.core.vessel_tracker.vessels)
            df_synth['color'] = df_synth.get('color', [[0,128,255,200]]*len(df_synth))
            deck_layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_synth,
                get_position=["lon","lat"],
                get_fill_color="color",
                get_radius=5000,
                pickable=True,
                opacity=0.8
            ))

        # 2) Real AIS vessels
        if layers_cfg['vessels'] and 'ais_vessels' in self.core.data:
            df_ais = self.core.data['ais_vessels']
            if not df_ais.empty:
                deck_layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=df_ais,
                    get_position=["lon","lat"],
                    get_fill_color=[255,140,0,200],
                    get_radius=2500,
                    pickable=True
                ))

        # 3) SST heatmap
        if layers_cfg['sst'] and 'satellite_sst' in self.core.data:
            df_sst = self.core.data['satellite_sst']
            deck_layers.append(pdk.Layer(
                "HeatmapLayer",
                data=df_sst,
                get_position=['longitude','latitude'],
                get_weight='sst',
                radius=10000,
                intensity=0.8,
                threshold=0.3
            ))

        # 4) Oil-risk samples (predicted points)

        if layers_cfg['oil_spills'] and 'oil_risk' in self.core.data:
            risk_df = self.core.data['oil_risk']
            deck_layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=risk_df,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=3000,
                pickable=True,
                opacity=0.6
            ))

        # 5) Water quality hexagons
        if layers_cfg.get('freshwater') and 'water_quality' in self.core.data:
            wq_df = self.core.data['water_quality']
            deck_layers.append(pdk.Layer(
                "HexagonLayer",
                data=wq_df,
                get_position=['longitude','latitude'],
                get_weight='quality_score',
                radius=5000,
                elevation_scale=50,
                extruded=False,
                coverage=1,
                color_range=[
                    [255,0,0,200],
                    [255,165,0,150],
                    [0,255,0,100],
                    [0,0,255,50]
                ],
                opacity=0.7,
                auto_highlight=True
            ))

        # build and render
        deck = pdk.Deck(
            map_style=style_map[style],
            initial_view_state=view,
            layers=deck_layers,
            tooltip={
                "html": "<b>Type:</b> {type}<br/><b>Speed:</b> {speed} kn",
                "style": {"backgroundColor":"steelblue","color":"white"}
            }
        )
        click_info = st.pydeck_chart(deck)

        # handle clicks
        if click_info and hasattr(click_info, "last_clicked") and isinstance(click_info.last_clicked, dict):
            lat = click_info.last_clicked['lat']
            lon = click_info.last_clicked.get('lng', click_info.last_clicked.get('lon'))
            self._show_point_analysis(lat, lon)


            
    def _show_point_analysis(self, lat: float, lon: float):
        st.subheader(f"🌍 Analysis for [{lat:.4f}, {lon:.4f}]")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Temperature
        with col1:
            st.metric("Water Temperature", self._get_temp_at_point(lat, lon))
        
        # Vessels
        with col2:
            vessels = self._get_nearby_vessels(lat, lon)
            st.metric("Nearby Vessels", len(vessels))
        
        # Pollution
        with col3:
            pollution = self._get_pollution_at_point(lat, lon)
            st.metric("Pollution Risk", pollution)
        
        # Water Quality - NEW
        with col4:
            quality = self._get_water_quality_stats(lat, lon)
            st.metric("Water Quality Score", f"{quality:.0%}")

    def _get_temp_at_point(self, lat: float, lon: float) -> str:
        if 'satellite_sst' in self.core.data:
            df = self.core.data['satellite_sst']
            nearest = df.iloc[((df['latitude']-lat)**2 + (df['longitude']-lon)**2).idxmin()]
            return f"{nearest['sst']:.1f}°C"
        return "N/A"

    def _get_nearby_vessels(self, lat: float, lon: float) -> list:
        radius_km = 20  # 20km radius
        vessels = []
        
        # Check synthetic vessels
        for v in self.core.vessel_tracker.vessels:
            if geo_distance((lat, lon), (v['lat'], v['lon'])).km <= radius_km:
                vessels.append(v)
        
        # Check real vessels
        if 'ais_vessels' in self.core.data:
            df = self.core.data['ais_vessels']
            for _, row in df.iterrows():
                if geo_distance((lat, lon), (row['lat'], row['lon'])).km <= radius_km:
                    vessels.append(row)
        
        return vessels

    def _get_pollution_at_point(self, lat: float, lon: float) -> str:
        if 'pollution_data' in self.core.data:
            df = self.core.data['pollution_data']
            point = Point(lon, lat)
            for _, row in df.iterrows():
                if any(Point(c[0], c[1]).distance(point) < 0.01 for c in row['geometry']):
                    return "High Risk"
        return "Low Risk"
        
    def _get_water_quality_stats(self, lat: float, lon: float) -> float:
        """Calculate average quality score within 20km radius"""
        if 'water_quality' not in self.core.data:
            return 0.0
        
        df = self.core.data['water_quality']
        center_point = Point(lon, lat)
        radius = 0.18  # ~20km in degrees
        
        # Find points within bounding box first for efficiency
        nearby = df[
            (df['latitude'].between(lat - radius, lat + radius)) &
            (df['longitude'].between(lon - radius, lon + radius))
        ]
        
        # Precise distance calculation
        def within_radius(row):
            point = Point(row['longitude'], row['latitude'])
            return center_point.distance(point) <= radius
        
        filtered = nearby[nearby.apply(within_radius, axis=1)]
        
        if not filtered.empty:
            return filtered['quality_score'].mean()
        return 0.0

    def _render_analytics(self):
        st.subheader("AUTONOMOUS ANALYTICS")

        # --- Environmental Conditions (from buoy_data) ---
        st.markdown("### Environmental Conditions")
        df_b = self.core.data.get('buoy_data', pd.DataFrame())
        if not df_b.empty:
            latest = df_b.iloc[-1]
            c1, c2 = st.columns(2)
            c1.metric("Sea‐Surface Temp", f"{latest['sst']:.1f} °C")
            c2.metric("Wave Height",      f"{latest['wave_height']:.1f} m")
            c3, c4 = st.columns(2)
            c3.metric("Anomaly Score",    f"{latest.get('anomaly_score',0):.2f}")
            c4.write("")  # placeholder
        else:
            st.info("No buoy data available.")

        # --- Vessel Traffic Alerts (within ~50 km of center) ---
        st.markdown("### Vessel Traffic Alerts")
        alerts = []
        for v in self.core.vessel_tracker.vessels:
            dist_deg = math.hypot(v['lat']-MAP_CENTER[0], v['lon']-MAP_CENTER[1])
            if dist_deg < 0.5:
                alerts.append({
                    'type':     v['type'],
                    'id':       v['id'],
                    'distance': dist_deg * 111,  # approx km per deg
                    'speed':    v['speed']
                })

        if alerts:
            for a in alerts:
                st.error(f"🚢 {a['type']} Vessel {a['id']} "
                         f"({a['distance']:.1f} km, {a['speed']:.0f} knots)")
        else:
            st.success("No vessels in vicinity.")

    def _render_alerts(self):
        st.subheader("ACTIVE ALERTS (Pollution & Anomalies)")
        count = 0
        while count < 5:
            alert = self.core.alert_queue.pop()
            if not alert:
                break
            st.markdown(
                f"- **{alert.alert_type.title()}** at {alert.location} "
                f"severity {alert.severity:.2f} ({alert.timestamp:%H:%M:%S})"
            )
            count += 1
