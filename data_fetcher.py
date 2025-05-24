import aiohttp
import io
import asyncio
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict
import numpy as np
from config import SAUDI_WATERS
from datetime import datetime, timezone

class DataFetcher:
    """Asynchronously fetch and parse all marine data sources."""
    def __init__(self):
        self.session: aiohttp.ClientSession = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Kick off all fetches in parallel and return a dict of DataFrames."""
        await self._ensure_session()
        tasks = {
            'satellite_sst':   self._get_satellite_data(),
            'buoy_data':       self._get_buoy_data(),
            'pollution_data':  self._get_pollution_data(),
            'fishing_data':    self._get_fishing_data(),
            'oil_spills':      self._get_oil_spills(),
            'ais_vessels':     self._get_ais_vessels(),
            'weather' :        self._get_weather_data()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        # Map keys back to results
        return dict(zip(tasks.keys(), [
            r if isinstance(r, pd.DataFrame) else pd.DataFrame()
            for r in results
        ]))

    async def _get_weather_data(self) -> pd.DataFrame:
        """Fetch weather data for key points in Saudi waters"""
        records = []
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        # Sample points in each region
        for region_name, region in SAUDI_WATERS.items():
            bounds = region['bounds']
            # Generate grid points
            lats = np.linspace(bounds['lat_min'], bounds['lat_max'], 3)
            lons = np.linspace(bounds['lon_min'], bounds['lon_max'], 3)
            
            for lat in lats:
                for lon in lons:
                    try:
                        url = (
                            "https://api.open-meteo.com/v1/forecast?"
                            f"latitude={lat:.4f}&longitude={lon:.4f}"
                            "&hourly=weathercode,windspeed_10m"
                            "&daily=sunrise,sunset"
                            f"&start_date={date_str}&end_date={date_str}&timezone=UTC"
                        )
                        async with self.session.get(url, timeout=10) as resp:
                            if resp.status == 200:
                                js = await resp.json()
                                now = datetime.now(timezone.utc)
                                hour = now.hour
                                
                                # Extract weather data
                                weather_code = js['hourly']['weathercode'][hour]
                                wind_speed = js['hourly']['windspeed_10m'][hour]
                                sunrise = pd.to_datetime(js['daily']['sunrise'][0])
                                sunset = pd.to_datetime(js['daily']['sunset'][0])
                                is_day = sunrise <= now <= sunset
                                
                                records.append({
                                    'latitude': lat,
                                    'longitude': lon,
                                    'weather_code': weather_code,
                                    'wind_speed': wind_speed,
                                    'is_day': is_day,
                                    'region': region_name
                                })
                    except Exception as e:
                        st.error(f"Weather fetch error at ({lat}, {lon}): {str(e)}")
        
        return pd.DataFrame(records)

    async def _get_satellite_data(self) -> pd.DataFrame:
        """Fetch yesterday’s SST & chlorophyll from NOAA ERDDAP."""
        try:
            params = {
                'start': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'end':   datetime.utcnow().strftime('%Y-%m-%d'),
                'variables': 'sst,chlorophyll'
            }
            url = REAL_TIME_SOURCES['satellite_sst']
            async with self.session.get(url, params=params, timeout=15) as resp:
                text = await resp.text()
            df = pd.read_csv(io.StringIO(text), skiprows=[1])
            return df[['latitude', 'longitude', 'sst', 'chlorophyll']].dropna()
        except Exception as e:
            st.error(f"Satellite Error: {e}")
            return pd.DataFrame()

    async def _get_buoy_data(self) -> pd.DataFrame:
        """Fetch latest buoy observations from NOAA NDBC."""
        try:
            url = REAL_TIME_SOURCES['noaa_buoys']
            async with self.session.get(url, timeout=10) as resp:
                text = await resp.text()
            df = pd.read_fwf(
                io.StringIO(text),
                skiprows=[0,1,2,3],
                na_values=['MM']
            )
            return df.rename(
            columns={
                'WTMP':'sst', 
                'WVHT':'wave_height', 
                'CHL':'chlorophyll',
                'PHPH':'ph' 
            }
            )
        except Exception as e:
            st.error(f"Buoy Error: {e}")
            return pd.DataFrame()

    async def _get_pollution_data(self) -> pd.DataFrame:
        """Fetch global plastic pollution features."""
        try:
            url = REAL_TIME_SOURCES['marine_pollution']
            async with self.session.get(url, timeout=10) as resp:
                data = await resp.json()
            features = data.get('features', [])
            records = []
            for feat in features:
                geom = feat.get('geometry', {})
                coords = geom.get('coordinates', [])
                props  = feat.get('properties', {})
                # handle both single-ring and multi-ring coords
                if isinstance(coords[0][0], list):
                    # nested rings
                    for ring in coords:
                        records.append({**props, 'geometry': ring})
                else:
                    records.append({**props, 'geometry': coords})
            return pd.DataFrame(records)
        except Exception as e:
            st.error(f"Pollution Error: {e}")
            return pd.DataFrame()

    async def _get_fishing_data(self) -> pd.DataFrame:
        """Fetch heatmap config & data for fishing activity."""
        try:
            url = REAL_TIME_SOURCES['fishing_activity']
            params = {'region':'middle_east','resolution':'high'}
            async with self.session.get(url, params=params, timeout=15) as resp:
                data = await resp.json()
            df = pd.DataFrame(data.get('data', []))
            # geometry comes as {"coordinates": [...]}
            df['geometry'] = df['geometry'].apply(lambda x: x.get('coordinates', []))
            return df.explode('geometry')
        except Exception as e:
            st.error(f"Fishing Error: {e}")
            return pd.DataFrame()

    async def _get_oil_spills(self) -> pd.DataFrame:
        """Fetch oil spill GeoJSON and flatten into lat/lon records."""
        try:
            url = REAL_TIME_SOURCES['oil_spills']
            async with self.session.get(url, timeout=10) as resp:
                data = await resp.json()
            features = data.get('features', [])
            records = []
            for feat in features:
                geom_type = feat['geometry']['type']
                coords    = feat['geometry']['coordinates']
                props     = feat.get('properties', {})
                if geom_type == 'Point':
                    lon, lat = coords
                    records.append({'latitude': lat, 'longitude': lon, **props})
                elif geom_type == 'Polygon':
                    # coords = [ [ [lon,lat], ... ], ... ]
                    for ring in coords:
                        for lon, lat in ring:
                            records.append({'latitude': lat, 'longitude': lon, **props})
                elif geom_type == 'MultiPolygon':
                    for poly in coords:
                        for ring in poly:
                            for lon, lat in ring:
                                records.append({'latitude': lat, 'longitude': lon, **props})
            return pd.DataFrame(records)
        except Exception as e:
            st.error(f"Oil Spills Error: {e}")
            return pd.DataFrame()
        
    async def _get_ais_vessels(self) -> pd.DataFrame:
        try:
            async with self.session.get(
                REAL_TIME_SOURCES['ais_vessels'],
                timeout=15
            ) as resp:
                data = await resp.json()

            records = []
            for vessel in data.get('vessels', []):
                # Ensure numeric conversion
                try:
                    records.append({
                        'id': str(vessel.get('mmsi', uuid.uuid4())),
                        'type': vessel.get('type', 'Unknown'),
                        'lat': float(vessel.get('latitude', 0)),
                        'lon': float(vessel.get('longitude', 0)),
                        'speed': float(vessel.get('speed', 0)),
                        'course': float(vessel.get('course', 0)),
                        'status': vessel.get('navStatus', 'Underway'),
                        'last_update': datetime.fromtimestamp(vessel.get('timestamp', 0))
                    })
                except ValueError:
                    continue
                    
            return pd.DataFrame(records).dropna(subset=['lat', 'lon'])
        
        except Exception as e:
            st.error(f"AIS Vessel Error: {e}")
            return pd.DataFrame()