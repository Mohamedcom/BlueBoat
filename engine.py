from pathlib import Path
import uuid
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point

from alerts import MarineAlert, PriorityAlertQueue
from vessel_tracker import SyntheticVesselTracker
from models import ThreatModel, CurrentModel, VesselBehaviorModel
from data_fetcher import DataFetcher
from features import fetch_sst, fetch_weather, WEATHER_DESC, CLIM_SST
from config import SAUDI_WATERS
from shapely.geometry import Point
from shapely.ops import unary_union
from cartopy.io import shapereader
from typing import Tuple

_land_shp   = shapereader.natural_earth(
    resolution='10m', category='physical', name='land'
)
_land_reader = shapereader.Reader(_land_shp)
_land_union  = unary_union(list(_land_reader.geometries()))

class QuantumMarineEngine:
    def __init__(self):
        # Vessel simulator
        self.vessel_tracker = SyntheticVesselTracker()
        # Placeholder for AIS/real vessels
        self.ais_vessels = pd.DataFrame()
        # Train oil‐spill risk model
        self.oil_model, self.oil_scaler = self._train_oil_model()
        # AI/ML models
        self.models = {
            'threat':   ThreatModel().build(),
            'current':  CurrentModel().build(),
            'behavior': VesselBehaviorModel().build(),
            'anomaly':  IsolationForest(n_estimators=200)
        }
        self.scaler = StandardScaler()
        self.alert_queue = PriorityAlertQueue()
        self.fetcher = DataFetcher()
        self.data = {}

    def _train_oil_model(self):
        """Train Random Forest for oil spill risk prediction."""
        data_dir = Path(__file__).parent / "data"
        csv_path = data_dir / "oil_spill.csv"
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['target'])
        y = df['target']
        scaler = StandardScaler().fit(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(scaler.transform(X), y)
        return rf, scaler

    def _predict_oil_risk(self):
        """Generate oil spill risk samples over _both_ Red Sea and Arabian Gulf."""
        today = dt.datetime.utcnow().date()
        samples = []
        # split evenly between the two basins
        per_region = 50

        for region_name, region_cfg in SAUDI_WATERS.items():
            bounds = region_cfg['bounds']
            df_region = self._random_sea_samples(per_region, bounds, today)
            samples.append(df_region)

        # concat both basins
        all_samples = pd.concat(samples, ignore_index=True)
        if all_samples.empty:
            return all_samples

        # run model
        X = all_samples[self.oil_scaler.feature_names_in_]
        probs = self.oil_model.predict_proba(self.oil_scaler.transform(X))[:,1]
        all_samples['risk']  = probs
        all_samples['color'] = all_samples['risk'].apply(
            lambda p: [int(255*p), int(255*(1-p)), 0, 200]
        )
        return all_samples

    def _random_sea_samples(self, n: int, box: Tuple[float,float,float,float], date: dt.date):
        """
        Uniformly sample n points in the bounding box (lat_min, lon_min, lat_max, lon_max),
        reject any point on land, and enrich with SST & weather.
        """
        lat_min, lon_min, lat_max, lon_max = box
        rng = np.random.default_rng()
        rec = []
        date_str = date.strftime('%Y-%m-%d')
        attempts = 0

        while len(rec) < n and attempts < n * 20:
            attempts += 1
            lat = rng.uniform(lat_min, lat_max)
            lon = rng.uniform(lon_min, lon_max)
            # skip land points
            if _land_union.contains(Point(lon, lat)):
                continue

            sst = fetch_sst(lat, lon, date_str)
            if sst == CLIM_SST:
                continue

            w = fetch_weather(lat, lon, date_str)
            desc, icon = WEATHER_DESC.get(w['code'], ("Unknown","❓"))

            rec.append({
                'lat':  lat,
                'lon':  lon,
                'sst':  sst,
                'desc': desc,
                'icon': icon,
                'wind': w['wind']
            })

        return pd.DataFrame(rec)

    def _random_gulf_samples(self, n: int, box: dict, date: dt.date) -> pd.DataFrame:
        """Randomly sample lat/lon in box, fetch SST & weather for features."""
        rng = np.random.default_rng(42)
        rec = []
        date_str = date.strftime('%Y-%m-%d')

        attempts = 0
        while len(rec) < n and attempts < n*15:
            attempts += 1
            lat = rng.uniform(box['lat_min'], box['lat_max'])
            lon = rng.uniform(box['lon_min'], box['lon_max'])
            sst = fetch_sst(lat, lon, date_str)
            if sst == CLIM_SST:
                continue
            w = fetch_weather(lat, lon, date_str)
            desc, icon = WEATHER_DESC.get(w['code'], ("Unknown","❓"))
            rec.append({
                'lat':  lat,
                'lon':  lon,
                'sst':  sst,
                'desc': desc,
                'icon': icon,
                'wind': w['wind']
            })

        return pd.DataFrame(rec)

    async def update_data(self):
        """Fetch all sources and run analyses."""
        self.data = await self.fetcher.fetch_all()
        self._detect_anomalies()
        self._update_alerts()
        self._calculate_water_quality()
        self.data['oil_risk'] = self._predict_oil_risk()

    def _detect_anomalies(self):
        """Run IsolationForest on buoy data and add `anomaly_score`."""
        df = self.data.get('buoy_data', pd.DataFrame())
        if not df.empty:
            feats = df[['sst','wave_height','chlorophyll']].dropna()
            if not feats.empty:
                scores = self.models['anomaly'].score_samples(self.scaler.fit_transform(feats))
                df.loc[feats.index, 'anomaly_score'] = scores

    def _update_alerts(self):
        """Create pollution alerts from GeoJSON features."""
        df = self.data.get('pollution_data', pd.DataFrame())
        for _, row in df.iterrows():
            loc = (row['geometry'][0][1], row['geometry'][0][0])
            alert = MarineAlert(
                id=str(uuid.uuid4()),
                alert_type='pollution',
                location=loc,
                severity=float(row.get('concentration', 1.0)),
                # use datetime.now(), not dt.now()
                timestamp=datetime.now(),
                description=f"Pollution: {row.get('concentration','N/A')}"
            )
            self.alert_queue.push(alert)


    def _calculate_water_quality(self):
        """Combine SST, pH, pollution and oil proximity into a `quality_score`."""
        parts = []
        if 'satellite_sst' in self.data:
            parts.append(self.data['satellite_sst'][['latitude','longitude','sst']])
        if 'buoy_data' in self.data:
            parts.append(self.data['buoy_data'][['latitude','longitude','ph']])

        if not parts:
            return

        df = pd.concat(parts).reset_index(drop=True)
        df['quality_score'] = 1.0

        # Oil proximity penalty
        if 'oil_risk' in self.data:
            oil_pts = [Point(r.lon, r.lat) for _, r in self.data['oil_risk'].iterrows()]
            def oil_pen(row):
                p = Point(row.longitude, row.latitude)
                d = min(p.distance(o) for o in oil_pts) if oil_pts else 0
                # normalize by Gulf width (~8° lat span)
                return min(d / (SAUDI_WATERS['arabian_gulf']['bounds'][3] - SAUDI_WATERS['arabian_gulf']['bounds'][1]), 1)
            df['oil_penalty'] = df.apply(oil_pen, axis=1)
            df['quality_score'] *= (1 - df['oil_penalty'])

        # pH scoring (ideal ~7.5)
        if 'ph' in df.columns:
            df['ph_score'] = np.clip(1 - abs(df['ph'] - 7.5)/3.0, 0, 1)
            df['quality_score'] *= df['ph_score']

        # Pollution concentration penalty
        if 'pollution_data' in self.data:
            poll = self.data['pollution_data']
            def poll_pen(row):
                p = Point(row.longitude, row.latitude)
                # find any GeoJSON polygon within small threshold
                for feat in poll.geometry:
                    if any(Point(x,y).distance(p) < 0.01 for x,y in feat):
                        return 1.0
                return 0.0
            df['pollution_penalty'] = df.apply(poll_pen, axis=1)
            df['quality_score'] *= (1 - df['pollution_penalty'])

        self.data['water_quality'] = df[['latitude','longitude','quality_score']].dropna()
        
    def _calculate_spill_penalty(self, lat, lon, oil_points):
        point = Point(lon, lat)
        min_dist = min(point.distance(oil) for oil in oil_points) if oil_points else 1e6
        return 0.8 if min_dist < 0.2 else 0.0  # 0.2 degrees ~22km

    def _calculate_pollution_penalty(self, lat, lon, pollution_df):
        point = Point(lon, lat)
        for _, row in pollution_df.iterrows():
            if any(Point(c[0], c[1]).distance(point) < 0.1 for c in row['coordinates']):
                return row.get('concentration', 0) / 100
        return 0

    def _detect_anomalies(self):
        df = self.data.get('buoy_data', pd.DataFrame())
        if not df.empty:
            feats = df[['sst','wave_height','chlorophyll']].dropna()
            if not feats.empty:
                scores = self.models['anomaly'].score_samples(self.scaler.fit_transform(feats))
                df.loc[feats.index, 'anomaly_score'] = scores

    def _update_alerts(self):
        df = self.data.get('pollution_data', pd.DataFrame())
        for _, row in df.iterrows():
            loc = (row['geometry'][0][1], row['geometry'][0][0])
            alert = MarineAlert(
                id=str(uuid.uuid4()),
                alert_type='pollution',
                location=loc,
                severity=float(row.get('concentration',1.0)),
                timestamp=dt.now(),
                description=f"Pollution: {row.get('concentration','N/A')}"
            )
            self.alert_queue.push(alert)
