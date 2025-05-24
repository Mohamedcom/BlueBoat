# features.py

import datetime as dt
import functools
import io
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
import warnings

# ── CONFIG ─────────────────────────────────────────────────
INCIDENT_CSV_URL = "https://incidentnews.noaa.gov/raw/incidents.csv"
CLIM_SST         = 29.0
TIMEOUT          = 10
RETRIES          = 3
BACKOFF_FACTOR   = 0.5
STATUS_FORCELIST = [429, 500, 502, 503, 504]

# Arabian Gulf bounding box
GULF_BOUNDS = {
    "lat_min": 24.0,
    "lat_max": 30.0,
    "lon_min": 48.0,
    "lon_max": 56.0
}

# suppress TLS warnings
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# build a retriable session
session = requests.Session()
retry = Retry(
    total=RETRIES,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=STATUS_FORCELIST,
    allowed_methods=["GET", "HEAD", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)


def _rget(url, **kw):
    """Helper: GET with retries and timeout."""
    try:
        r = session.get(url, timeout=TIMEOUT, **kw)
        r.raise_for_status()
        return r
    except Exception:
        return None


def load_incidents():
    """
    Fetch raw oil-spill incidents from NOAA, filter to Arabian Gulf,
    and return only valid lat/lon points.
    """
    r = _rget(INCIDENT_CSV_URL)
    if not r:
        raise RuntimeError("Could not fetch incidents CSV")
    df = pd.read_csv(io.StringIO(r.text))
    # only oil threats
    df = df[df["threat"] == "Oil"].dropna(subset=["lat", "lon"])
    # numeric volume
    df["vol_gal"] = pd.to_numeric(
        df["max_ptl_release_gallons"], errors="coerce"
    ).fillna(0)
    # clamp to Arabian Gulf
    gb = GULF_BOUNDS
    df = df[
        df["lat"].between(gb["lat_min"], gb["lat_max"]) &
        df["lon"].between(gb["lon_min"], gb["lon_max"])
    ]
    return df


@functools.lru_cache(maxsize=512)
def fetch_sst(lat, lon, date_str):
    """Sea-surface temperature at given UTC date (YYYY-MM-DD) via Open-Meteo marine API."""
    now = dt.datetime.utcnow()
    hour = now.hour
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=sea_surface_temperature"
        f"&start_date={date_str}&end_date={date_str}&timezone=UTC"
    )
    r = _rget(url)
    if not r:
        return CLIM_SST
    arr = r.json().get("hourly", {}).get("sea_surface_temperature", [])
    return float(arr[hour]) if hour < len(arr) and arr[hour] is not None else CLIM_SST


@functools.lru_cache(maxsize=512)
def fetch_weather(lat, lon, date_str):
    """Basic weathercode and windspeed from Open-Meteo at given UTC date."""
    now = dt.datetime.utcnow()
    hour = now.hour
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=weathercode,windspeed_10m"
        f"&daily=sunrise,sunset"
        f"&start_date={date_str}&end_date={date_str}&timezone=UTC"
    )
    r = _rget(url)
    if not r:
        return {"code": 0, "wind": 0.0, "is_day": True}
    js = r.json()
    codes = js["hourly"].get("weathercode", [])
    winds = js["hourly"].get("windspeed_10m", [])
    code = int(codes[hour]) if hour < len(codes) else 0
    wind = float(winds[hour]) if hour < len(winds) else 0.0
    sunrise = pd.to_datetime(js["daily"]["sunrise"][0]).tz_convert("UTC")
    sunset = pd.to_datetime(js["daily"]["sunset"][0]).tz_convert("UTC")
    now0 = now.replace(minute=0, second=0, microsecond=0)
    is_day = sunrise <= now0 <= sunset
    return {"code": code, "wind": wind, "is_day": is_day}


# ── Human-friendly weather descriptions & icons ──────────────────
WEATHER_DESC = {
    0:  ("Clear",             "☀️"),
    1:  ("Mainly Clear",      "☀️"),
    2:  ("Partly Cloudy",     "⛅"),
    3:  ("Overcast",          "☁️"),
    45: ("Fog",               "🌫️"),
    48: ("Rime Fog",          "🌫️"),
    61: ("Rain",              "🌧️"),
    63: ("Moderate Rain",     "🌧️"),
    65: ("Heavy Rain",        "🌧️"),
    80: ("Rain Showers",      "🌦️"),
    95: ("Thunderstorm",      "⛈️"),
    99: ("Thunderstorm w/Hail","⛈️")
}
