import random
import uuid
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple
from geopy.distance import distance as geo_distance
from shapely.geometry import Point
from config import LAND_UNION

class SyntheticVesselTracker:
    def __init__(self, n: int = 983, min_sep_nm: float = 2.5):
        self.min_sep = min_sep_nm
        self.route_network = {  # Add this FIRST
            'red_sea': [
                (25.0, 37.0), (24.5, 38.5), (23.8, 40.0), (22.5, 41.2), (21.0, 42.5)
            ],
            'arabian_gulf': [
                (27.0, 50.0), (26.5, 51.5), (25.8, 53.0), (25.0, 54.5), (24.2, 56.0)
            ]
        }
        self.vessels = self._generate_initial_fleet(n)  # Initialize AFTER route_network

    def _random_water_point(self) -> Tuple[float, float]:
        # More focused on busy shipping lanes
        return random.choice([
            (26.5 + random.uniform(-0.5, 0.5), 53.0 + random.uniform(-1, 1)),
            (25.2 + random.uniform(-0.3, 0.3), 55.0 + random.uniform(-0.5, 0.5)),
            (22.8 + random.uniform(-0.2, 0.2), 39.5 + random.uniform(-0.7, 0.7))
        ])

    def _generate_initial_fleet(self, n: int) -> List[Dict[str, Any]]:
        fleet = []
        vessel_types = {
            'Container': {'speed_range': (18, 25), 'color': [0, 100, 200, 255]},  # Added alpha channel
            'Tanker': {'speed_range': (12, 20), 'color': [200, 50, 50, 255]},
            'Fishing': {'speed_range': (5, 15), 'color': [50, 150, 50, 255]},
            'Passenger': {'speed_range': (15, 22), 'color': [200, 0, 150, 255]}
        }
        
        while len(fleet) < n:
            lat, lon = self._random_water_point()
            v_type = random.choice(list(vessel_types.keys()))
            speed = random.uniform(*vessel_types[v_type]['speed_range'])
            color = vessel_types[v_type]['color']  # Now includes alpha channel
            route = random.choice(list(self.route_network.keys()))
            
            fleet.append({
                'id': str(uuid.uuid4()),
                'type': v_type,
                'lat': lat,
                'lon': lon,
                'speed': speed,
                'course': 0,
                'color': color,  # Proper RGBA array
                'route': route,
                'waypoint_idx': 0,
                'last_update': datetime.now()
            })
        return fleet

    def update_positions(self):
        """Move vessels along predefined routes"""
        for v in self.vessels:
            route = v['route']
            wp_idx = v['waypoint_idx']
            waypoints = self.route_network[route]
            
            current = (v['lat'], v['lon'])
            target = waypoints[wp_idx]
            
            # Calculate bearing to waypoint
            bearing = self._calculate_bearing(current, target)
            v['course'] = bearing
            
            # Move vessel
            dist_deg = v['speed'] * 0.00027778  # 1 knot ≈ 0.00027778 deg/sec
            new_pos = self._move_along_bearing(current, bearing, dist_deg)
            
            # Update position if in water
            if not LAND_UNION.contains(Point(new_pos[1], new_pos[0])):
                v['lat'], v['lon'] = new_pos
                v['last_update'] = datetime.now()
            
            # Check waypoint progression
            if geo_distance(new_pos, target).nautical < 1:
                v['waypoint_idx'] = (wp_idx + 1) % len(waypoints)

    def _calculate_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Calculate compass bearing between two points"""
        lat1, lon1 = math.radians(start[0]), math.radians(start[1])
        lat2, lon2 = math.radians(end[0]), math.radians(end[1])
        
        d_lon = lon2 - lon1
        x = math.sin(d_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        return math.degrees(math.atan2(x, y)) % 360

    def _move_along_bearing(self, point: Tuple[float, float], bearing: float, distance_deg: float) -> Tuple[float, float]:
        """Move point along bearing by specified distance"""
        lat, lon = math.radians(point[0]), math.radians(point[1])
        angular_distance = distance_deg * math.pi / 180
        
        new_lat = math.asin(math.sin(lat) * math.cos(angular_distance) + 
                  math.cos(lat) * math.sin(angular_distance) * math.cos(math.radians(bearing)))
        
        new_lon = lon + math.atan2(
            math.sin(math.radians(bearing)) * math.sin(angular_distance) * math.cos(lat),
            math.cos(angular_distance) - math.sin(lat) * math.sin(new_lat)
        )
        
        return math.degrees(new_lat), math.degrees(new_lon)
