# alerts.py

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Dict

@dataclass
class MarineAlert:
    id: str
    alert_type: str
    location: Tuple[float, float]
    severity: float
    timestamp: datetime
    description: str
    confirmed: bool = False

class PriorityAlertQueue:
    """Max-priority queue of MarineAlert by severity, safe on empty."""
    def __init__(self):
        self._heap: List[Tuple[float, MarineAlert]] = []
        self._index: Dict[str, MarineAlert] = {}

    def push(self, alert: MarineAlert):
        if alert.id not in self._index:
            heapq.heappush(self._heap, (-alert.severity, alert))
            self._index[alert.id] = alert

    def pop(self) -> Optional[MarineAlert]:
        try:
            while True:
                _, alert = heapq.heappop(self._heap)
                # only return alerts still in the index
                if alert.id in self._index:
                    del self._index[alert.id]
                    return alert
        except IndexError:
            return None
