#distance.py

import math
from typing import List, Tuple

City = Tuple[str, float, float]

#returns miles between two points using the Haversine formula
def haversine_miles(lat1, lon1, lat2, lon2):

    r = 3958.7613  #earth's radius in miles

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c

# builds a matrix where dist[i][j] is distance from city i to city j
def build_distance_matrix(cities):
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        _, lat1, lon1 = cities[i]

        for j in range(n):
            if i == j:
                continue

            _, lat2, lon2 = cities[j]
            dist[i][j] = haversine_miles(lat1, lon1, lat2, lon2)

    return dist