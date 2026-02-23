#tsp_io.py

from typing import List, Tuple

City = Tuple[str, float, float]  #name, lat, lon

def load_cities(path: str) -> List[City]:
    
    #reads tsp.dat and returns list of (name, lat, lon)
    
    cities: List[City] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            lat = float(parts[-2])
            lon = float(parts[-1])
            name = " ".join(parts[:-2])

            cities.append((name, lat, lon))

    return cities