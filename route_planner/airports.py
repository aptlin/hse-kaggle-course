from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from route_planner.config import AIRPORTS, ARRIVAL_AIRPORT, DEPARTURE_AIRPORT, config


@dataclass(frozen=True, eq=True)
class Airports:
    base = {
        AIRPORTS: pd.read_csv(config.AIRPORTS_PATH),
    }
    sample_size = config.HISTORICAL_DATA_SAMPLE_SIZE

    @cached_property
    def latlon(self):
        details = self.base[AIRPORTS].set_index("index").to_dict()
        arps = details["Latitude"].keys()
        return {
            airport: (details["Latitude"][airport], details["Longitude"][airport])
            for airport in arps
        }

    def departure_lat(self, row):
        return self.latlon.get(row[DEPARTURE_AIRPORT], (0, 0))[0]

    def departure_lon(self, row):
        return self.latlon.get(row[DEPARTURE_AIRPORT], (0, 0))[1]

    def arrival_lat(self, row):
        return self.latlon.get(row[ARRIVAL_AIRPORT], (0, 0))[0]

    def arrival_lon(self, row):
        return self.latlon.get(row[ARRIVAL_AIRPORT], (0, 0))[1]


airports = Airports()
