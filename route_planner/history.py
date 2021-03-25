from dataclasses import dataclass
from functools import cached_property
from route_planner.airports import airports

import pandas as pd

from route_planner.config import *
from route_planner.data import persist_to_csv, read_tag
from route_planner.weather import weather


@dataclass(frozen=True, eq=True)
class History:
    base = {
        TRAIN: {
            HISTORY: read_tag(config.TRAIN_DATA_DIR, config.HISTORY_DATA_TAG),
        },
        TEST: {
            HISTORY: read_tag(config.TEST_DATA_DIR, config.HISTORY_DATA_TAG),
        },
    }
    sample_size = config.HISTORICAL_DATA_SAMPLE_SIZE

    @cached_property
    def collected(self):
        df = (
            pd.concat(
                [self.base[TRAIN][HISTORY], self.base[TEST][HISTORY]],
                ignore_index=True,
            )
            .set_index(FLIGHT_ID)
            .sample(self.sample_size)
            .rename({"part": PART_ID})
        )
        return self.featurize(df)

    @staticmethod
    def featurize(df):
        df = df.copy()
        df.loc[:, DEPARTURE_LAT] = df.progress_apply(airports.departure_lat, axis=1)
        df.loc[:, DEPARTURE_LON] = df.progress_apply(airports.departure_lon, axis=1)
        df.loc[:, ARRIVAL_LAT] = df.progress_apply(airports.arrival_lat, axis=1)
        df.loc[:, ARRIVAL_LON] = df.progress_apply(airports.arrival_lon, axis=1)

        return History.cast_categorical(df)

    @cached_property
    @persist_to_csv(config.AUGMENTED_HISTORY)
    def augmented_with_weather(self):
        df = self.collected.copy()
        df[START_TYPE] = weather.predict(TYPE, self.simulated_departure_weather)
        df[START_LEVEL] = weather.predict(LEVEL, self.simulated_departure_weather)
        df[END_TYPE] = weather.predict(TYPE, self.simulated_arrival_weather)
        df[END_LEVEL] = weather.predict(LEVEL, self.simulated_arrival_weather)
        return df

    @staticmethod
    def cast_categorical(df):
        categorical = [
            START_TYPE,
            START_LEVEL,
            END_TYPE,
            END_LEVEL,
            DEPARTURE_AIRPORT,
            ARRIVAL_AIRPORT,
        ]
        to_remove = [PART_ID]
        for category in categorical:
            if category in df.columns:
                df.loc[:, category] = df[category].astype("category")

        for removed in to_remove:
            if removed in df.columns:
                df.drop([removed], axis=1, inplace=True)
        return df

    @cached_property
    def simulated_departure_weather(self):
        return self.simulate_weather_from_history(start=True)

    @cached_property
    def simulated_arrival_weather(self):
        return self.simulate_weather_from_history(start=False)

    def simulate_weather_from_history(self, start):
        stage = "start" if start else "end"
        tag = "departure" if start else "arrival"

        df = self.collected
        weather_df = weather.collected
        mean_period = (weather_df[END_TIME] - weather_df[START_TIME]).mean()

        if stage == "start":
            start_time = df.loc[
                :, ["departure_planned_time", "departure_actual_time"]
            ].min(axis=1)
            end_time = start_time + mean_period
        else:
            end_time = df.loc[:, ["arrival_planned_time", "arrival_actual_time"]].max(
                axis=1
            )
            start_time = end_time - mean_period

        return pd.DataFrame(
            {
                START_TIME: start_time.astype("int64"),
                END_TIME: end_time.astype("int64"),
                LATITUDE: df[f"{tag}_lat"].astype("float64"),
                LONGITUDE: df[f"{tag}_lon"].astype("float64"),
            }
        )


history = History()
