from dataclasses import dataclass
from functools import cached_property, lru_cache

import pandas as pd
from geopy.distance import distance

from route_planner.airports import airports
from route_planner.cancellation import cancellation
from route_planner.config import *
from route_planner.data import persist_to_csv, persist_to_json, read_tag
from route_planner.delay import delay
from route_planner.history import history
from route_planner.weather import weather


@dataclass(frozen=True, eq=True)
class Schedule:
    base = {
        TRAIN: {
            SCHEDULE: read_tag(config.TRAIN_DATA_DIR, config.SCHEDULE_DATA_TAG),
        },
        TEST: {
            SCHEDULE: read_tag(config.TEST_DATA_DIR, config.SCHEDULE_DATA_TAG),
        },
    }

    @cached_property
    def collected(self):
        df = pd.concat(
            [
                self.base[TRAIN][SCHEDULE].set_index(FLIGHT_ID),
                self.base[TEST][SCHEDULE].set_index(FLIGHT_ID),
            ],
        )

        df.rename({"part": PART_ID}, inplace=True)

        return df

    @cached_property
    def scheduled_departure_simulated_weather(self):
        return self.simulate_weather_from_schedule(start=True)

    @cached_property
    def scheduled_arrival_simulated_weather(self):
        return self.simulate_weather_from_schedule(start=False)

    def simulate_weather_from_schedule(self, start: bool):
        stage = "start" if start else "end"
        tag = "departure" if start else "arrival"

        df = history.featurize(self.collected)

        if stage == "start":
            start_time = df.loc[:, DEPARTURE_PLANNED_TIME]
            end_time = start_time + weather.mean_time_period
        else:
            end_time = df.loc[:, ARRIVAL_PLANNED_TIME]
            start_time = end_time - weather.mean_time_period

        return pd.DataFrame(
            {
                START_TIME: start_time.astype("int64"),
                END_TIME: end_time.astype("int64"),
                LATITUDE: df[f"{tag}_lat"].astype("float64"),
                LONGITUDE: df[f"{tag}_lon"].astype("float64"),
            }
        )

    @cached_property
    @persist_to_csv(config.AUGMENTED_SCHEDULE_WEATHER)
    def augmented_with_weather(self):
        df = self.collected.copy()
        df[START_TYPE] = weather.predict(
            TYPE, self.scheduled_departure_simulated_weather
        )
        df[START_LEVEL] = weather.predict(
            LEVEL, self.scheduled_departure_simulated_weather
        )
        df[END_TYPE] = weather.predict(TYPE, self.scheduled_arrival_simulated_weather)
        df[END_LEVEL] = weather.predict(LEVEL, self.scheduled_arrival_simulated_weather)

        return df

    @cached_property
    @persist_to_csv(config.AUGMENTED_SCHEDULE)
    def augmented(self):
        df = history.featurize(self.augmented_with_weather.copy())
        cancelled_proba = cancellation.predict(df)
        df[CANCELLED] = cancelled_proba > 0.5

        arrival_delay = delay.predict(ARRIVAL, df)
        departure_delay = delay.predict(DEPARTURE, df)

        df[ARRIVAL_DELAY] = arrival_delay
        df[DEPARTURE_DELAY] = departure_delay

        return df

    @cached_property
    def cancelled_proba_by_id(self):
        df = self.augmented
        return df[CANCELLED].to_dict()

    @cached_property
    def arrival_delay_by_id(self):
        df = self.augmented
        return df[ARRIVAL_DELAY].to_dict()

    @cached_property
    def departure_delay_by_id(self):
        df = self.augmented
        return df[DEPARTURE_DELAY].to_dict()

    @cached_property
    def route_details_by_id(
        self,
    ):
        return self.collected.loc[
            :,
            [
                DEPARTURE_AIRPORT,
                ARRIVAL_AIRPORT,
                DEPARTURE_PLANNED_TIME,
                ARRIVAL_PLANNED_TIME,
                PART_ID,
            ],
        ].to_dict()

    @cached_property
    def departure_airport(self):
        return self.route_details_by_id[DEPARTURE_AIRPORT]

    @cached_property
    def arrival_airport(self):
        return self.route_details_by_id[ARRIVAL_AIRPORT]

    @cached_property
    def departure_planned_time(self):
        return self.route_details_by_id[DEPARTURE_PLANNED_TIME]

    @cached_property
    def arrival_planned_time(self):
        return self.route_details_by_id[ARRIVAL_PLANNED_TIME]

    @lru_cache(maxsize=None)
    def flight_distance(self, flight_id: str):
        departure_airport = airports.latlon[self.departure_airport[flight_id]]
        arrival_airport = airports.latlon[self.arrival_airport[flight_id]]
        return distance(departure_airport, arrival_airport).m

    @lru_cache(maxsize=None)
    def flight_length(self, flight_id: str):
        return (
            self.arrival_planned_time[flight_id]
            - self.departure_planned_time[flight_id]
        )


schedule = Schedule()
