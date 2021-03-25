import random as r
from dataclasses import dataclass

import pandas as pd
from geopy.distance import distance
from tqdm import tqdm

from route_planner.airports import airports
from route_planner.config import *
from route_planner.data import Data, test_data, train_data
from route_planner.schedule import schedule

tqdm.pandas()


@dataclass(frozen=True, eq=True)
class Features:
    featurized_fn = {
        TRAIN: config.FEATURIZED_TRAIN,
        TEST: config.FEATURIZED_TEST,
    }
    augmented_fn = {TRAIN: config.AUGMENTED_TRAIN, TEST: config.AUGMENTED_TEST}

    data: Data

    def instantiate(self):
        split = self.data.split
        if self.featurized_fn[split].exists():
            return pd.read_csv(self.featurized_fn[split]).set_index(
                [QUERY_ID, SUGGESTION_ID, PART_ID]
            )

        df = self.data.dataframe.copy()

        df[FLIGHTS_STRING_LENGTH] = df[FLIGHTS].progress_apply(len)
        df.drop([FLIGHTS], axis=1, inplace=True)

        df[START_DATE] = pd.to_datetime(df[START_TIME], unit="s")
        df[START_WEEKDAY] = df[START_DATE].dt.weekday
        df[START_WEEK_OF_YEAR] = df[START_DATE].dt.isocalendar().week
        df[START_HOUR] = df[START_DATE].dt.hour
        df[START_MINUTE] = df[START_DATE].dt.minute
        df[START_TIME_EPOCH] = df[START_TIME].astype("int64") // 1e6
        df[START_EARLY_NIGHT] = (df[START_HOUR] > 19) | (df[START_HOUR] < 3)

        df[HOPS] = df.progress_apply(self.hops, axis=1)

        df[DIRECT_TARGET_DISTANCE] = df.progress_apply(
            self.direct_target_distance, axis=1
        )
        df[HYPOTHETICAL_TIME_ON_FOOT] = df.progress_apply(
            self.hypothetical_time_on_foot, axis=1
        )

        df[ITINERARY_DISTANCE] = df.progress_apply(self.itinerary_distance, axis=1)
        df[ITINERARY_TIME] = df.progress_apply(self.itinerary_time, axis=1)
        df[AVERAGE_FLIGHT_DISTANCE] = (df[ITINERARY_DISTANCE] / df[HOPS]).fillna(0)

        df[DEPARTURE_TRANSFER_DISTANCE] = df.progress_apply(
            self.departure_transfer_distance, axis=1
        )

        df[DEPARTURE_TRANSFER_TIME] = df.progress_apply(
            self.departure_transfer_time, axis=1
        )

        df[PREP_TIME] = df.progress_apply(self.prep_time, axis=1)

        df[ARRIVAL_TRANSFER_DISTANCE] = df.progress_apply(
            self.arrival_transfer_distance, axis=1
        )

        df[ARRIVAL_TRANSFER_TIME] = df.progress_apply(
            self.arrival_transfer_time, axis=1
        )

        df[TIME_IN_AIR] = df.progress_apply(self.time_in_air, axis=1)
        df[WASTED_TIME] = df.progress_apply(self.wasted_time, axis=1)

        df[PLANNED_END_TIME] = df[START_TIME] + df[ITINERARY_TIME]
        df[PLANNED_END_DATE] = pd.to_datetime(df[PLANNED_END_TIME], unit="s")
        df[PLANNED_END_WEEKDAY] = df[PLANNED_END_DATE].dt.weekday
        df[PLANNED_END_WEEK_OF_YEAR] = df[PLANNED_END_DATE].dt.isocalendar().week
        df[PLANNED_END_HOUR] = df[PLANNED_END_DATE].dt.hour
        df[PLANNED_END_MINUTE] = df[PLANNED_END_DATE].dt.minute
        df[PLANNED_END_TIME_EPOCH] = df[PLANNED_END_TIME].astype("int64") // 1e6
        df[PLANNED_END_EARLY_NIGHT] = (df[PLANNED_END_HOUR] > 19) | (
            df[PLANNED_END_HOUR] < 3
        )

        df[AIR_OPPORTUNITY_COST] = df[TIME_IN_AIR] * self.data.get_speed(df)

        df.drop(
            [
                START_DATE,
                PLANNED_END_DATE,
            ],
            axis=1,
            inplace=True,
        )
        df.to_csv(self.featurized_fn[split], index=False)
        return df.set_index([QUERY_ID, SUGGESTION_ID, PART_ID])

    def augmented(self):
        split = self.data.split
        if self.augmented_fn[split].exists():
            return pd.read_csv(self.augmented_fn[split]).set_index(
                [QUERY_ID, SUGGESTION_ID, PART_ID]
            )

        df = self.instantiate().reset_index()
        df[NOT_CANCELLED] = df.progress_apply(self.not_cancelled, axis=1)
        df[REAL_ITINERARY_TIME] = df.progress_apply(self.real_itinerary_time, axis=1)

        df.to_csv(self.augmented_fn[split], index=False)
        return df.set_index([QUERY_ID, SUGGESTION_ID, PART_ID])

    @staticmethod
    def cast_categorical(df):
        categorical = [
            HOPS,
            START_WEEKDAY,
            START_WEEK_OF_YEAR,
            START_HOUR,
            START_MINUTE,
            START_EARLY_NIGHT,
            PLANNED_END_WEEKDAY,
            PLANNED_END_WEEK_OF_YEAR,
            PLANNED_END_HOUR,
            PLANNED_END_MINUTE,
            PLANNED_END_EARLY_NIGHT,
            FLIGHTS_STRING_LENGTH,
        ]
        for col in categorical:
            if col in df:
                df[col] = df.loc[:, col].astype("category")

        return df

    def get_key(self, row):
        return self.data.format_key((row[QUERY_ID], row[SUGGESTION_ID]))

    def departure_transfer_distance(self, row):
        key = self.get_key(row)
        flights = self.data.itinerary[key]
        val = 0.0
        if flights:
            flight_id = flights[0]
            departure_airport = airports.latlon[schedule.departure_airport[flight_id]]
            origin_coor = self.data.origin_coor[key]
            val = distance(departure_airport, origin_coor).m

        return val

    def departure_transfer_time(self, row):
        return self.departure_transfer_distance(row) / self.data.get_speed(row)

    def arrival_transfer_distance(self, row):
        key = self.get_key(row)
        flights = self.data.itinerary[key]
        val = 0.0
        if flights:
            flight_id = flights[-1]
            arrival_airport = airports.latlon[schedule.arrival_airport[flight_id]]
            destination_coor = self.data.destination_coor[key]
            val = distance(arrival_airport, destination_coor).m

        return val

    def arrival_transfer_time(self, row):
        return self.arrival_transfer_distance(row) / self.data.get_speed(row)

    def distance_by_air(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        return sum(self.data.flight_distance(flight_id) for flight_id in flights)

    def direct_target_distance(self, row):
        key = self.get_key(row)
        return self.data.direct_target_distance[key]

    def hypothetical_time_on_foot(self, row):
        dist = self.direct_target_distance(row)
        return dist / self.data.get_speed(row)

    def itinerary_distance(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        if not flights:
            return self.data.direct_target_distance[key]
        else:
            val = self.departure_transfer_distance(row)
            now = schedule.departure_planned_time[flights[0]]
            for flight_id in flights:
                departure_planned_time = schedule.departure_planned_time[flight_id]
                arrival_planned_time = schedule.arrival_planned_time[flight_id]
                departure_airport = airports.latlon[
                    schedule.departure_airport[flight_id]
                ]

                if now > departure_planned_time:
                    destination_coor = self.data.destination_coor[key]
                    return val + distance(departure_airport, destination_coor)
                else:
                    val += schedule.flight_distance(flight_id)
                    now += arrival_planned_time - now

            return val + self.arrival_transfer_distance(row)

    def prep_time(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        if not flights:
            return 0.0
        else:
            now = schedule.departure_planned_time[flights[0]]
            return (now - self.data.get_start_time(row),)

    def itinerary_time(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        if not flights:
            return self.data.direct_target_distance[key] / self.data.get_speed(row)
        else:
            now = schedule.departure_planned_time[flights[0]]
            val = max(
                self.departure_transfer_time(row),
                now - self.data.get_start_time(row),
            )
            for flight_id in flights:
                departure_planned_time = schedule.departure_planned_time[flight_id]
                arrival_planned_time = schedule.arrival_planned_time[flight_id]
                departure_airport = airports.latlon[
                    schedule.departure_airport[flight_id]
                ]

                if now > departure_planned_time:
                    destination_coor = self.data.destination_coor[key]
                    return val + distance(
                        departure_airport, destination_coor
                    ).m / self.data.get_speed(row)
                else:
                    offset = arrival_planned_time - now
                    val += offset
                    now += offset

            return val + self.arrival_transfer_time(row)

    def time_in_air(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        val = 0.0
        if not flights:
            return val
        else:
            now = schedule.departure_planned_time[flights[0]]

            for flight_id in flights:
                departure_planned_time = schedule.departure_planned_time[flight_id]
                arrival_planned_time = schedule.arrival_planned_time[flight_id]

                if now > departure_planned_time:
                    return val
                else:
                    offset = arrival_planned_time - now
                    val += self.data.flight_length(flight_id)
                    now += offset

            return val

    def wasted_time(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        val = 0.0
        if not flights:
            return val
        else:
            now = schedule.departure_planned_time[flights[0]]

            for flight_id in flights:
                departure_planned_time = schedule.departure_planned_time[flight_id]
                arrival_planned_time = schedule.arrival_planned_time[flight_id]

                if now > departure_planned_time:
                    return val
                else:
                    offset = arrival_planned_time - now
                    val += offset - self.data.flight_length(flight_id)
                    now += offset

            return val

    def hops(self, row):
        key = self.get_key(row)
        return self.data.hops[key]

    def not_cancelled(self, row):
        key = self.get_key(row)
        flights = self.data.itinerary[key]

        proba = 1
        if not flights:
            return proba

        for flight_id in flights:
            proba *= 1 - schedule.cancelled_proba_by_id.get(flight_id, 0)

        return proba

    @staticmethod
    def random_component(v):
        if v > 0.6:
            return r.random() * v
        else:
            return (2 * r.random() - 1) * v * 1e-3

    def real_itinerary_time(self, row):
        key = self.get_key(row)

        flights = self.data.itinerary[key]

        if not flights:
            return self.data.direct_target_distance[key] / self.data.get_speed(row)
        else:
            now = schedule.departure_planned_time[flights[0]]
            val = max(
                self.departure_transfer_time(row),
                now - self.data.get_start_time(row),
            )
            for flight_id in flights:
                arrival_delay = schedule.arrival_delay_by_id[flight_id]
                departure_delay = schedule.departure_delay_by_id[flight_id]
                cancelled = schedule.cancelled_proba_by_id[flight_id] > 0.5
                departure_planned_time = schedule.departure_planned_time[flight_id]
                arrival_planned_time = schedule.arrival_planned_time[flight_id]
                departure_airport = airports.latlon[
                    schedule.departure_airport[flight_id]
                ]

                if cancelled or (
                    now
                    > departure_planned_time
                    + self.random_component(departure_delay) * departure_planned_time
                ):
                    destination_coor = self.data.destination_coor[key]
                    return val + distance(
                        departure_airport, destination_coor
                    ).m / self.data.get_speed(row)

                offset = (
                    arrival_planned_time + self.random_component(arrival_delay) - now
                )
                val += offset
                now += offset

            return val + self.arrival_transfer_time(row)


train_features = Features(train_data)
test_features = Features(test_data)
