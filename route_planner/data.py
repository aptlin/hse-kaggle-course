""" Sets up data utilities
"""
import json
from dataclasses import dataclass
from functools import cached_property
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import distance

from route_planner.config import *


def read_part(fn):
    df = pd.read_csv(fn)
    pth = Path(fn).stem
    part = pth.split("_")[0]
    df.loc[:, PART_ID] = part
    df.loc[:, PART_ID] = df[PART_ID].astype("category")
    return df


def read_tag(basedir: Path, tag: str):
    return pd.concat(map(read_part, glob(str(basedir / f"[0-9]_{tag}.csv"))))


def persist_to_json(file_name):
    def decorator(original_func):
        try:
            cache = json.load(open(file_name, "r"))
        except (IOError, ValueError):
            cache = {}

        def inner(*args):
            param = str(json.dumps(args))
            if param not in cache:
                cache[param] = original_func(*args)
                json.dump(cache, open(file_name, "w"))
            return cache[param]

        return inner

    return decorator


def persist_to_csv(file_name):
    def decorator(original_func):
        def inner(*args):
            try:
                df = pd.read_csv(file_name)
            except (IOError, ValueError):
                df = original_func(*args)
                df.to_csv(file_name, index=False)
            return df

        return inner

    return decorator


def persist_to_npy(file_name):
    def decorator(original_func):
        try:
            if file_name[-4:] != ".npy":
                cache = np.load(f"{file_name}.npy", allow_pickle=True)[()]
            else:
                cache = np.load(file_name, allow_pickle=True)[()]
        except (IOError, ValueError):
            cache = {}

        def inner(*args):
            param = str(json.dumps(args))
            if param not in cache:
                cache[param] = original_func(*args)
                np.save(file_name, cache)
            return cache[param]

        return inner

    return decorator


@dataclass(frozen=True, eq=True)
class Data:
    test: bool
    base = {
        TRAIN: {
            DATAFRAME: pd.read_csv(config.TRAIN_DATAFRAME_PATH),
        },
        TEST: {
            DATAFRAME: pd.read_csv(config.TEST_DATAFRAME_PATH),
        },
    }

    @cached_property
    def split(self):
        return TEST if self.test else TRAIN

    @cached_property
    def dataframe(self):
        return self.base[self.split][DATAFRAME]

    @cached_property
    def keys(self):
        return list(
            map(
                self.format_key,
                self.dataframe.set_index([QUERY_ID, SUGGESTION_ID]).index,
            )
        )

    def format_key(self, key):
        return json.dumps(tuple(key))

    @cached_property
    def origin_coor(self):
        details = (
            self.dataframe.set_index([QUERY_ID, SUGGESTION_ID])
            .loc[:, [START_LAT, START_LON]]
            .to_dict()
        )
        keys = details[START_LAT].keys()
        return {
            self.format_key(key): (details[START_LAT][key], details[START_LON][key])
            for key in keys
        }

    @cached_property
    def destination_coor(self):
        details = (
            self.dataframe.set_index([QUERY_ID, SUGGESTION_ID])
            .loc[:, [END_LAT, END_LON]]
            .to_dict()
        )
        keys = details[END_LAT].keys()
        return {
            self.format_key(key): (details[END_LAT][key], details[END_LON][key])
            for key in keys
        }

    @cached_property
    def direct_target_distance(self):
        return {
            key: distance(self.origin_coor[key], self.destination_coor[key]).m
            for key in self.keys
        }

    @cached_property
    def itinerary(self):
        return {
            self.format_key(key): json.loads(flights)
            for key, flights in self.dataframe.set_index([QUERY_ID, SUGGESTION_ID])
            .flights.to_dict()
            .items()
        }

    @cached_property
    def hops(self):
        details = self.itinerary
        return {key: len(flights) for key, flights in details.items()}

    @staticmethod
    def get_speed(row):
        return row[SPEED] / 3.6

    @staticmethod
    def get_start_time(row):
        return row[START_TIME]


train_data = Data(test=False)
test_data = Data(test=True)
