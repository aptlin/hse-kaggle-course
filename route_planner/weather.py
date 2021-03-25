from dataclasses import dataclass
from functools import cached_property

import lightgbm as lgb
import numpy as np
import pandas as pd

from route_planner.config import *
from route_planner.data import read_tag
from route_planner.model import PlannerModel


@dataclass(frozen=True, eq=True)
class Weather(PlannerModel):
    base = {
        TRAIN: {
            WEATHER: read_tag(config.TRAIN_DATA_DIR, config.WEATHER_DATA_TAG),
        },
        TEST: {
            WEATHER: read_tag(config.TEST_DATA_DIR, config.WEATHER_DATA_TAG),
        },
    }

    model_fn = {
        TYPE: config.WEATHER_TYPE_MODEL_PATH,
        LEVEL: config.WEATHER_LEVEL_MODEL_PATH,
    }

    model_config = {
        TYPE: config.WEATHER_TYPE_MODEL_CONFIG,
        LEVEL: config.WEATHER_LEVEL_MODEL_CONFIG,
    }

    @cached_property
    def collected(self):
        df = pd.concat(
            [self.base[TRAIN][WEATHER], self.base[TEST][WEATHER]],
            ignore_index=True,
        ).loc[:, [START_TIME, END_TIME, LATITUDE, LONGITUDE, TYPE, LEVEL]]

        return df

    def __init__(self):
        super().__init__()

    def predict(self, tag, df):
        if tag == TYPE:
            prediction = self.type_clf.predict(df)
        else:
            prediction = self.level_clf.predict(df)

        return np.argmax(prediction, axis=1)

    @cached_property
    def mean_time_period(self):
        weather_df = self.collected
        return (weather_df[END_TIME] - weather_df[START_TIME]).mean()

    @cached_property
    def type_clf(self, frac=0.8):
        if self.model_fn[TYPE].exists():
            model = self.load_model(self.model_fn[TYPE])
        else:
            model = lgb.LGBMClassifier(**self.model_config[TYPE])
            model = self.train_type_clf(model, frac)
        return model

    @cached_property
    def level_clf(self, frac=0.8):
        if self.model_fn[LEVEL].exists():
            model = self.load_model(self.model_fn[LEVEL])
        else:
            model = lgb.LGBMClassifier(**self.model_config[LEVEL])
            model = self.train_level_clf(model, frac)
        return model

    def train_type_clf(self, model, frac=0.8):
        df = self.collected

        X_weather = df.drop([TYPE, LEVEL], axis=1)
        weather_type = df[TYPE]

        (
            X_weather_train,
            y_type_train,
            X_weather_val,
            y_type_val,
        ) = self.split([X_weather, weather_type], frac=frac)

        model.fit(
            X_weather_train,
            y_type_train,
            eval_set=[(X_weather_val, y_type_val)],
            **config.WEATHER_TYPE_TRAINING_CONFIG
        )

        model.booster_.save_model(str(self.model_fn[TYPE].resolve()))

        return self.load_model(self.model_fn[TYPE])

    def train_level_clf(self, model, frac=0.8):
        df = self.collected

        X_weather = df.drop([TYPE, LEVEL], axis=1)
        weather_level = df[LEVEL]

        (
            X_weather_train,
            y_level_train,
            X_weather_val,
            y_level_val,
        ) = self.split([X_weather, weather_level], frac=frac)

        model.fit(
            X_weather_train,
            y_level_train,
            eval_set=[(X_weather_val, y_level_val)],
            **config.WEATHER_LEVEL_TRAINING_CONFIG
        )

        model.booster_.save_model(str(self.model_fn[LEVEL].resolve()))

        return self.load_model(self.model_fn[LEVEL])


weather = Weather()
