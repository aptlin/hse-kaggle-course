from dataclasses import dataclass
from functools import cached_property

import lightgbm as lgb
import numpy as np

from route_planner.config import *
from route_planner.history import history
from route_planner.model import PlannerModel


@dataclass(frozen=True, eq=True)
class Delay(PlannerModel):
    model_fn = {
        ARRIVAL: config.ARRIVAL_DELAY_MODEL,
        DEPARTURE: config.DEPARTURE_DELAY_MODEL,
    }
    model_config = config.DELAY_MODEL_CONFIG
    model_training_config = config.DELAY_TRAINING_CONFIG

    def __init__(self):
        super().__init__()

    def predict(self, tag, df):
        model = self.arrival_reg if tag == ARRIVAL else self.departure_reg
        prediction = self.prepare_target(model.predict(df))
        return prediction

    @cached_property
    def departure_reg(self, frac=0.8):
        if self.model_fn[DEPARTURE].exists():
            model = self.load_model(self.model_fn[DEPARTURE])
        else:
            model = lgb.LGBMRegressor(**self.model_config)
            model = self.train_delay_reg(model, DEPARTURE, frac)
        return model

    @cached_property
    def arrival_reg(self, frac=0.8):
        if self.model_fn[ARRIVAL].exists():
            model = self.load_model(self.model_fn[ARRIVAL])
        else:
            model = lgb.LGBMRegressor(**self.model_config)
            model = self.train_delay_reg(model, ARRIVAL, frac)
        return model

    @staticmethod
    def prepare_target(offset):
        return np.clip((offset - offset.mean()) / offset.std(), -3.0, 3.0)

    def train_delay_reg(self, model, tag, frac=0.8):
        df = history.cast_categorical(history.augmented_with_weather.copy())

        X_delayed = df.drop([DEPARTURE_ACTUAL_TIME, ARRIVAL_ACTUAL_TIME], axis=1)

        actual = (
            df[ARRIVAL_ACTUAL_TIME] if tag == ARRIVAL else df[DEPARTURE_ACTUAL_TIME]
        )
        expected = (
            df[ARRIVAL_PLANNED_TIME] if tag == ARRIVAL else df[DEPARTURE_PLANNED_TIME]
        )

        y_delayed = self.prepare_target((actual - expected) / expected)

        (
            X_delayed_train,
            y_delayed_train,
            X_delayed_val,
            y_delayed_val,
        ) = self.split([X_delayed, y_delayed], frac=frac)

        model.fit(
            X_delayed_train,
            y_delayed_train,
            eval_set=[(X_delayed_val, y_delayed_val)],
            **config.DELAY_TRAINING_CONFIG
        )

        model.booster_.save_model(str(self.model_fn[tag].resolve()))

        return self.load_model(self.model_fn[tag])


delay = Delay()
