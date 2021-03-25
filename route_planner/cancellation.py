from dataclasses import dataclass
from functools import cached_property

import lightgbm as lgb
import numpy as np

from route_planner.config import *
from route_planner.history import history
from route_planner.model import PlannerModel


@dataclass(frozen=True, eq=True)
class Cancellation(PlannerModel):
    model_fn = config.CANCELLATION_MODEL
    model_config = config.CANCELLATION_MODEL_CONFIG
    model_training_config = config.CANCELLATION_TRAINING_CONFIG

    def __init__(self):
        super().__init__()

    def predict(self, df):
        prediction = self.clf.predict(df)
        return prediction

    @cached_property
    def clf(self, frac=0.8):
        if self.model_fn.exists():
            model = self.load_model(self.model_fn)
        else:
            model = lgb.LGBMClassifier(**self.model_config)
            model = self.train_cancellation_clf(model, frac)
        return model

    def train_cancellation_clf(self, model, frac=0.8):
        df = history.cast_categorical(history.augmented_with_weather)

        X_cancelled = df.drop(
            [CANCELLED, DEPARTURE_ACTUAL_TIME, ARRIVAL_ACTUAL_TIME], axis=1
        )

        y_cancelled = df[CANCELLED]

        (
            X_cancelled_train,
            y_cancelled_train,
            X_cancelled_val,
            y_cancelled_val,
        ) = self.split([X_cancelled, y_cancelled], frac=frac)

        model.fit(
            X_cancelled_train,
            y_cancelled_train,
            eval_set=[(X_cancelled_val, y_cancelled_val)],
            **config.CANCELLATION_TRAINING_CONFIG
        )

        model.booster_.save_model(str(self.model_fn.resolve()))

        return self.load_model(self.model_fn)


cancellation = Cancellation()
