from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

tqdm.pandas()


@dataclass(frozen=True, eq=True)
class Settings:
    CACHE_BASEDIR = Path(".cache")

    DATA_BASEDIR = Path("data")
    TEST_DATA_DIR = DATA_BASEDIR / "test_data" / "test_data"
    TRAIN_DATA_DIR = DATA_BASEDIR / "train_data" / "train_data"

    TRAIN_DATAFRAME_PATH = DATA_BASEDIR / "train.csv"
    TEST_DATAFRAME_PATH = DATA_BASEDIR / "test.csv"
    AIRPORTS_PATH = DATA_BASEDIR / "airports.csv"

    SCHEDULE_DATA_TAG = "schedule"
    WEATHER_DATA_TAG = "weather_data"
    HISTORY_DATA_TAG = "data"

    WEATHER_TYPE_MODEL_CONFIG = {
        "n_estimators": 3000,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
    }

    WEATHER_TYPE_TRAINING_CONFIG = {
        "early_stopping_rounds": 100,
        "eval_metric": ["auc_mu", "multi_logloss"],
    }

    WEATHER_TYPE_MODEL_PATH = CACHE_BASEDIR / "type_clf.lgb"

    WEATHER_LEVEL_MODEL_CONFIG = {
        "n_estimators": 3000,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
    }

    WEATHER_LEVEL_TRAINING_CONFIG = {
        "early_stopping_rounds": 100,
        "eval_metric": ["auc_mu", "multi_logloss"],
    }

    WEATHER_LEVEL_MODEL_PATH = CACHE_BASEDIR / "level_clf.lgb"

    FEATURIZED_TRAIN = CACHE_BASEDIR / "featurized_train.csv"
    FEATURIZED_TEST = CACHE_BASEDIR / "featurized_test.csv"

    AUGMENTED_TRAIN = CACHE_BASEDIR / "augmented_train.csv"
    AUGMENTED_TEST = CACHE_BASEDIR / "augmented_test.csv"

    AUGMENTED_SCHEDULE = CACHE_BASEDIR / "augmented_schedule.csv"
    AUGMENTED_SCHEDULE_WEATHER = CACHE_BASEDIR / "augmented_schedule_weather.csv"
    CANCELLED_PROBA_BY_ID = CACHE_BASEDIR / "cancelled_proba_by_id.json"
    DEPARTURE_DELAY_BY_ID = CACHE_BASEDIR / "departure_delay_id.json"
    ARRIVAL_DELAY_BY_ID = CACHE_BASEDIR / "arrival_delay_id.json"

    AUGMENTED_HISTORY = CACHE_BASEDIR / "augmented_history.csv"
    CANCELLATION_MODEL = CACHE_BASEDIR / "cancellation_clf.lgb"
    CANCELLATION_MODEL_CONFIG = {
        "n_estimators": 3000,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
    }

    CANCELLATION_TRAINING_CONFIG = {
        "early_stopping_rounds": 100,
        "eval_metric": ["auc", "logloss"],
    }

    ARRIVAL_DELAY_MODEL = CACHE_BASEDIR / "arrival_delay_reg.lgb"
    DEPARTURE_DELAY_MODEL = CACHE_BASEDIR / "departure_delay_reg.lgb"

    DELAY_MODEL_CONFIG = {
        "n_estimators": 3000,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
    }

    DELAY_TRAINING_CONFIG = {
        "early_stopping_rounds": 400,
        "eval_metric": ["l1", "l2"],
    }

    HISTORICAL_DATA_SAMPLE_SIZE = int(5e5)

    def __init__(self):
        ensure_dir_exists = [self.CACHE_BASEDIR]

        for dir_ in ensure_dir_exists:
            dir_.mkdir(parents=True, exist_ok=True)


config = Settings()

AIRPORTS = "airports"
TRAIN = "train"
TEST = "test"
VAL = "val"

HISTORY = "history"
DATAFRAME = "dataframe"

QUERY_ID = "query_id"
SUGGESTION_ID = "suggestion_id"
SPEED = "speed"

START_LAT = "start_lat"
START_LON = "start_long"

END_LAT = "end_lat"
END_LON = "end_long"

FLIGHTS = "flights"

PART_ID = "part_id"

START_TIME = "start_time"
END_TIME = "end_time"

ARRIVAL_AIRPORT = "arrival_air"
ARRIVAL_PLANNED_TIME = "arrival_planned_time"

ARRIVAL = "arrival"
DEPARTURE = "departure"

DEPARTURE_LAT = "departure_lat"
DEPARTURE_LON = "departure_lon"
ARRIVAL_LAT = "arrival_lat"
ARRIVAL_LON = "arrival_lon"

ARRIVAL_TRANSFER_DISTANCE = "arrival_transfer_distance"
DEPARTURE_TRANSFER_DISTANCE = "departure_transfer_distance"
ARRIVAL_TRANSFER_TIME = "arrival_transfer_time"
DEPARTURE_TRANSFER_TIME = "departure_transfer_time"

ITINERARY_DISTANCE = "itinerary_distance"
ITINERARY_TIME = "itinerary_time"
REAL_ITINERARY_TIME = "real_itinerary_time"

DISTANCE_BY_AIR = "distance_by_air"

PREP_TIME = "prep_time"
TIME_IN_AIR = "time_in_air"
WASTED_TIME = "wasted_time"

FLIGHTS_STRING_LENGTH = "flights_string_length"

START_DATE = "start_date"
START_WEEKDAY = "start_weekday"
START_WEEK_OF_YEAR = "start_week_of_year"
START_HOUR = "start_hour"
START_MINUTE = "start_minute"
START_TIME_EPOCH = "start_time_epoch"
START_EARLY_NIGHT = "start_early_night"

PLANNED_END_TIME = "planned_end_time"
PLANNED_END_DATE = "planned_end_date"
PLANNED_END_WEEKDAY = "planned_end_weekday"
PLANNED_END_WEEK_OF_YEAR = "planned_end_week_of_year"
PLANNED_END_HOUR = "planned_end_hour"
PLANNED_END_MINUTE = "planned_end_minute"
PLANNED_END_TIME_EPOCH = "planned_end_time_epoch"
PLANNED_END_EARLY_NIGHT = "planned_end_early_night"

HOPS = "hops"

DIRECT_TARGET_DISTANCE = "direct_target_distance"
HYPOTHETICAL_TIME_ON_FOOT = "hypothetical_time_on_foot"
AVERAGE_FLIGHT_DISTANCE = "average_flights_distance"

AIR_OPPORTUNITY_COST = "air_opportunity_cost"

START_TYPE = "start_type"
END_TYPE = "end_type"
START_LEVEL = "start_level"
END_LEVEL = "end_level"

CANCELLED = "cancelled"
NOT_CANCELLED = "not_cancelled"

DEPARTURE_ACTUAL_TIME = "departure_actual_time"
ARRIVAL_ACTUAL_TIME = "arrival_actual_time"
DEPARTURE_DELAY = "departure_delay"
ARRIVAL_DELAY = "arrival_delay"
SCHEDULE = "schedule"
LATITUDE = "latitude"
LONGITUDE = "longtitude"
DEPARTURE_AIRPORT = "departure_air"
DEPARTURE_PLANNED_TIME = "departure_planned_time"

FLIGHT_ID = "flight_id"

WEATHER = "weather"
TYPE = "type"
LEVEL = "level"
