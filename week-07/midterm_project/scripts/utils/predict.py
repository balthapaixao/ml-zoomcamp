import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def read_model():
    with open("../../data/models/model.bin", "rb") as f:
        model, dv, scaler = pickle.load(f)
    return model, dv, scaler


def transform_input(df, dv, scaler):
    df = df.copy()
    df = df[
        [
            "flight_code",
            "stops",
            "time_of_day",
            "day_of_week",
            "holiday",
            "days_until_flight",
            "flight_duration",
        ]
    ]
    df_dict = df.to_dict(orient="records")
    X = dv.transform(df_dict)
    X = scaler.transform(X)
    return X
