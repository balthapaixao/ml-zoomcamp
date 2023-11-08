import pickle
import numpy as np
import pandas as pd
import preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def read_model():
    with open("../../data/models/model.bin", "rb") as f:
        model, dv, scaler = pickle.load(f)
    return model, dv, scaler


def encode_categorical_features(
    input_df: pd.DataFrame, dv: DictVectorizer
) -> pd.DataFrame:
    categorical_cols = [
        "airline",
        "from",
        "to",
        "class",
        "departure_time",
        "arrival_time",
        "dow",
        "holiday",
    ]

    df = input_df.copy()
    df[categorical_cols] = df[categorical_cols].astype(str)
    train_dict = df[categorical_cols].to_dict(orient="records")

    categorical_features_df = dv.transform(train_dict)
    df_train_categorical = pd.DataFrame(
        categorical_features_df, columns=dv.get_feature_names_out()
    )

    df_train_continuous = df.drop(columns=categorical_cols)
    df = pd.concat([df_train_continuous, df_train_categorical], axis=1)

    return df


def scale_numerical_features(
    input_df: pd.DataFrame, scaler: StandardScaler = None
) -> pd.DataFrame:
    numerical_cols = ["duration", "days_until"]

    df_numerical = input_df.copy()

    df_numerical[numerical_cols] = scaler.transform(df_numerical[numerical_cols])
    return df_numerical


def transform_input_dataframe(
    input_df: pd.DataFrame, dv: DictVectorizer, scaler: StandardScaler
) -> pd.DataFrame:
    df = encode_categorical_features(input_df, dv)
    df = scale_numerical_features(df, scaler)
    return df


def predict_price(input_dict: dict) -> float:
    model, dv, scaler = read_model()

    input_df = preprocess.prepare_data(dict_data=input_dict)
    df = transform_input_dataframe(input_df, dv, scaler)

    return model.predict(df)[0]
