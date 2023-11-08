import numpy as np
import pandas as pd
import holidays
from datetime import datetime


def create_flight_code(df: pd.DataFrame) -> pd.DataFrame:
    df["flight_code"] = df["ch_code"] + df["num_code"].astype(str)
    df = df.drop(columns=["ch_code", "num_code"])

    return df


def classify_stops(df: pd.DataFrame) -> pd.DataFrame:
    df["stop"].str.split("-")
    df["stops"] = df["stop"].str.split("-").str[0]
    df["stops"] = df["stops"].replace("non", "0")
    df = df.drop(columns=["stop"])

    return df


def classify_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    early_morning = ["00", "01", "02", "03", "04", "05"]
    morning = ["06", "07", "08", "09", "10", "11"]
    afternoon = ["12", "13", "14", "15", "16", "17"]
    evening = ["18", "19", "20", "21", "22", "23"]

    df["departure_time"] = df["dep_time"].str.split(":").str[0]
    df["arrival_time"] = df["arr_time"].str.split(":").str[0]

    df["departure_time"] = df["departure_time"].replace(early_morning, "early_morning")
    df["departure_time"] = df["departure_time"].replace(morning, "morning")
    df["departure_time"] = df["departure_time"].replace(afternoon, "afternoon")
    df["departure_time"] = df["departure_time"].replace(evening, "evening")

    df["arrival_time"] = df["arrival_time"].replace(early_morning, "early_morning")
    df["arrival_time"] = df["arrival_time"].replace(morning, "morning")
    df["arrival_time"] = df["arrival_time"].replace(afternoon, "afternoon")
    df["arrival_time"] = df["arrival_time"].replace(evening, "evening")

    df = df.drop(columns=["dep_time", "arr_time"])

    return df


def classify_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["date"], dayfirst=True)
    df["dow"] = df["datetime"].dt.day_name()

    return df


def classify_holiday(df: pd.DataFrame) -> pd.DataFrame:
    indian_holidays = holidays.India()
    df["holiday"] = df["date"].apply(lambda x: indian_holidays.get(x))

    df["holiday"] = df["holiday"].fillna(0)
    df.loc[df["holiday"] != 0, "holiday"] = 1

    return df


def days_until_flight(df: pd.DataFrame) -> pd.DataFrame:
    compare_date = datetime(2022, 2, 10)
    # compare_date = datetime.today().date()
    df["days_until"] = (df["datetime"] - compare_date).dt.days

    ...


def create_flight_duration(df: pd.DataFrame) -> pd.DataFrame:
    df["duration"] = df["time_taken"].str.replace("h", ":").str.replace("m", "")
    df["duration"] = df["duration"].str.split(":")
    df["duration_hours"] = df["duration"].str[0]
    df["duration_minutes"] = df["duration"].str[1]
    df["duration"] = pd.to_numeric(
        df["duration_hours"], errors="coerce"
    ) * 60 + pd.to_numeric(df["duration_minutes"], errors="coerce")

    df = df.drop(columns=["duration_hours", "duration_minutes", "time_taken", "date"])


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = create_flight_code(df)
    df = classify_stops(df)
    df = classify_time_of_day(df)
    df = classify_day_of_week(df)
    df = classify_holiday(df)
    df = days_until_flight(df)
    df = create_flight_duration(df)

    return df


def prepare_data(dict_data: dict) -> pd.DataFrame:
    df = pd.DataFrame(dict_data)
    df = preprocess_data(df)

    return df
