import pandas as pd
import numpy as np
from . import preprocess, predict
import pickle

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def read_data() -> pd.DataFrame:
    df_economy = pd.read_csv("../data/economy.csv")
    df_business = pd.read_csv("../data/business.csv")

    df_economy["price"] = df_economy["price"].str.replace(",", "").astype(int)
    df_business["price"] = df_business["price"].str.replace(",", "").astype(int)

    df_economy["class"] = "economy"
    df_business["class"] = "business"
    df = pd.concat([df_economy, df_business], ignore_index=True)
    return df


def train_model(df: pd.DataFrame):
    target_column = "price"
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=57)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train[target_column]
    y_test = df_test[target_column]

    df_train = df_train.drop(columns=target_column)
    df_test = df_test.drop(columns=target_column)

    df_train, dv, scaler = predict.transform_input_dataframe(df_train, None, None)
    df_test, _, _ = predict.transform_input_dataframe(df_test, dv, scaler)

    model = RandomForestRegressor(n_jobs=-1, random_state=57)
    model.fit(df_train, y_train)

    y_pred = model.predict(df_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    with open("../data/models/model.bin", "wb") as f:
        pickle.dump((model, dv, scaler), f)


def pipeline():
    df = read_data()
    df_preprocessed = preprocess.prepare_data(df)

    train_model(df_preprocessed)


if __name__ == "__main__":
    pipeline()
