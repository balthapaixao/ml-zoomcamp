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
