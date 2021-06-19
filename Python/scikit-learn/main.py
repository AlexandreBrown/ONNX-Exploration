import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

FEATURES_COLUMNS = [
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_AIRLINE_ID",
    "OP_CARRIER",
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID",
    "ORIGIN_AIRPORT_SEQ_ID",
    "ORIGIN",
    "DEST_AIRPORT_ID",
    "DEST_AIRPORT_SEQ_ID",
    "DEST",
    "DEP_TIME",
    "DEP_DEL15",
    "DEP_TIME_BLK",
    "DISTANCE",
]

TARGETS_COLUMNS = ["ARR_TIME"]


def load_data():
    data_set = pd.read_csv("../../Datasets/Jan_2020_ontime.csv")

    data_set.replace("", float("NaN"), inplace=True)
    data_set.dropna(subset=FEATURES_COLUMNS + TARGETS_COLUMNS, inplace=True)

    x = data_set[FEATURES_COLUMNS].values

    op_unique_labelEncoder = preprocessing.LabelEncoder()
    x[:, 2] = op_unique_labelEncoder.fit_transform(x[:, 2])
    op_carrier_labelEncoder = preprocessing.LabelEncoder()
    x[:, 4] = op_carrier_labelEncoder.fit_transform(x[:, 4])
    tail_num_labelEncoder = preprocessing.LabelEncoder()
    x[:, 5] = tail_num_labelEncoder.fit_transform(x[:, 5])
    origin_labelEncoder = preprocessing.LabelEncoder()
    x[:, 9] = origin_labelEncoder.fit_transform(x[:, 9])
    dest_labelEncoder = preprocessing.LabelEncoder()
    x[:, 12] = dest_labelEncoder.fit_transform(x[:, 12])
    dep_time_labelEncoder = preprocessing.LabelEncoder()
    x[:, 15] = dep_time_labelEncoder.fit_transform(x[:, 15])

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    y = np.ravel(data_set[TARGETS_COLUMNS].values)

    print("-------------------------------------")
    print("X shape : ", x.shape)
    print("X samples : ", x[:5])
    print("-------------------------------------")
    print("y shape : ", y.shape)
    print("y samples : ", y[:5])
    print("-------------------------------------")

    X_trainset, X_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=42)

    return X_trainset, X_testset, y_trainset, y_testset


def test_decision_tree(X_trainset, X_testset, y_trainset, y_testset):
    model = DecisionTreeRegressor(criterion="mse")
    model.fit(X_trainset, y_trainset)
    predictions = model.predict(X_testset)
    mse_score = mean_squared_error(y_testset, predictions)
    r2_score = model.score(X_testset, y_testset)
    print("Decision Tree score (MSE) :", mse_score)
    print("Decision Tree score (R2) :", r2_score)
    return model


def test_random_forest(X_trainset, X_testset, y_trainset, y_testset):
    model = RandomForestRegressor(criterion="mse")
    model.fit(X_trainset, y_trainset)
    predictions = model.predict(X_testset)
    mse_score = mean_squared_error(y_testset, predictions)
    r2_score = model.score(X_testset, y_testset)
    print("Random Forest score (MSE) :", mse_score)
    print("Random Forest score (R2) :", r2_score)
    return model


def test_linear_regression(X_trainset, X_testset, y_trainset, y_testset):
    model = LinearRegression()
    model.fit(X_trainset, y_trainset)
    predictions = model.predict(X_testset)
    mse_score = mean_squared_error(y_testset, predictions)
    r2_score = model.score(X_testset, y_testset)
    print("Linear Regression score (MSE) :", mse_score)
    print("Linear Regression score (R2) :", r2_score)
    return model


def save_to_onnx(model):
    features_type = [('float_input', FloatTensorType([None, len(FEATURES_COLUMNS)]))]
    onx_model = convert_sklearn(model, initial_types=features_type)

    with open("random_forest_model.onnx", "wb") as f:
        f.write(onx_model.SerializeToString())


if __name__ == '__main__':
    X_trainset, X_testset, y_trainset, y_testset = load_data()

    print("Results: ")
    test_decision_tree(X_trainset, X_testset, y_trainset, y_testset)
    rf_model = test_random_forest(X_trainset, X_testset, y_trainset, y_testset)
    test_linear_regression(X_trainset, X_testset, y_trainset, y_testset)
    print("-------------------------------------")
    print("Saving model to ONNX...")
    save_to_onnx(rf_model)
    print("Model saved successfully!")
