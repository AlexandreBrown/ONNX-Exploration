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
from sklearn.svm import SVR


def load_data():
    data_set = pd.read_csv("../../Datasets/Jan_2020_ontime.csv")

    features_columns = [
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

    targets_columns = ["ARR_TIME"]

    data_set.replace("", float("NaN"), inplace=True)
    data_set.dropna(subset=features_columns + targets_columns, inplace=True)

    x = data_set[features_columns].values

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

    y = np.ravel(data_set[targets_columns].values)

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


def test_random_forest(X_trainset, X_testset, y_trainset, y_testset):
    model = RandomForestRegressor(criterion="mse")
    model.fit(X_trainset, y_trainset)
    predictions = model.predict(X_testset)
    mse_score = mean_squared_error(y_testset, predictions)
    r2_score = model.score(X_testset, y_testset)
    print("Random Forest score (MSE) :", mse_score)
    print("Random Forest score (R2) :", r2_score)


def test_linear_regression(X_trainset, X_testset, y_trainset, y_testset):
    model = LinearRegression()
    model.fit(X_trainset, y_trainset)
    predictions = model.predict(X_testset)
    mse_score = mean_squared_error(y_testset, predictions)
    r2_score = model.score(X_testset, y_testset)
    print("Linear Regression score (MSE) :", mse_score)
    print("Linear Regression score (R2) :", r2_score)


def test_svr(X_trainset, X_testset, y_trainset, y_testset):
    model = SVR()
    model.fit(X_trainset, y_trainset)
    predictions = model.predict(X_testset)
    mse_score = mean_squared_error(y_testset, predictions)
    r2_score = model.score(X_testset, y_testset)
    print("SVR score (MSE) :", mse_score)
    print("SVR score (R2) :", r2_score)


if __name__ == '__main__':
    X_trainset, X_testset, y_trainset, y_testset = load_data()

    print("Results: ")
    test_decision_tree(X_trainset, X_testset, y_trainset, y_testset)
    test_random_forest(X_trainset, X_testset, y_trainset, y_testset)
    test_linear_regression(X_trainset, X_testset, y_trainset, y_testset)
    test_svr(X_trainset, X_testset, y_trainset, y_testset)
    print("-------------------------------------")
