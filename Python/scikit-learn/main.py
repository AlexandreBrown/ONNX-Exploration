import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
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
    "DIVERTED",
    "CANCELLED",
    "DISTANCE",
]

TARGETS_COLUMNS = ["ARR_TIME"]


def load_data():
    data_set = pd.read_csv("../../Datasets/Jan_2020_ontime.csv")

    data_set.replace("", float("NaN"), inplace=True)
    data_set.dropna(subset=FEATURES_COLUMNS + TARGETS_COLUMNS, inplace=True)

    for column in data_set.columns:
        if data_set[column].dtype == "object":
            encoder = LabelEncoder()
            data_set[column] = encoder.fit_transform(data_set[column])

    print("-------------------------------------")
    data_set.info()

    _, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(data_set.corr(), annot=True, linewidths=.5, ax=ax)
    plt.show()

    x = data_set[FEATURES_COLUMNS].values

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
    print("Random Forest")

    estimators_score = {}
    models = {}

    minimum_number_of_estimators = 20
    maximum_number_of_estimators = 70
    number_of_estimators_step_size = 5

    for number_of_estimators in range(minimum_number_of_estimators,
                                      maximum_number_of_estimators + 1,
                                      number_of_estimators_step_size):
        print("Training Random Forest model with {} estimators ...".format(number_of_estimators))
        model = RandomForestRegressor(n_estimators=number_of_estimators, criterion="mse")
        model.fit(X_trainset, y_trainset)
        r2_score = model.score(X_testset, y_testset)
        estimators_score[number_of_estimators] = r2_score
        print("R2 Score ", r2_score)
        print("Max Depth ", model.estimators_[0].tree_.max_depth)

    best_number_of_estimators = max(estimators_score, key=estimators_score.get)
    print("Estimators results ", estimators_score)
    print("best number of estimators (depth unlimited)", best_number_of_estimators)
    print("Random Forest Best R2 score : ", estimators_score[best_number_of_estimators])

    minimum_depth = 1
    maximum_depth = 30
    depth_step_size = 1

    depth_score = {}

    for max_depth in range(minimum_depth, maximum_depth + 1, depth_step_size):
        print("Training Random Forest model with maximum depth of {} ...".format(max_depth))
        model = RandomForestRegressor(n_estimators=50, criterion="mse", max_depth=max_depth)
        model.fit(X_trainset, y_trainset)
        r2_score = model.score(X_testset, y_testset)
        print("R2 Score ", r2_score)
        depth_score[max_depth] = r2_score
        models[max_depth] = model

    estimators_number_of_tests = ((
                                              maximum_number_of_estimators - minimum_number_of_estimators) / number_of_estimators_step_size) + 1
    estimators_x = np.linspace(minimum_number_of_estimators, maximum_number_of_estimators,
                               int(estimators_number_of_tests))
    estimators_results = estimators_score.values()
    plt.plot(estimators_x, estimators_results, "r")
    for estimator_x in estimators_x:
        plt.axvline(x=estimator_x)
    plt.xticks(np.arange(min(estimators_x), max(estimators_x) + 1, number_of_estimators_step_size))
    plt.show()

    depths_number_of_tests = ((maximum_depth - minimum_depth + 1) / depth_step_size)
    depths_x = np.linspace(minimum_depth, maximum_depth, int(depths_number_of_tests))
    depths_results = depth_score.values()
    plt.plot(depths_x, depths_results, "b")
    for depth_x in depths_x:
        plt.axvline(x=depth_x)
    plt.xticks(np.arange(min(depths_x), max(depths_x) + 1, depth_step_size))
    plt.show()

    best_number_of_depth = max(depth_score, key=depth_score.get)
    print("Best number of depth : ", best_number_of_depth)

    return models[best_number_of_depth]


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
