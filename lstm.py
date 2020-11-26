import pandas as pd
import json
from math import sqrt
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from tensorflow import random as tf_random
from numpy.random import seed as np_set_seed


tf_random.set_seed(0)  # Set the seeds for reproducible results
np_set_seed(0)


def series_to_in_out_pairs(data, n_in=1, n_out=1, leave_cols=None):
    """
    :param data: pandas dataframe to be encoded
    :param n_in: number of consecutive lag observations used for prediction
    :param n_out: number of consecutive data points for output
    :param leave_cols: variables to keep one of in input data
    :return: pandas dataframe with input-output pairs ready for supervised learning
    """
    if leave_cols is None:
        leave_cols = []
    scale_data = data[[col for col in list(data.columns) if col not in leave_cols]]
    cols, names = list(), list()
    for lag in range(n_in, 0, -1):
        cols.append(scale_data.shift(lag))
        names += ["{:s}_(t-{:d})".format(col, lag) for col in scale_data.columns]
    for lag in range(0, n_out):
        cols.append(scale_data.shift(-lag))
        names += ["{:s}_(t+{:d})".format(col, lag) for col in scale_data.columns]
    aggregated = pd.concat(cols, axis=1)
    aggregated.columns = names
    if len(leave_cols) > 0:
        aggregated[leave_cols] = data[leave_cols]
        # reorder so leave_cols are first
        new_order = [col for col in leave_cols]
        [new_order.append(col) for col in aggregated.columns if col not in leave_cols]
        aggregated = aggregated[new_order]
    return aggregated


def date_to_number(d):
    """
    Change date to show elapsed days since the time series' beginning at 2019-12-31
    :param d: string iso_format "YYYY-MM-DD"
    :return: int days since 2019-12-31
    """
    return (date.fromisoformat(d) - date.fromisoformat("2019-12-31")).days


def number_to_datetime(n):
    """
    Revert date_to_number(d)
    :param n: int days since 2019-12-31
    :return: string iso_format "YYYY-MM-DD"
    """
    return datetime.fromisoformat("2019-12-31") + timedelta(days=n)


with open("data/iso_country_codes.json", "r") as read_file:
    iso_codes = json.load(read_file)

with open("data/train_test_codes.json", "r") as read_file:
    train_codes, test_codes = json.load(read_file)

start_date, end_date = "2020-03-01", "2020-10-31"
n_lag_days = 7
n_daily_features = 1
n_single_features = 4
n_obs = n_lag_days * n_daily_features + n_single_features  # 11 in this case

raw_euro_data = pd.read_csv("data/euro_countries_filled.csv", index_col=0)
euro_data = raw_euro_data.copy(deep=True)
euro_data["date"] = euro_data["date"].apply(date_to_number)
euro_data = euro_data[euro_data["date"] <= date_to_number(end_date)]

# Scale chosen features
forecast_columns = ["date", "latitude", "longitude", "stringency_index",
                    "new_cases_smoothed_per_million", "iso_code"]
scale_columns = ["date", "latitude", "longitude", "stringency_index",
                 "new_cases_smoothed_per_million"]
forecast_data = euro_data[forecast_columns].copy(deep=True)
scaler = MinMaxScaler(feature_range=(0, 1))
forecast_data[scale_columns] = scaler.fit_transform(forecast_data[scale_columns])

forecast_data = series_to_in_out_pairs(forecast_data, n_in=n_lag_days, n_out=1,
                                       leave_cols=["date", "latitude", "longitude",
                                                   "stringency_index", "iso_code"])


def date_scaling(d):
    return (date_to_number(d) - scaler.data_min_[0]) / (scaler.data_max_[0] - scaler.data_min_[0])


def invert_date_scaling(s):
    return number_to_datetime(s * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0])


# We remove all observations before march.
# Europe's habits have changed since before march 2020, newer data is more interesting
# I choose the period from 2020-03-01 to 2020-10-27, that is 241 days
forecast_data = forecast_data[forecast_data["date"] >= date_scaling(start_date)]
forecast_data = forecast_data[forecast_data["date"] <= date_scaling(end_date)]

train_df = forecast_data[forecast_data["iso_code"].isin(train_codes)]
test_df = forecast_data[forecast_data["iso_code"].isin(test_codes)]
# After split into training and validation data we no longer need iso_code
train_df = train_df.drop(columns=["iso_code"])
test_df = test_df.drop(columns=["iso_code"])
train = train_df.values
test = test_df.values

train_x, train_y = train[:, : n_obs], train[:, -1]
test_x, test_y = test[:, : n_obs], test[:, -1]
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

# Define a model
model = Sequential()
model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
# Fit network
history = model.fit(
    train_x,
    train_y,
    epochs=400,
    # 1 Batch for each country so weights are updated after each country is processed
    batch_size=(date_to_number(end_date) - date_to_number(start_date) + 1),
    validation_data=(test_x, test_y),
    verbose=2,
    shuffle=False
)
# Plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(["Training loss", "Validation loss"])
plt.savefig("plots/lstm/training_loss.png")

# Evaluate model
scaled_prediction = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
scaled_prediction = concatenate((test_x[:, 0:n_single_features], scaled_prediction), axis=1)
prediction = scaler.inverse_transform(scaled_prediction)
y_hat = prediction[:, -1]

actual_test = test_y.reshape((len(test_y), 1))
actual_test = concatenate((test_x[:, 0:n_single_features], actual_test), axis=1)
actual_test = scaler.inverse_transform(actual_test)
y = actual_test[:, -1]


first_pred_date = (date.fromisoformat(end_date) + timedelta(days=1))
n_pred_steps = 7
pred_dates = [(first_pred_date + timedelta(days=i)).isoformat() for i in range(n_pred_steps)]

lstm_predictions = pd.DataFrame()
lstm_predictions["date"] = pred_dates
country_mse_list = []
# Plot 7 steps ahead forecast for chosen countries and compute MSE of recursive prediction on test set

plt.clf()
for code in iso_codes:
    country_data = forecast_data[forecast_data["iso_code"] == code]
    country_data = country_data.drop(columns=["iso_code"])
    last_entry = country_data.tail(1)
    last_entry = last_entry.values[:, : n_obs]
    last_entry = last_entry.reshape(1, 1, n_obs)
    prediction = model.predict(last_entry)
    predicted_values = concatenate((
        [date_scaling(first_pred_date.isoformat())],
        last_entry[0, 0, 1:n_single_features],
        last_entry[0, 0, n_single_features + 1:],
        prediction[0]))
    predicted_values = predicted_values.reshape((1, 1, n_obs))

    for i in range(1, n_pred_steps):  # 6 more days
        last_entry = predicted_values[-1:, :, :]
        new_prediction = model.predict(last_entry)
        new_date = (first_pred_date + timedelta(days=i)).isoformat()
        new_last_row = concatenate((
            [date_scaling(new_date)],
            predicted_values[-1, 0, 1:n_single_features - 1],
            predicted_values[-1, 0, n_single_features:],
            new_prediction[0]))
        new_last_row = new_last_row.reshape((1, 1, test_x.shape[1]))
        predicted_values = concatenate((predicted_values, new_last_row))

    predicted_values = predicted_values.reshape((predicted_values.shape[0], predicted_values.shape[2]))
    predicted_values = concatenate((predicted_values[:, 0:n_single_features], predicted_values[:, -1:]), axis=1)
    predicted_values = scaler.inverse_transform(predicted_values)

    pred_x = predicted_values[:, 0].reshape(n_pred_steps)
    pred_y = predicted_values[:, n_single_features].reshape(n_pred_steps)
    pred_dates = [number_to_datetime(round(x)) for x in pred_x]

    country_test = raw_euro_data[raw_euro_data["iso_code"] == code]
    country_test = country_test[country_test["date"] >= "2020-11-01"]
    country_test = country_test[country_test["date"] <= "2020-11-07"]
    test_y = country_test["new_cases_smoothed_per_million"].values
    country_mse = mean_squared_error(test_y, pred_y)
    country_mse_list.append(country_mse)

    # Demo countries
    if code in ["BEL", "FIN", "NOR", "SWE"]:
        recent_history = raw_euro_data[raw_euro_data["iso_code"] == code]
        recent_history = recent_history[recent_history["date"] >= "2020-10-17"]
        recent_history = recent_history[recent_history["date"] <= "2020-11-07"]
        recent_y = recent_history["new_cases_smoothed_per_million"].values
        recent_dates = [datetime.fromisoformat(d) for d in recent_history[["date"]].values.flatten()]
        country_RMSE = sqrt(country_mse)
        print("{:s} 7-days-ahead RMSE: {:.2f}".format(code, country_RMSE))

        plt.plot_date(recent_dates, recent_y, "r.-")
        plt.plot_date(pred_dates, pred_y, "b.-")
        plt.xticks(rotation=20, horizontalalignment="right")
        plt.title(code)
        plt.xlabel("Time")  # Doesn't render because it is pushed below the picture by the dates
        plt.ylabel("Cases")
        plt.grid(True, "major", "y", color="grey", linewidth=0.2)
        plt.legend(["New cases (7-days smoothed)", "Recursive 7-day forecast"])
        plt.savefig("plots/lstm/{:s}_pred.png".format(code))
        plt.clf()

        lstm_predictions["lstm_{:s}".format(code)] = pred_y

total_rmse = sqrt(sum(country_mse_list) / len(country_mse_list))
print("RMSE of entire test set: {:.2f}".format(total_rmse))

lstm_predictions.to_csv("predictions/lstm_predictions.csv")
