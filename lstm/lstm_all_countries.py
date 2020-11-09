import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def compute_new_smoothed_cases_per_million(row):
    return row["new_cases_smoothed"] / row["population"] * 1000000


def series_to_in_out_pairs(data, n_in=1, n_out=1, leave_cols=[]):
    """
    :param data: pandas dataframe to be encoded
    :param n_in: number of consecutive lag observations used for prediction
    :param n_out: number of consecutive data points for output
    :param leave_cols: variables to keep one of in input data
    :return: pandas dataframe with input-output pairs ready for supervised learning
    """
    scale_data = data[[col for col in list(data.columns) if col not in leave_cols]]
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(scale_data.shift(i))
        names += ["{:s}_(t-{:d})".format(col, i) for col in scale_data.columns]
    for i in range(0, n_out):
        cols.append(scale_data.shift(-i))
        names += ["{:s}_(t+{:d})".format(col, i) for col in scale_data.columns]
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


with open("../data/iso_country_codes.json", "r") as read_file:
    iso_codes = json.load(read_file)

with open("../data/train_test_codes.json", "r") as read_file:
    train_codes, test_codes = json.load(read_file)

euro_data = pd.read_csv("../data/euro_countries_filled.csv", index_col=0)

euro_data["date"] = euro_data["date"].apply(date_to_number)
euro_data = euro_data[euro_data["date"] < date_to_number("2020-10-28")]
euro_data["new_smooth_per_mill"] = euro_data.apply(lambda row: compute_new_smoothed_cases_per_million(row), axis=1)

# Scale chosen features
forecast_columns = ["date", "latitude", "longitude", "new_smooth_per_mill", "iso_code"]
scale_columns = ["date", "latitude", "longitude", "new_smooth_per_mill"]
forecast_data = euro_data[forecast_columns].copy(deep=True)
scaler = MinMaxScaler(feature_range=(0, 1))
forecast_data[scale_columns] = scaler.fit_transform(forecast_data[scale_columns])

forecast_data = series_to_in_out_pairs(forecast_data, n_in=7, n_out=1, leave_cols=["date", "latitude", "longitude", "iso_code"])


def date_scaling(d): return (date_to_number(d) - scaler.data_min_[0]) / (scaler.data_max_[0] - scaler.data_min_[0])


# All countries have data from 2019-12-31 to 2020-10-27.
# The first 7 have nan values due to series_to_in_out_pairs()
# We remove all observations before march.
# Europe's habits have changed since before march 2020, newer data is more interesting
# I choose the period from 2020-03-01 to 2020-10-27, that is 241 days
start_date, end_date = "2020-03-01", "2020-10-27"
forecast_data = forecast_data[forecast_data["date"] >= date_scaling(start_date)]
forecast_data = forecast_data[forecast_data["date"] <= date_scaling(end_date)]

train_df = forecast_data[forecast_data["iso_code"].isin(train_codes)]
test_df = forecast_data[forecast_data["iso_code"].isin(test_codes)]
# After split into training and test data we no longer need iso_code
train_df = train_df.drop(columns=["iso_code"])
test_df = test_df.drop(columns=["iso_code"])
train = train_df.values
test = test_df.values


n_lag_days = 7
n_daily_features = 1
n_single_features = 3
n_obs = n_lag_days * n_daily_features + n_single_features  # 10 in this case

train_x, train_y = train[:, : n_obs], train[:, -1]
test_x, test_y = test[:, : n_obs], test[:, -1]
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
# Define a model
model = Sequential()
model.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer="adam")
# Fit network
history = model.fit(
    train_x,
    train_y,
    epochs=100,
    batch_size=date_to_number(end_date) - date_to_number(start_date) + 1,
    validation_data=(test_x, test_y),
    verbose=2,
    shuffle=False
)
# Plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig("training_loss.png")

# Evaluate model
scaled_prediction = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
scaled_prediction = concatenate((test_x[:, 0:3], scaled_prediction), axis=1)
prediction = scaler.inverse_transform(scaled_prediction)
y_hat = prediction[:, -1]

actual_test = test_y.reshape((len(test_y), 1))
actual_test = concatenate((test_x[:, 0:3], actual_test), axis=1)
actual_test = scaler.inverse_transform(actual_test)
y = actual_test[:, -1]

mae = mean_absolute_error(y_hat, y)
print("MAE: ", mae)
print("\n")

# Plot 7 steps ahead forecast for all countries.
n_pred_steps = 7
for code in iso_codes:
    country_data = forecast_data[forecast_data["iso_code"] == code]
    country_data = country_data.drop(columns=["iso_code"])

    last_entry = country_data[country_data["date"] == date_scaling(end_date)]
    last_entry = last_entry.values[:, : n_obs]
    last_entry = last_entry.reshape(1, 1, test_x.shape[1])
    prediction = model.predict(last_entry)
    first_pred_date = (date.fromisoformat(end_date)+timedelta(days=1))
    predicted_values = concatenate((
        [date_scaling(first_pred_date.isoformat())],
        last_entry[0, 0, 1:3],
        last_entry[0, 0, 4:],
        prediction[0]))
    predicted_values = predicted_values.reshape((1, 1, test_x.shape[1]))

    for i in range(1, n_pred_steps):  # 6 more days
        last_entry = predicted_values[-1:, :, :]
        new_prediction = model.predict(last_entry)
        new_date = (first_pred_date + timedelta(days=i)).isoformat()
        new_last_row = concatenate((
            [date_scaling(new_date)],
            predicted_values[-1, 0, 1:3],
            predicted_values[-1, 0, 4:],
            new_prediction[0]))
        new_last_row = new_last_row.reshape((1, 1, test_x.shape[1]))
        predicted_values = concatenate((predicted_values, new_last_row))

    predicted_values = predicted_values.reshape((predicted_values.shape[0], predicted_values.shape[2]))
    predicted_values = concatenate((predicted_values[:, 0:3], predicted_values[:, -1:]), axis=1)
    predicted_values = scaler.inverse_transform(predicted_values)

    pred_x = predicted_values[:, 0].reshape(n_pred_steps)
    pred_y = predicted_values[:, 3].reshape(n_pred_steps)
    pred_dates = [number_to_datetime(round(x)) for x in pred_x]

    recent_history = country_data[["date", "latitude", "longitude", "new_smooth_per_mill_(t+0)"]].values
    recent_history = scaler.inverse_transform(recent_history)
    recent_x = recent_history[-21:, 0].reshape(-14)
    recent_y = recent_history[-21:, 3].reshape(-14)
    recent_dates = [number_to_datetime(round(x)) for x in recent_x]

    plt.plot_date(recent_dates, recent_y, "r-")
    plt.plot_date(pred_dates, pred_y, "b-")
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.title(code)
    plt.xlabel("Time")  # FIXME: Find out why this doesn't render
    plt.ylabel("Cases")
    plt.grid(True, "major", "y", color="grey", linewidth=0.2)
    plt.legend(["New cases per 1 000 000 population 7-days smoothed", "7 days ahead forecast"])
    plt.savefig("plots/{:s}.png".format(code))
    plt.clf()
