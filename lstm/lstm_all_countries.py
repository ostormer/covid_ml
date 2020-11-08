import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import date
from numpy import array
import matplotlib.pyplot as plt
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
        new_order = []
        [new_order.append(col) for col in aggregated.columns if col in leave_cols]
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


with open("../data/iso_country_codes.json", "r") as read_file:
    iso_codes = json.load(read_file)

with open("../data/train_test_codes.json", "r") as read_file:
    train_codes, test_codes = json.load(read_file)

euro_data = pd.read_csv("../data/euro_countries_filled.csv", index_col=0)

euro_data["date"] = euro_data["date"].apply(date_to_number)
euro_data = euro_data[euro_data["date"] < date_to_number("2020-10-28")]
euro_data["new_smooth_per_mill"] = euro_data.apply(lambda row: compute_new_smoothed_cases_per_million(row), axis=1)

# Scale chosen features
forecast_columns = ["new_smooth_per_mill", "latitude", "longitude", "date", "iso_code"]
scale_columns = ["new_smooth_per_mill", "latitude", "longitude", "date"]
forecast_data = euro_data[forecast_columns].copy(deep=True)
scaler = MinMaxScaler(feature_range=(0, 1))
forecast_data[scale_columns] = scaler.fit_transform(forecast_data[scale_columns])

forecast_data = series_to_in_out_pairs(forecast_data, n_in=7, n_out=1, leave_cols=["latitude", "longitude", "date", "iso_code"])
print("\n\n", forecast_data, "\n\n")


def date_scaling(d): return (date_to_number(d) - scaler.data_min_[3]) / (scaler.data_max_[3] - scaler.data_min_[3])


# All countries have data from 2019-12-31 to 2020-10-27.
# The first 7 and last 6 observations have nan values due to series_to_in_out_pairs()
# We remove those 6 last observations and all observations before march.
# Europe's habits have changed since before march 2020, newer data is more interesting
# I choose the period from 2020-03-01 to 2020-10-21, that is 235 days
forecast_data = forecast_data[forecast_data["date"] >= date_scaling("2020-03-01")]
forecast_data = forecast_data[forecast_data["date"] <= date_scaling("2020-10-21")]

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
test_x, test_y = test[:, : -1], test[:, -1]
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
print(train_x.shape, train_y.shape, test_x.shape, train_y.shape)
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
    batch_size=235,
    validation_data=(test_x, test_y),
    verbose=2,
    shuffle=False
)
# Plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


model2 = Sequential()
model2.add(Dense(50, input_shape=(train_x.shape[1], train_x.shape[2]), activation="relu"))
model2.add(Dense(20, activation="relu"))
model2.add(Dense(1, activation="relu"))
model2.compile(loss="mae", optimizer="adam")
history2 = model2.fit(
    train_x,
    train_y,
    epochs=100,
    batch_size=235,
    validation_data=(test_x, test_y),
    verbose=2,
    shuffle=False
)
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.legend()
plt.show()
