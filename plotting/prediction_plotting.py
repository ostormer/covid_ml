import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pandas as pd
from datetime import datetime, timedelta

hw_pred = pd.read_csv("../predictions/hw_predictions.csv")  # File has moved (?)
lstm_pred = pd.read_csv("../predictions/lstm_predictions.csv")
rf_pred = pd.read_csv("../predictions/rf_predictions.csv")
euro_data = pd.read_csv("../data/euro_countries_filled.csv")
euro_data = euro_data[euro_data["date"] >= "2020-10-17"]
euro_data = euro_data[euro_data["date"] <= "2020-11-07"]

nor_data = euro_data[euro_data["iso_code"] == "NOR"]
recent_dateTimes = [datetime.fromisoformat(d) for d in nor_data["date"].values]
recent_dates = date2num(recent_dateTimes)
pred_dates = recent_dates[-7:]

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(1920/150, 1080/150)

for i, isoCode in enumerate(["NOR", "SWE", "FIN", "BEL"]):
    hw_y = hw_pred["Holt-Winter Additive + Damped {:s}".format(isoCode)]
    lstm_y = lstm_pred["lstm_{:s}".format(isoCode)]
    rf_y = rf_pred["rf_{:s}".format(isoCode)]
    country_data = euro_data[euro_data["iso_code"] == isoCode]
    recent_y = country_data["new_cases_smoothed_per_million"]
    axs[i // 2, i % 2].plot_date(pred_dates, hw_y, "r-")
    axs[i // 2, i % 2].plot_date(pred_dates, lstm_y, "m-")
    axs[i // 2, i % 2].plot_date(pred_dates, rf_y, "g-")
    axs[i // 2, i % 2].plot_date(recent_dates, recent_y, "-")
    axs[i // 2, i % 2].set_title(isoCode)

for ax in axs.flat:
    ax.set(xlabel="Date", ylabel="Cases per million")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

fig.tight_layout(pad=3)
fig.suptitle("Prediction of new cases (7-days smoothed) for sample countries")
fig.savefig("../plots/sample_pred.png")
