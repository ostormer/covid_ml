import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import json
plt.close("all")


euroData = pd.read_csv("data/euro_countries_padded.csv")
with open("data/iso_country_codes.json", "r") as read_file:
    countryCodes = json.load(read_file)

for isoCode in countryCodes:
    countryData = euroData[euroData["iso_code"] == isoCode]
    countryName = (countryData.location.unique()[0])
    dateTimes = [datetime.strptime(date, "%Y-%m-%d") for date in countryData["date"]]
    dates = date2num(dateTimes)

    plt.plot_date(dates, countryData["total_cases"], "-")
    plt.plot_date(dates, countryData["new_cases_smoothed"]*10, "-")
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.title(countryName)
    plt.xlabel("Time")  # FIXME: Find out why this doesn't render
    plt.ylabel("Cases")
    plt.grid(True, "major", "y", color="grey", linewidth=0.2)
    plt.legend(["Total cases", "New cases * 10 (7 days smoothed)"])
    plt.savefig("plots/{:s}.png".format(isoCode))
    plt.clf()
