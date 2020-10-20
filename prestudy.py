import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
plt.close("all")


fullDataSet = pd.read_csv("data/owid-covid-data_20_10_2020.csv")
# Select what columns we are interested in
selectedData = fullDataSet[[
    "iso_code",
    "continent",
    "location",
    "date",
    "total_cases",
    "new_cases",
    "new_cases_smoothed",
    "population",
    "total_tests",
    "new_tests",
    "new_tests_smoothed"
]]
# Keep only data on european countries
euroCountryCodes = selectedData[selectedData["continent"] == "Europe"].iso_code.unique()
euroData = selectedData[selectedData["iso_code"].isin(euroCountryCodes)]
euroData.to_csv("data/euro_countries.csv")

for isoCode in euroCountryCodes:
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
