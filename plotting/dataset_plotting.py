import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import json
plt.close("all")


euroData = pd.read_csv("../data/euro_countries_filled.csv")
with open("../data/iso_country_codes.json", "r") as read_file:
    countryCodes = json.load(read_file)

euroData = euroData[euroData["date"] <= "2020-11-06"]

for isoCode in countryCodes:
    countryData = euroData[euroData["iso_code"] == isoCode]
    countryName = (countryData.location.unique()[0])
    dateTimes = [datetime.strptime(date, "%Y-%m-%d") for date in countryData["date"]]
    dates = date2num(dateTimes)

    plt.plot_date(dates, countryData["new_cases_smoothed_per_million"], "-")
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.title(countryName)
    plt.xlabel("Time")  # Does not render because it is pushed below the frame by the dates
    plt.ylabel("Cases")
    plt.grid(True, "major", "y", color="grey", linewidth=0.2)
    plt.legend(["New cases per million (7-day smoothed)"])
    plt.savefig("plots/cases_smoothed/{:s}.png".format(isoCode))
    plt.clf()

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(1920/150, 1080/150)

for i, isoCode in enumerate(["NOR", "SWE", "FIN", "BEL"]):
    countryData = euroData[euroData["iso_code"] == isoCode]
    countryName = (countryData.location.unique()[0])
    dateTimes = [datetime.strptime(date, "%Y-%m-%d") for date in countryData["date"]]
    dates = date2num(dateTimes)
    axs[i // 2, i % 2].plot_date(dates, countryData["new_cases_smoothed_per_million"], "-")
    axs[i // 2, i % 2].set_title(countryName)


for ax in axs.flat:
    ax.set(xlabel="Date", ylabel="Cases per million")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

fig.tight_layout(pad=3)
fig.suptitle("New cases (7-days smoothed) for sample countries")
fig.savefig("../plots/sample_plots.png")
