import pandas as pd
from matplotlib import pyplot as plt

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
    plt.plot(countryData["total_cases"])
    plt.plot(countryData["new_cases_smoothed"])
    plt.title(countryName)
    plt.legend()
    plt.savefig("plots/{:s}.png".format(isoCode))
    plt.clf()
