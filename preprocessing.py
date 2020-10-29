import pandas as pd
from numpy import zeros, ones
from json import dump
from datetime import date, timedelta

fullDataSet = pd.read_csv("data/owid-covid-data_2020-10-29.csv")

euroCountryCodes = fullDataSet[fullDataSet["continent"] == "Europe"].iso_code.unique()
euroDataAllColumns = fullDataSet[fullDataSet["iso_code"].isin(euroCountryCodes)]
selectedData = euroDataAllColumns[[
    "iso_code",
    "location",
    "date",
    "total_cases",
    "new_cases",
    "new_cases_smoothed",
    "population",
    "total_tests",
    "new_tests",
    "new_tests_smoothed",

]]
# Keep only data on european mainland countries with pop > 600 000

largeCountryData = selectedData[selectedData["population"] > 600000]
# List of countries to exclude due to them being islands
excludedCountryCodes = ["CYP", "GBR", "IRL", "ISL", "MLT"]
mainLandData = largeCountryData[~largeCountryData["iso_code"].isin(excludedCountryCodes)]

isoCountryCodes = [code for code in mainLandData["iso_code"].unique()]

# Padding data so all series start on the same first date
mainLandData[["new_cases", "new_cases_smoothed"]] = mainLandData[["new_cases", "new_cases_smoothed"]].fillna(0)
firstEuropeanReportedDateString = mainLandData["date"].min()

for isoCode in isoCountryCodes:
    firstDateString = mainLandData[mainLandData["iso_code"] == isoCode]["date"].min()
    if firstDateString > firstEuropeanReportedDateString:
        countryData = mainLandData[mainLandData["iso_code"] == isoCode]
        firstRowOfCountry = countryData[countryData["date"] == firstDateString]
        i = firstRowOfCountry.index[0]
        location = firstRowOfCountry.loc[i, "location"]
        population = firstRowOfCountry.loc[i, "population"]
        dates = []
        d = date.fromisoformat(firstEuropeanReportedDateString)
        while d < date.fromisoformat(firstDateString):
            dates.append(d.isoformat())
            d += timedelta(days=1)
        paddingLength = len(dates)
        paddingDict = {
            "iso_code": [isoCode] * paddingLength,
            "location": [location] * paddingLength,
            "date": dates,
            "total_cases": zeros(paddingLength),
            "new_cases": zeros(paddingLength),
            "new_cases_smoothed": zeros(paddingLength),
            "population": ones(paddingLength) * population,
            "total_tests": zeros(paddingLength),
            "new_tests": zeros(paddingLength),
            "new_tests_smoothed": zeros(paddingLength)
        }
        padding = pd.DataFrame(paddingDict)
        mainLandData = mainLandData.append(padding)

euroData = mainLandData.sort_values(by=["iso_code", "date"])
euroData.reset_index(drop=True)

# Save preprocessed data set to csv file
euroData.to_csv("data/euro_countries_padded.csv")

# Save list of ISO coutry codes to json file
with open("data/iso_country_codes.json", "w") as write_file:
    dump(isoCountryCodes, write_file)
