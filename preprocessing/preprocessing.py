import pandas as pd
from numpy import zeros, ones, insert
from numpy.random import choice
from json import dump
from datetime import date, timedelta, datetime, time
from backports.datetime_fromisoformat import MonkeyPatch
from math import floor
from scipy.integrate import cumtrapz

MonkeyPatch.patch_fromisoformat()

fullDataSet = pd.read_csv("../data/owid-covid-data_2020-11-10.csv")

euroCountryCodes = fullDataSet[fullDataSet["continent"] == "Europe"].iso_code.unique()
euroDataAllColumns = fullDataSet[fullDataSet["iso_code"].isin(euroCountryCodes)]
selectedData = euroDataAllColumns[[
    "iso_code",
    "location",
    "date",
    "new_cases",
    "new_cases_smoothed",
    "population",
    "total_tests",  # There are holes in the data relating to number of tests
    "new_tests",  # See above ^
    "new_tests_smoothed",  # See above ^^
    "stringency_index",
]]
# Keep only data on european mainland countries with pop > 600 000

largeCountryData = selectedData[selectedData["population"] > 600000]
# List of countries to exclude due to them being islands or including negative case data
# Some are excluded due to not having stringency_index reported
excludedCountryCodes = ["CYP", "GBR", "IRL", "ISL", "MLT", "LUX", "OWID_KOS", "MNE", "MKD"]
mainLandData = largeCountryData[~largeCountryData["iso_code"].isin(excludedCountryCodes)].copy()

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
            "new_cases": zeros(paddingLength),
            "new_cases_smoothed": zeros(paddingLength),
            "population": ones(paddingLength) * population,
            "total_tests": zeros(paddingLength),
            "new_tests": zeros(paddingLength),
            "new_tests_smoothed": zeros(paddingLength),
            "stringency_index": zeros(paddingLength)  # Assuming no measures were taken before COVID cases were reported
        }
        padding = pd.DataFrame(paddingDict)
        mainLandData = mainLandData.append(padding)

euroData = mainLandData.sort_values(by=["iso_code", "date"])
euroData = euroData.reset_index(drop=True)

# Fill stringency index, assuming it was 0 in the end of 2019
firstRowsAllCountriesIndex = euroData[euroData["date"] == "2019-12-31"].index
for i in firstRowsAllCountriesIndex:
    euroData["stringency_index"].iloc[i] = 0  # Set stringency index to 0 for 2019-12-31 for all countries
euroData[["stringency_index"]] = euroData[["stringency_index"]].fillna(method="ffill")

# Fill in missing dates
newData = euroData.copy()

previousDate = date.fromisoformat("2020-12-31")
previousRow = None
for index, row in euroData.iterrows():
    thisDate = date.fromisoformat(row["date"])
    if previousRow is not None and previousRow["iso_code"] == row["iso_code"]:
        if thisDate - previousDate > timedelta(days=1):
            # There is a hole in the data that needs to be filled
            nMissingDates = (thisDate - previousDate).days
            for d in range(nMissingDates - 1):
                dateToAdd = previousDate + timedelta(days=1+d)
                newRow = row.copy(deep=True)
                newRow["date"] = dateToAdd.isoformat()
                newRow["new_cases"] = 0
                newRow["new_tests"] = 0
                newData = newData.append(newRow, ignore_index=True)

    previousDate = thisDate
    previousRow = row

sortedData = newData.copy().sort_values(by=["iso_code", "date"])
sortedData = sortedData.reset_index(drop=True)

# Add latitude and longitude
geoData = pd.read_csv("../data/countries_codes_and_coordinates.csv")
geoDict = {}
for index, row in geoData.iterrows():
    code = row["Alpha-3 code"].strip("\" ")
    geoDict[code] = {
        "latitude": row["Latitude (average)"],
        "longitude": row["Longitude (average)"]
    }

sortedData["latitude"] = sortedData.apply(lambda row: geoDict[row["iso_code"]]["latitude"].strip("\" "), axis=1)
sortedData["longitude"] = sortedData.apply(lambda row: geoDict[row["iso_code"]]["longitude"].strip("\" "), axis=1)

# Fill in missing test values using interpolation
groupedData = [group[1] for group in sortedData.groupby('iso_code')]
interpolatedData = pd.DataFrame()
for series in groupedData:
    # Fill NaNs up until first non-NaN value with 0
    series = series.interpolate('zero', fill_value=0, limit_direction='backward')
    # Then interpolate
    interpolatedData = interpolatedData.append(series.interpolate(method='polynomial', order=3))
print(interpolatedData["iso_code"].value_counts()["DEU"])
# TODO: Could just remove total tests and this following block of code
# FIXME: This following block of code removes Germany from the dataset. WHY??????
# Fill in missing total tests in Sweden and France by integrating
integratedData = interpolatedData.groupby('iso_code').filter(lambda x: x['iso_code'].iloc[0] in ['SWE', 'FRA'])
integratedData = integratedData.fillna(method='ffill').fillna(value=0)
integratedData = [group[1] for group in integratedData.groupby('iso_code')]
for i in range(len(integratedData)):
    integratedData[i]['total_tests'] = insert(cumtrapz(integratedData[i]['new_tests']), 0, 0)
    interpolatedData.update(integratedData[i])

# Drop remaining countries with no test data
interpolatedData = interpolatedData.dropna(subset=['new_tests', 'total_tests'])
print(interpolatedData["iso_code"].value_counts()["DEU"])
# Save preprocessed data set to csv file
interpolatedData.to_csv("../data/euro_countries_filled.csv")

# Save list of ISO coutry codes to json file
with open("../data/iso_country_codes.json", "w") as write_file:
    dump(isoCountryCodes, write_file)

# Split countries into training and validation data
train_codes = [code for code in choice(isoCountryCodes, floor(len(isoCountryCodes) * 0.75), replace=False)]
train_codes.sort()
validation_codes = [code for code in isoCountryCodes if code not in train_codes]
validation_codes.sort()

with open("../data/train_test_codes.json", "w") as write_file:
    dump((train_codes, validation_codes), write_file)
