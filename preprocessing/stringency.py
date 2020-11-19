import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime


data = pd.read_csv("../data/euro_countries_filled.csv")
with open("../data/iso_country_codes.json", "r") as read_file:
    euro_codes = json.load(read_file)
data = data[data["iso_code"].isin(euro_codes)]
data = data[["date", "iso_code", "stringency_index"]]
# data = data[data["date"] >= "2020-03-01"]
# data = data[data["date"] <= "2020-10-31"]

data.info()

for code in euro_codes:
    country_data = data[data["iso_code"] == code]
    plt.clf()
    plt.ylim((0, 100))
    plt.plot_date([datetime.fromisoformat(d) for d in country_data["date"]], country_data["stringency_index"], 'b-')
    plt.title(code)
    plt.savefig("stringency_plots/stringency_{:s}".format(code))