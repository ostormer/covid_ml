import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime


data = pd.read_csv("../data/euro_countries_filled.csv")
with open("../data/iso_country_codes.json", "r") as read_file:
    euro_codes = json.load(read_file)
data = data[data["iso_code"].isin(euro_codes)]
data = data[data["date"] <= "2020-10-06"]

for code in euro_codes:
    country_data = data[data["iso_code"] == code]
    plt.clf()
    plt.ylim((0, 100))
    dates = [datetime.fromisoformat(d) for d in country_data["date"]]
    plt.plot_date(dates, country_data["stringency_index"], 'b-')
    plt.title(code)
    plt.savefig("stringency_plots/stringency_{:s}".format(code))
