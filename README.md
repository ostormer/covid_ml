# COVID-19 Machine Learning Project

## Description
Machine Learning project forecasting development of COVID-19 i European countries.
The project was carried out as part of the course Machine Learning (TDT4173) at NTNU during fall 2020.

As part of the project we have created an LSTM deep learning model, an ensemble Random Forest model, and a statistical exponential smoothing model to use as a baseline. All code is written i python 3, either in .py files or Jupyter notebooks.


## Report and podcast
A project report was written as part of the project, and can be found here: [`report_link`](report_link.pdf). <!-- TODO: upload report and update link! -->

We also recorded a podcast, which can be found in [`podcast/ML-Podcast.m4a`](podcast/ML-Podcast.m4a)

## Installation guide
To pip install all required python packages to run the project, open a terminal window, optionally set up a v-env, and run the command:

`pip install requirements.txt`

### Preprocessing

Running the file `preprocessing.py` preprocesses the raw data from the file `data/owid-cowid-data_2020-11-22.csv` and saves the preprocessed data into the file `data/euro_countries_filled.csv`. The preprocessed data in this file is used by the models for training and evaluation.

### Running the models

- `Holt-Winters.ipynb` Trains and evaluates the Holt-Winters' model.
    - As this is a jupyter notebook, it must be run by booting up the jupyter server and running all blocks of code.
    - The output of the notebook can be seen without setting anything up by opening it on [github.com in a browser](https://github.com/ostormer/covid_ml/blob/main/Holt-Winters.ipynb)
    
- `lstm.py` Trains and evaluates the LSTM model.

- `SOMETHING SOMETHING RF .ipynb` Trains and evaluates the Random Forest model.
    - As this is a jupyter notebook, it must be run by booting up the jupyter server and running all blocks of code.
    - The output of the notebook can be seen without setting anything up by opening it on [github.com in a browser](SOMETHING_SOMETHING_RF_GITHUB_LINK)

All these three models save their predictions on the test set and sample countries as csv-files in the folder `predictions/`

### Plotting the results

The plots shown in the report are generated by running the files in the `plotting` directory.
- `plotting/dataset_plotting.py` generates the plot `plots/sample_plots.png` (Figure 1 in the report), as well as a plot of `new_cases_smoothed_per_million` for each country. These country-specific plots are saved in the gitignored folder `plots/cases_smoothed/`.
- `plotting/prediction_plotting.py` generates the plot `plots/sample_pred.png` (Figure 2 in the report) by using the test set predictions saved in `predictions/`.
