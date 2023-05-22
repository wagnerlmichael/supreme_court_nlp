# Forecasting Justice: An Examination of Supreme Court Oral Arguments’ Predictive Power  

### Authors: *Núria Adell Raventós, Jonas Heim, Matt Kaufmann, Sergio Olalla Ubierna, Michael Wagner*

## Data Cleaning

This project involved multiple data cleaning/data wrangling steps. These scripts can be found in

* '/text_cleaning/flattener_with_filter.ipynb'
* '/text_cleaning/utterance_flattener.ipynb'

The flattener_with_filter.ipynb takes the convokit overall case information jsonl file (with win side, judge vote records, etc.) and unpacks the nested dictionaries into a csv file containing columns we planned to use for our models which can then be joined with the utterance data that is created and joined together the utterance_flattener.ipynb notebook.

The utterance flattener notebook output is the csv files that can be found in the 'data' directory.

## Exploratory Data Analysis

Our exploratory data analysis is found in the `/eda` directory with some key notebooks listed here:
* `/eda/summary_stats.ipynb`
* `/eda/eda.ipynb`
* `/eda/new_baselines.ipynb`

## Helper Functions

In order to make our modeling code more concise, we abstract a lot of our functions away in the `/src` directory.

## Modeling
