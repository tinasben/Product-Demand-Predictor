# Product Demand Forecast Final Project

# Store Item Demand Forecasting

**Description**

We are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

What's the best way to deal with seasonality? Should stores be modeled separately, or can you pool them together? Does deep learning work better than ARIMA? Can either beat xgboost?

**Evaluation**

Submissions are evaluated on [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) between forecasts and actual values. We define [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) = 0 when the actual and predicted values are both 0.

**Variables:**
- date
- store
- item
- sales

## Running the App
-Ensure to delete input_data.csv on each run to avoid using old data in new run.
-Run the following commands in the terminal and not in VSCode terminal.
-Create a virtual environment:
```bash
virtualenv flask
```
-Activate the virtual environment:
```bash
flask\Scripts\activate
```
-Install relevant libraries using:
```bash
pip install -r requirements.txt
```
-Simply run and ensure the interpreter selected in VSCode is the flask venv:
```bash
app.py
```

## About the Stack

We started the full stack application for you. It is designed with some key functional areas:

### lighgbmmodel

The [model](./lightgbm_model.ipynb) notebook contains a our implementation of the lightgbm model to perform prediction of sales when given date range, store and item values. Our workflow involved:
1. `Exploratory Data Analysis`
2. `Feature Engineering`
3. `Base Model`
4. `Feature Importance`
5. `Hyperparameter Tuning`
6. `Final Model`

> View the [notebook](./lightgbm_model.ipynb) for more details.

### App

The [app](./app.py) file contains a FLASK application serving a RESTful API that will consume user input from a Flutter frontend and call the featureEngineering file. The predict API does the follwing:

1. Takes user input from the form in the frontend.
2. Creates a dataframe of all possible combinations with the columns `date, store, item, sales`.
3. Calls the `featureEngineering` module to generate features using `input_data.csv`.
4. Imports the `final_model.pkl` and passes the input to the model to perform prediction.
5. Returns a json object with tht prediction.


> View the [app](./app.py) for more details.

### featureEngineering

The [featureEngineering](./featureEngineering.py) app takes in `input_data.csv` and performs feature engineering to the user input has the same number of features as the ones used when training the model.

1. Takes user input from the Flask API.
2. Generates features so that `input_data` has 25 features that model expects.
3. Produces `X_test_final` that the Flask API will use to perform prediction.

> View the [featureEngineering](./featureEngineering.py) for more details.

### gain

The [gain](./gain.py) file contains the gain value which it obtains from the training dataset to trim features generated down to the 25 important features the model expects.

1. Takes training dataset.
2. Creates features using the dataset.
3. Calculates gain of the features using the `first_model`.
4. Outputs the gain to `featureEngineering`.

> View the [gain](./gain.py) for more details.
# Product-Demand-Predictor
