# Import relevant libraries

from flask import Flask, render_template, request
from datetime import datetime,timedelta, date
import jsonify
import pickle
import numpy as np
import pandas as pd
from itertools import product

# Importing featureEngineering at this point will cause an error since it needs to be imported after input_data.csv is created
# from featureEngineering import *

# Flask Initialization
app = Flask(__name__)

# generate_combinations function will generate all possible combinations of date, store, and item
def generate_combinations(start_date, end_date, store, item):
    # Create a list of dates between start_date and end_date
    days = [start_date + timedelta(days=x) for x in range(0, (end_date-start_date).days + 1)]
    # Create a list of all possible combinations of store and item
    combinations = list(product([store], [item]))
    # Create a list of all possible combinations of date, store, and item
    date_combinations = [(day, store, item) for day in days for store, item in combinations]
    return date_combinations


# API to load index page
@app.route('/', methods = ['POST', 'GET'])
def index():
    return render_template('index.html')

# API to perform prediction
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the form
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        store = request.form["store"]
        item = request.form["item"]

        # call funtion to generate all possible combinations of date, store, and item
        combinations = generate_combinations(start_date, end_date, store, item)
        print(combinations)
     


        # Prepare the input data in the format expected by the model
        input_data = pd.DataFrame(combinations, columns=['date', 'store', 'item'])
        input_data['date'] = pd.to_datetime(input_data['date'])
        input_data['store'] = input_data['store'].astype(int)
        input_data['item'] = input_data['item'].astype(int)
        input_data = input_data.assign(sales = "")
        print(input_data)
        print(input_data.dtypes)

        # Feature Engineering - Generate csv file for feature engineering
        input_data.to_csv("input_data.csv", index=None)

        # Lazy import - call featureEngineering only after input_data.csv is created
        module = __import__("featureEngineering")
        print(module.X_test_final.shape)
        print(module.df.shape)
        print(module.train_final.shape)
        

        # Make prediction using model 
        loaded_model = pickle.load(open('final_model.pkl', 'rb'))
        prediction = loaded_model.predict(module.X_test_final)

        # Take the first value of prediction
        output = prediction[0]
        print(output)
        # input the prediction into the input_data dataframe in the sales column
        for i in range(len(prediction)):
            input_data.at[i,'sales'] = prediction[i]
        input_data['sales'] = input_data['sales'].astype(int)
        print(input_data)
        table = input_data.to_html()
        # print(table)
        # write html to file
        with open("./templates/result.html", "w") as file:
            pass
        css = "<style>    table {        font-family: Arial, sans-serif;        background-color: #f2f2f2;    }    th {        background-color: #4CAF50;        color: white;        padding: 8px;        text-align: left;    }    td {        border: 1px solid #ddd;        padding: 8px;    }    tr:hover {        background-color: #ddd;    }</style>"
        text_file = open("./templates/result.html", "w")
        text_file.write(css)
        text_file.write(table)
        text_file.close()
        data = input_data.to_json('result.json', orient="records")
        print(data)
        # Return the result as a json object
        return render_template('result.html')
    
    @app.route('/result', methods = ['POST', 'GET'])
    def result():
        return render_template('result.html', data=result)



if __name__=="__main__":
    app.run(debug=True)