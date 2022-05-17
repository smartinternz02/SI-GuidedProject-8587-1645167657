from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "eOrzaCZKtd_8V08dAXW-ec9EZ57nBO9Zy_iJ5pHXx1Uu"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
@app.route('/')
def home():
    return render_template("indexEA.html")


@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    total=[input_features]
    # features_value = [np.array(input_features)]
    # features_name = ['Education', 'JobInvolvement', 'JobLevel', 'DailyRate(USD)',
    #                 'MonthlyIncome(USD)', 'NoofCompaniesWorked'
    #   , 'TotalWorkingYears', 'YearsAtCompany',
    #               'YearsInCurrentRole', 'YearsSinceLastPromotion',
    #              'YearsWithCurrentManager', 'TrainingTimesLastYear', 'PerformanceRating']

    # df = pd.DataFrame(features_value, columns=features_name)
    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": [array_of_input_fields], "values": [array_of_values_to_be_scored, another_array_of_values_to_be_scored]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/e5446775-3413-4e83-8104-0de1636ac859/predictions?version=2022-03-05', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())

    
    predictions = response_scoring.json()
    print(predictions)
    
    pred = response_scoring.json()

    output = pred['predictions'][0]['values'][0][0]

    return render_template('resultEA.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=False)