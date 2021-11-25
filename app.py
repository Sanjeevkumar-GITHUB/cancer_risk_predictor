# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:45:53 2021

@author: sanje
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScale

app = Flask(__name__)
model = pickle.load(open('canp_model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index1.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = int(request.form['Gender'])
        Air_Pollution = int(request.form['Air Pollution'])
        Alcohol_use = int(request.form['Alcohol use'])
        Dust_Allergy = int(request.form['Dust Allergy'])
        OccuPational_Hazards = int(request.form['OccuPational Hazards'])
        Genetic_Risk = int(request.form['Genetic Risk'])
        chronic_Lung_Disease = int(request.form['chronic Lung Disease'])
        Balanced_Diet = int(request.form['Balanced Diet'])
        Obesity = int(request.form['Obesity'])
        Smoking = int(request.form['Smoking'])
        Passive_Smoker = int(request.form['Passive Smoker'])
        Chest_Pain = int(request.form['Chest Pain'])
        Coughing_of_Blood = int(request.form['Coughing of Blood'])
        Fatigue = int(request.form['Fatigue'])
        Weight_Loss = int(request.form['Weight Loss'])
        Shortness_of_Breath = int(request.form['Shortness of Breath'])
        Wheezing = int(request.form['Wheezing'])
        Swallowing_Difficulty = int(request.form['Swallowing Difficulty'])
        Clubbing_of_Finger_Nails = int(request.form['Clubbing of Finger Nails'])
        Frequent_Cold = int(request.form['Frequent Cold'])
        Dry_Cough = int(request.form['Dry Cough'])
        Snoring = int(request.form['Snoring'])
        prediction = model.predict(np.array([[Age, Gender, Air_Pollution, Alcohol_use, Dust_Allergy,
                                              OccuPational_Hazards, Genetic_Risk, chronic_Lung_Disease, Balanced_Diet,
                                              Obesity, Smoking, Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue,
                                              Weight_Loss, Shortness_of_Breath, Wheezing, Swallowing_Difficulty,
                                              Clubbing_of_Finger_Nails, Frequent_Cold, Dry_Cough, Snoring]]))
        output = prediction
        if (output == 'High'):
            return render_template('index1.html', prediction_texts="you have high cancer risk")
        elif (output == 'Low'):
            return render_template('index1.html', prediction_text="You have low cancer risk")
        else:
            return render_template('index1.html', prediction_text="You have medium cancer risk")

    else:
        return render_template('index1.html')


if __name__ == "__main__":
    app.run(debug=True)
