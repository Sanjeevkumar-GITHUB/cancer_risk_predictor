import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
model = pickle.load(open('canp_model.pkl', 'rb')) 

# Create application
app1 = Flask(__name__)

# Bind home function to URL
@app1.route('/')
def home():
    return render_template('index1.html')

# Bind predict function to URL
@app1.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [i for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 'High':
        return render_template('index1.html', 
                               result = 'The patient is not likely to have high risk!')
    elif output == 'Low' :
        return render_template('index1.html', 
                               result = 'The patient is likely to have low risk!')
    else:
         return render_template('index1.html', 
                               result = 'The patient is likely to have medium risk!')

if __name__ == '__main__':
#Run the application
    app1.run()
