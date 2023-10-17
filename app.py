# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:27:35 2023

@author: rushi
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl' , 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = []
    for value in request.form.values():
        # Convert "yes" to 1 and "no" to 0, if applicable
        if value.lower() == 'yes' or 'semi-furnished':
            int_value = 1
        elif value.lower() == 'no' or 'unfurnished':
            int_value = 0
        elif value.lower() == 'furnished':
            int_value = 2
        else:
            try:
                # Attempt to convert the input value to an integer
                int_value = int(value)
            except ValueError:
                # Handle the case where the input value is not a valid integer
                return render_template('error.html', error_message='Invalid input: Please enter valid integers or "yes" or "no".')

        int_features.append(int_value)

    # Continue with your processing using int_features...
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='House Price should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)