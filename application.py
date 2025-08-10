from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get numerical inputs
            Temperature = float(request.form.get('Temperature', 0))
            RH = float(request.form.get('RH', 0))  
            Ws = float(request.form.get('Ws', 0))  # Changed from Wind to Ws
            Rain = float(request.form.get('Rain', 0))
            FFMC = float(request.form.get('FFMC', 0))
            DMC = float(request.form.get('DMC', 0))
            ISI = float(request.form.get('ISI', 0))
            # Convert Classes and Region to string, then encode
            Classes_input = request.form.get('Classes', '').lower()
            Region_input = request.form.get('Region', '').lower()

            # Example encoding (adjust as per your model)
            Classes = 1 if Classes_input == 'fire' else 0
            Region = 1 if Region_input == 'north' else 0

            # Prepare input array
            input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            scaled_input = scaler.transform(input_data)
            result = ridge_model.predict(scaled_input)

            return render_template('home.html', results=f"{result[0]:.2f}")
        except Exception as e:
            return render_template('home.html', results=f"Error: {str(e)}")
    else:
        return render_template('home.html', results="")
if __name__ == "__main__":
    app.run(debug=True)
