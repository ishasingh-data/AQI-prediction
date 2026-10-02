from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = load_model('aqi_gru_model.h5', compile=False)
scaler = joblib.load('aqi_scaler.pkl')

def categorize_aqi(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input values
        input_data = [float(x) for x in request.form.values()]
        if len(input_data) != 5:
            return render_template('index.html', prediction_text="Please enter all 5 input features.")

        # Add dummy AQI column for scaler compatibility
        input_with_dummy = input_data + [0]  # 5 features + dummy AQI
        input_array = np.array(input_with_dummy).reshape(1, -1)

        # Scale full input
        scaled_input = scaler.transform(input_array)
        scaled_features = scaled_input[0][:-1]  # Remove dummy AQI

        # Create sequence of 24 timesteps for GRU input
        sequence = np.tile(scaled_features, (24, 1)).reshape(1, 24, 5)

        # Predict AQI (scaled)
        prediction_scaled = model.predict(sequence, verbose=0)[0][0]

        # Combine with features to inverse scale
        inverse_input = np.hstack([scaled_features, prediction_scaled]).reshape(1, -1)
        predicted_aqi = scaler.inverse_transform(inverse_input)[0][-1]

        # Get AQI category
        category = categorize_aqi(predicted_aqi)

        # Redirect to result page with query parameters
        return redirect(url_for('result', aqi=f"{predicted_aqi:.2f}", category=category))

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/result')
def result():
    aqi = request.args.get('aqi')
    category = request.args.get('category')
    if aqi and category:
        return render_template('result.html', aqi=aqi, category=category)
    else:
        # Missing data, redirect back to input page
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
