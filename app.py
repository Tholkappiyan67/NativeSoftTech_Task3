import numpy as np
import pickle
from flask import Flask, request, render_template

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('forest_fire.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        temperature = float(request.form.get('Temperature'))
        oxygen = float(request.form.get('Oxygen'))
        humidity = float(request.form.get('Humidity'))

        # Prepare input for model
        input_data = np.array([[temperature, oxygen, humidity]])  # Ensure correct shape

        # Make prediction
        prediction = model.predict_proba(input_data)[:, 1]  # Get probability of fire

        # Convert to percentage
        probability = prediction[0] * 100

        return f"The predicted probability of forest fire is: {probability:.2f}%"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
