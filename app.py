import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "stacking_model.pkl"
model = joblib.load(MODEL_PATH)

# Define feature columns (must match training data)
FEATURE_COLUMNS = [
    "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3", 
    "Temperature", "Humidity", "WindSpeed", 
    "RespiratoryCases", "CardiovascularCases", 
    "HospitalAdmissions", "HealthImpactScore"
]

# Define mapping for Health Impact classes
IMPACT_MAP = {
    0: "Good Health Impact (Air quality is excellent, low risk(Class 0))",
    1: "Moderate Health Impact (Minor effects on sensitive groups(Class 1))",
    2: "Unhealthy for Sensitive Groups (Moderate risk, sensitive groups affected(Class 2))",
    3: "Unhealthy (High risk for everyone(Class 3))",
    4: "Hazardous (Severe risk, poses serious health issues(Class 4))"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect input data from the form
            input_data = {col: float(request.form[col]) for col in FEATURE_COLUMNS}

            # Convert input data to a DataFrame with the correct column names
            df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

            # Print the input data for debugging purposes (optional)
            print("Input DataFrame for Prediction:\n", df)

            # Make prediction without scaling
            prediction = model.predict(df)[0]

            # Map the numeric prediction to a meaningful representation
            prediction_result = IMPACT_MAP.get(prediction, "Unknown Health Impact Class")

            # Return prediction result to the UI
            return render_template("index.html", prediction=prediction_result)
        except Exception as e:
            # In case of error, show error message on UI
            return render_template("index.html", error=str(e))

    # If it's a GET request, just render the empty form
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
