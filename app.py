from flask import Flask, request, jsonify  # Flask for creating the web app, request for handling input, jsonify for JSON responses
import joblib  # For loading the saved model and scaler files
import numpy as np  # For reshaping input data into the format the model expects

app = Flask(__name__)  # Initialize the Flask application - why? This is the core object for routing and running the server

# Load the trained model and scaler - why? To make predictions without retraining every time the app runs
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])  # Define the API endpoint - why? Allows external tools to send data and get churn prediction
def predict():
    try:
        data = request.json  # Get JSON input from request
        # Extract all 30 features in exact order from training - why? Matches model/scaler input to avoid shape error; use 0.0 default if missing
        features = [
            data.get('SeniorCitizen', 0.0),
            data.get('tenure', 0.0),
            data.get('MonthlyCharges', 0.0),
            data.get('TotalCharges', 0.0),
            data.get('gender_Male', 0.0),
            data.get('Partner_Yes', 0.0),
            data.get('Dependents_Yes', 0.0),
            data.get('PhoneService_Yes', 0.0),
            data.get('MultipleLines_No phone service', 0.0),
            data.get('MultipleLines_Yes', 0.0),
            data.get('InternetService_Fiber optic', 0.0),
            data.get('InternetService_No', 0.0),
            data.get('OnlineSecurity_No internet service', 0.0),
            data.get('OnlineSecurity_Yes', 0.0),
            data.get('OnlineBackup_No internet service', 0.0),
            data.get('OnlineBackup_Yes', 0.0),
            data.get('DeviceProtection_No internet service', 0.0),
            data.get('DeviceProtection_Yes', 0.0),
            data.get('TechSupport_No internet service', 0.0),
            data.get('TechSupport_Yes', 0.0),
            data.get('StreamingTV_No internet service', 0.0),
            data.get('StreamingTV_Yes', 0.0),
            data.get('StreamingMovies_No internet service', 0.0),
            data.get('StreamingMovies_Yes', 0.0),
            data.get('Contract_One year', 0.0),
            data.get('Contract_Two year', 0.0),
            data.get('PaperlessBilling_Yes', 0.0),
            data.get('PaymentMethod_Credit card (automatic)', 0.0),
            data.get('PaymentMethod_Electronic check', 0.0),
            data.get('PaymentMethod_Mailed check', 0.0)
        ]
        features = np.array(features).reshape(1, -1)  # Reshape to model input format
        scaled = scaler.transform(features)  # Scale input - matches training
        pred = model.predict(scaled)  # Predict
        score = model.predict_proba(scaled)[0][1]  # Probability of churn
        is_churn = bool(pred[0])  # True if churn predicted
        return jsonify({'churn': is_churn, 'score': score})  # Return JSON result
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Handle errors gracefully

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Run locally for testing - why? Debug mode shows errors