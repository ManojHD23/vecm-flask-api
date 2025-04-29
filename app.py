# === Imports ===
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.vector_ar.vecm import VECMResults

# === Create Flask App ===
app = Flask(__name__)

# === Load Models (Important) ===
vecm_fit = joblib.load('models/vecm_model.pkl')   # Load VECM model
pca_model = joblib.load('models/pca_model.pkl')    # Load PCA model

# === Preprocessing Function ===
def preprocess_input(new_data):
    """
    Takes new raw input DataFrame and applies PCA transformation.
    """
    numeric_new_data = new_data.select_dtypes(include=[np.number])
    pca_transformed = pca_model.transform(numeric_new_data)
    pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])])
    return pca_df

# === Forecasting Function ===
def forecast_vecm(new_preprocessed, steps_ahead=1):
    """
    Forecast future values based on preprocessed PCA data.
    """
    forecast_pca = vecm_fit.predict(steps=steps_ahead)
    forecast_pca_df = pd.DataFrame(forecast_pca, columns=new_preprocessed.columns)
    return forecast_pca_df

# === Routes ===

# Health Check (optional but professional)
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'message': 'VECM Forecasting API is up and running!'})

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Receive data
        data = request.get_json()
        
        # Step 2: Convert to DataFrame
        input_df = pd.DataFrame(data)

        # Step 3: Preprocess
        preprocessed_input = preprocess_input(input_df)

        # Step 4: Forecast
        steps_ahead = int(request.args.get('steps', 5))  # default forecast 5 steps
        forecast = forecast_vecm(preprocessed_input, steps_ahead=steps_ahead)

        # Step 5: Return prediction
        return jsonify(forecast.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
