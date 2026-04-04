import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # IMPORTANT for frontend connection

# -----------------------------
# LOAD MODEL + ENCODER
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.abspath(
    os.path.join(BASE_DIR, "../../ml_models/risk_prediction/model.h5")
)

encoder_path = os.path.abspath(
    os.path.join(BASE_DIR, "../../ml_models/risk_prediction/label_encoder.pkl")
)

model = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(encoder_path)

print("✅ Model loaded successfully")

# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route('/')
def home():
    return "SafeHer AI API is running 🚀"

# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    try:
        data = request.json

        # Debug
        print("📡 API CALLED:", data)

        lat = data['lat']
        long = data['long']
        hour = data['hour']
        crime_score = data['crime_score']
        crowd_density = data['crowd_density']

        input_data = np.array([[lat, long, hour, crime_score, crowd_density]])

        prediction = model.predict(input_data)
        class_index = np.argmax(prediction)

        risk_level = label_encoder.inverse_transform([class_index])[0]
        risk_score = int(np.max(prediction) * 100)

        return jsonify({
            "risk_score": risk_score,
            "level": risk_level
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)})

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)