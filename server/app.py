import pickle as pkl
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load scaler and model
script_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(script_dir, 'scaler.pkl')
scaler = pkl.load(open(scaler_path, 'rb'))

file_path = os.path.join(script_dir, 'nb.pkl')
with open(file_path, 'rb') as f:
    model = pkl.load(f)

# Prediction function
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, Dpf, Age):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, Dpf, Age]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    if prediction == 1:
        return {
            'prediction': "You have high chances of Diabetes! Please consult a Doctor",
            'gif_url': "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOTZlY2pwcDNtcnNhc2JwdDk4YnVqenRpcXl0OXFxdWRya3U0dmZ4aCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6wrebnKWmvx4ZBio/giphy.gif"
        }
    else:
        return {
            'prediction': "You have low chances of Diabetes. Please maintain a healthy lifestyle",
            'gif_url': "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2txb242N3pkMmp0ODRiangydm9raDY5OHBhYmw1Y2NobjM0cGZtNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/W1GG6RYUcWxoHl3jV9/giphy.gif"
        }

# API route
@app.route('/predict', methods=['POST'])
def predictions():
    try:
        data = request.get_json()
        Age = float(data.get('Age'))
        Pregnancies = float(data.get('Pregnancies'))
        Glucose = float(data.get('Glucose'))
        BloodPressure = float(data.get('BloodPressure'))
        Insulin = float(data.get('Insulin'))
        Bmi = float(data.get('BMI'))
        SkinThickness = float(data.get('SkinThickness'))
        Dpf = float(data.get('DPF'))

        result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, Dpf, Age)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Render will pick port from environment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
