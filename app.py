# app.py
# Flask Web Application for Titanic Survival Prediction

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'titanic_model.pkl'
FEATURES_PATH = 'feature_columns.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Please run 'python train_model.py' first!")
    model = None
    feature_columns = None


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Prepare input data
        input_data = {
            'Pclass': int(data.get('pclass')),
            'Age': float(data.get('age')),
            'SibSp': int(data.get('sibsp', 0)),
            'Parch': int(data.get('parch', 0)),
            'Fare': float(data.get('fare')),
            'Sex_male': 1 if data.get('sex').lower() == 'male' else 0,
            'Embarked_Q': 1 if data.get('embarked') == 'Q' else 0,
            'Embarked_S': 1 if data.get('embarked') == 'S' else 0
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'survived': bool(prediction),
            'confidence': float(max(model.predict_proba(input_df)[0]))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("\n" + "="*60)
        print("⚠️  WARNING: Model file not found!")
        print("="*60)
        print("Please run: python train_model.py")
        print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("🚀 Starting Flask Server...")
    print("="*60)
    print("📍 Server: http://127.0.0.1:5001")
    print("📍 Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)