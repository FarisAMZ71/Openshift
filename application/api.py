#!/usr/bin/env python3
"""
Flask API for serving housing price predictions
"""

import os
import json
import pickle
import logging
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontends

# Global variables for model and scaler
model = None
scaler = None
metadata = None

def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, metadata
    
    try:
        # Load the model
        model_path = os.getenv('MODEL_PATH', 'models/housing_model.keras')
        logger.info(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)
        
        # Load the scaler
        scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = os.getenv('METADATA_PATH', 'models/metadata.json')
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("✅ All model artifacts loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model artifacts: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for OpenShift probes"""
    if model is not None and scaler is not None:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_version': metadata.get('model_version', 'unknown')
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Model not loaded',
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint for OpenShift probes"""
    if model is not None and scaler is not None:
        return jsonify({'ready': True}), 200
    else:
        return jsonify({'ready': False}), 503

@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint for monitoring"""
    return jsonify({
        'model_version': metadata.get('model_version', 'unknown'),
        'training_date': metadata.get('training_date', 'unknown'),
        'test_mae': metadata.get('test_mae', 0),
        'feature_names': metadata.get('feature_names', []),
        'framework': metadata.get('framework', 'unknown'),
        'framework_version': metadata.get('framework_version', 'unknown')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        features = data.get('features', None)
        if features is None:
            return jsonify({
                'error': 'No features provided',
                'expected_format': {
                    'features': metadata.get('feature_names', [])
                }
            }), 400
        
        # Validate feature count
        expected_features = len(metadata.get('feature_names', []))
        if len(features) != expected_features:
            return jsonify({
                'error': f'Expected {expected_features} features, got {len(features)}',
                'expected_features': metadata.get('feature_names', [])
            }), 400
        
        # Prepare input for prediction
        input_array = np.array(features).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        predicted_price = float(prediction[0][0]) * 100000  # Convert to dollars
        
        # Log the prediction
        logger.info(f"Prediction made: ${predicted_price:.2f}")
        
        # Return prediction
        return jsonify({
            'prediction': predicted_price,
            'unit': 'USD',
            'features_used': dict(zip(metadata.get('feature_names', []), features)),
            'model_version': metadata.get('model_version', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'batch' not in data:
            return jsonify({'error': 'No batch data provided'}), 400
        
        batch = data['batch']
        predictions = []
        
        for item in batch:
            features = item.get('features', [])
            if len(features) != len(metadata.get('feature_names', [])):
                predictions.append({
                    'error': 'Invalid feature count',
                    'id': item.get('id', 'unknown')
                })
                continue
            
            # Prepare and predict
            input_array = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled, verbose=0)
            predicted_price = float(prediction[0][0]) * 100000
            
            predictions.append({
                'id': item.get('id', 'unknown'),
                'prediction': predicted_price,
                'unit': 'USD'
            })
        
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'model_version': metadata.get('model_version', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        'name': 'Housing Price Prediction API',
        'version': metadata.get('model_version', 'unknown') if metadata else 'unknown',
        'endpoints': {
            '/': 'API documentation (this page)',
            '/health': 'Health check endpoint',
            '/ready': 'Readiness check endpoint',
            '/metrics': 'Model metrics and metadata',
            '/predict': 'Single prediction endpoint (POST)',
            '/batch_predict': 'Batch prediction endpoint (POST)'
        },
        'example_request': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'features': [
                    8.3252,    # MedInc
                    41.0,      # HouseAge  
                    6.984,     # AveRooms
                    1.024,     # AveBedrms
                    322.0,     # Population
                    2.556,     # AveOccup
                    37.88,     # Latitude
                    -122.23    # Longitude
                ]
            }
        }
    })

if __name__ == '__main__':
    # Load model artifacts on startup
    if not load_model_artifacts():
        logger.error("Failed to load model artifacts. Exiting.")
        exit(1)
    
    # Run the Flask app
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)