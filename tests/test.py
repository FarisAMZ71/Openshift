#!/usr/bin/env python3
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def validate_model():
    """Validate the model performance meets minimum requirements"""
    try:
        # Load model
        with open('model/housing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data (you'd have a separate test dataset)
        # For demo purposes, using sample data
        test_features = np.array([[4.526, 28.0, 5.118, 1.073, 558.0, 2.547, 33.49, -117.16]])
        
        # Make prediction
        prediction = model.predict(test_features)
        
        # Basic validation checks
        assert prediction[0] > 0, "Prediction must be positive"
        assert prediction[0] < 10.0, "Prediction seems too high"
        
        print(f"Model validation passed. Sample prediction: {prediction[0]:.2f}")
        return True
        
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

if __name__ == "__main__":
    if validate_model():
        exit(0)
    else:
        exit(1)