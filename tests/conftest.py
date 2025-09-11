import pytest
import tempfile
import os
import sys
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add the application directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'application'))

@pytest.fixture
def sample_features():
    """Sample features for testing predictions"""
    return [
        8.3252,    # MedInc
        41.0,      # HouseAge  
        6.984,     # AveRooms
        1.024,     # AveBedrms
        322.0,     # Population
        2.556,     # AveOccup
        37.88,     # Latitude
        -122.23    # Longitude
    ]

@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        'feature_names': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                         'Population', 'AveOccup', 'Latitude', 'Longitude'],
        'model_version': '1.0.0',
        'training_date': '2024-01-01T00:00:00',
        'test_mae': 0.5,
        'framework': 'xgboost',
        'framework_version': '1.7.0',
        'model_type': 'XGBRegressor',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }

@pytest.fixture
def mock_model():
    """Mock XGBoost model for testing"""
    mock = Mock()
    mock.predict.return_value = np.array([2.5])  # Mock prediction
    return mock

@pytest.fixture
def mock_scaler():
    """Mock StandardScaler for testing"""
    mock = Mock()
    mock.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    return mock

@pytest.fixture
def temp_model_files(sample_metadata):
    """Create temporary model files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create models directory
        models_dir = os.path.join(temp_dir, 'models')
        os.makedirs(models_dir)
        
        # Create metadata file
        metadata_path = os.path.join(models_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        # Mock model and scaler files (they would exist in real scenario)
        model_path = os.path.join(models_dir, 'housing_model.json')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        
        # Create empty files to simulate existence
        open(model_path, 'w').close()
        open(scaler_path, 'w').close()
        
        yield {
            'models_dir': models_dir,
            'metadata_path': metadata_path,
            'model_path': model_path,
            'scaler_path': scaler_path
        }