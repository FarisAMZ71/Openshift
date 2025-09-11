import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
import sys
import os

# Import the API module
import api

class TestAPIEndpoints:
    """Test suite for Flask API endpoints"""
    
    @pytest.fixture
    def client(self, mock_model, mock_scaler, sample_metadata):
        """Create test client with mocked dependencies"""
        with patch.object(api, 'model', mock_model), \
             patch.object(api, 'scaler', mock_scaler), \
             patch.object(api, 'metadata', sample_metadata):
            
            api.app.config['TESTING'] = True
            with api.app.test_client() as client:
                yield client

    def test_health_check_healthy(self, client):
        """Test health check when model is loaded"""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['model_version'] == '1.0.0'

    def test_health_check_unhealthy(self):
        """Test health check when model is not loaded"""
        with patch.object(api, 'model', None):
            api.app.config['TESTING'] = True
            with api.app.test_client() as client:
                response = client.get('/health')
                data = json.loads(response.data)
                
                assert response.status_code == 503
                assert data['status'] == 'unhealthy'
                assert 'error' in data

    def test_readiness_check_ready(self, client):
        """Test readiness check when model is loaded"""
        response = client.get('/ready')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['ready'] is True

    def test_readiness_check_not_ready(self):
        """Test readiness check when model is not loaded"""
        with patch.object(api, 'model', None):
            api.app.config['TESTING'] = True
            with api.app.test_client() as client:
                response = client.get('/ready')
                data = json.loads(response.data)
                
                assert response.status_code == 503
                assert data['ready'] is False

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get('/metrics')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'model_version' in data
        assert 'training_date' in data
        assert 'test_mae' in data
        assert 'feature_names' in data
        assert data['framework'] == 'xgboost'

    def test_root_endpoint(self, client):
        """Test root endpoint documentation"""
        response = client.get('/')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['name'] == 'Housing Price Prediction API'
        assert 'endpoints' in data
        assert 'example_request' in data

    def test_predict_success(self, client, sample_features):
        """Test successful single prediction"""
        payload = {'features': sample_features}
        response = client.post('/predict', json=payload)
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'prediction' in data
        assert data['unit'] == 'USD'
        assert 'features_used' in data
        assert 'model_version' in data
        assert 'timestamp' in data
        assert isinstance(data['prediction'], (int, float))

    def test_predict_no_data(self, client):
        """Test prediction with no data"""
        response = client.post('/predict', data='', content_type='application/json')
        data = json.loads(response.data)
        
        # Flask test client returns 500 for malformed JSON, which triggers exception handling
        assert response.status_code == 500
        assert 'error' in data
        assert data['error'] == 'Prediction failed'

    def test_predict_no_features(self, client):
        """Test prediction with no features"""
        payload = {}
        response = client.post('/predict', json=payload)
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data
        assert data['error'] == 'No data provided'

    def test_predict_wrong_feature_count(self, client):
        """Test prediction with wrong number of features"""
        payload = {'features': [1.0, 2.0, 3.0]}  # Too few features
        response = client.post('/predict', json=payload)
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data
        assert 'Expected 8 features' in data['error']
        assert 'expected_features' in data

    def test_batch_predict_success(self, client, sample_features):
        """Test successful batch prediction"""
        payload = {
            'batch': [
                {'id': 'house_1', 'features': sample_features},
                {'id': 'house_2', 'features': sample_features}
            ]
        }
        response = client.post('/batch_predict', json=payload)
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'predictions' in data
        assert len(data['predictions']) == 2
        assert data['total'] == 2
        assert 'model_version' in data
        assert 'timestamp' in data
        
        for prediction in data['predictions']:
            assert 'id' in prediction
            assert 'prediction' in prediction
            assert 'unit' in prediction

    def test_batch_predict_no_data(self, client):
        """Test batch prediction with no data"""
        response = client.post('/batch_predict', data='', content_type='application/json')
        data = json.loads(response.data)
        
        # Flask test client returns 500 for malformed JSON, which triggers exception handling
        assert response.status_code == 500
        assert 'error' in data
        assert data['error'] == 'Batch prediction failed'

    def test_batch_predict_invalid_features(self, client):
        """Test batch prediction with invalid features"""
        payload = {
            'batch': [
                {'id': 'house_1', 'features': [1.0, 2.0]},  # Too few features
                {'id': 'house_2', 'features': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}  # Correct
            ]
        }
        response = client.post('/batch_predict', json=payload)
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert len(data['predictions']) == 2
        assert 'error' in data['predictions'][0]
        assert 'prediction' in data['predictions'][1]

    @patch('api.logger')
    def test_predict_exception_handling(self, mock_logger, client, sample_features):
        """Test prediction error handling"""
        with patch.object(api, 'model') as mock_model:
            mock_model.predict.side_effect = Exception("Model error")
            
            payload = {'features': sample_features}
            response = client.post('/predict', json=payload)
            data = json.loads(response.data)
            
            assert response.status_code == 500
            assert 'error' in data
            assert data['error'] == 'Prediction failed'
            mock_logger.error.assert_called()

    @patch('api.logger')
    def test_batch_predict_exception_handling(self, mock_logger, client, sample_features):
        """Test batch prediction error handling"""
        with patch.object(api, 'model') as mock_model:
            mock_model.predict.side_effect = Exception("Model error")
            
            payload = {'batch': [{'id': 'house_1', 'features': sample_features}]}
            response = client.post('/batch_predict', json=payload)
            data = json.loads(response.data)
            
            assert response.status_code == 500
            assert 'error' in data
            assert data['error'] == 'Batch prediction failed'
            mock_logger.error.assert_called()

class TestModelLoading:
    """Test suite for model loading functionality"""
    
    @patch('api.xgb.Booster')
    @patch('api.pickle.load')
    @patch('builtins.open')
    @patch('api.json.load')
    @patch('os.getenv')
    def test_load_model_artifacts_success(self, mock_getenv, mock_json_load, 
                                        mock_open, mock_pickle_load, mock_booster):
        """Test successful model artifacts loading"""
        # Setup mocks
        mock_getenv.side_effect = lambda key, default: default
        mock_json_load.return_value = {'model_version': '1.0.0'}
        mock_model = Mock()
        mock_booster.return_value = mock_model
        mock_scaler = Mock()
        mock_pickle_load.return_value = mock_scaler
        
        # Test
        result = api.load_model_artifacts()
        
        assert result is True
        assert api.model == mock_model
        assert api.scaler == mock_scaler
        assert api.metadata == {'model_version': '1.0.0'}

    @patch('api.logger')
    @patch('api.xgb.Booster')
    def test_load_model_artifacts_failure(self, mock_booster, mock_logger):
        """Test model artifacts loading failure"""
        mock_booster.side_effect = Exception("File not found")
        
        result = api.load_model_artifacts()
        
        assert result is False
        mock_logger.error.assert_called()