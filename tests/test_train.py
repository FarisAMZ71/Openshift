import pytest
import numpy as np
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Import the train module
import train

class TestTrainingPipeline:
    """Test suite for model training pipeline"""
    
    def test_download_and_prepare_data(self):
        """Test data download and preparation"""
        X_train, X_test, y_train, y_test, scaler, feature_names = train.download_and_prepare_data()
        
        # Check data shapes
        assert X_train.shape[1] == 8  # 8 features in California housing dataset
        assert X_test.shape[1] == 8
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check train/test split ratio (should be 80/20)
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        assert 0.15 < test_ratio < 0.25  # Allow some tolerance
        
        # Check scaler is fitted
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')
        
        # Check feature names
        expected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                           'Population', 'AveOccup', 'Latitude', 'Longitude']
        assert feature_names == expected_features
        
        # Check data is scaled (mean should be close to 0, std close to 1)
        assert abs(np.mean(X_train)) < 0.1
        assert abs(np.std(X_train) - 1.0) < 0.1

    def test_create_xgb_model(self):
        """Test XGBoost model creation"""
        model = train.create_xgb_model()
        
        assert isinstance(model, xgb.XGBRegressor)
        assert model.n_estimators == 1000
        assert model.max_depth == 6
        assert model.learning_rate == 0.1
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.8
        assert model.random_state == 42
        assert model.reg_alpha == 0.1
        assert model.reg_lambda == 1.0

    @patch('train.train_test_split')
    def test_train_model(self, mock_split):
        """Test model training process"""
        # Create mock data
        X_train = np.random.rand(100, 8)
        X_test = np.random.rand(20, 8)
        y_train = np.random.rand(100)
        y_test = np.random.rand(20)
        
        # Mock train_test_split for validation split
        mock_split.return_value = (X_train[:80], X_train[80:], y_train[:80], y_train[80:])
        
        # Create and train model
        model = train.create_xgb_model()
        model.set_params(n_estimators=10, early_stopping_rounds=5)  # Reduce for testing
        
        trained_model, test_mae = train.train_model(model, X_train, y_train, X_test, y_test)
        
        assert isinstance(trained_model, xgb.XGBRegressor)
        assert isinstance(test_mae, float)
        assert test_mae > 0
        mock_split.assert_called_once()

    def test_save_artifacts(self):
        """Test saving model artifacts"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Create a simple model with explicit parameters
                model = xgb.XGBRegressor(
                    n_estimators=10, 
                    max_depth=3,  # Set explicit value instead of None
                    learning_rate=0.1,
                    random_state=42
                )
                X_dummy = np.random.rand(10, 8)
                y_dummy = np.random.rand(10)
                model.fit(X_dummy, y_dummy)
                
                # Create a simple scaler
                scaler = StandardScaler()
                scaler.fit(X_dummy)
                
                feature_names = ['feature_' + str(i) for i in range(8)]
                test_mae = 0.5
                
                # Save artifacts
                train.save_artifacts(model, scaler, feature_names, test_mae)
                
                # Check files were created
                assert os.path.exists('models/housing_model.json')
                assert os.path.exists('models/scaler.pkl')
                assert os.path.exists('models/metadata.json')
                
                # Check metadata content
                with open('models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                assert metadata['feature_names'] == feature_names
                assert metadata['test_mae'] == test_mae
                assert metadata['framework'] == 'xgboost'
                assert 'training_date' in metadata
                assert 'model_version' in metadata
                
            finally:
                os.chdir(original_cwd)

class TestTrainingUtils:
    """Test utility functions in training module"""
    
    def test_reproducibility(self):
        """Test that training is reproducible with same random seed"""
        np.random.seed(42)
        X_train_1, X_test_1, y_train_1, y_test_1, scaler_1, _ = train.download_and_prepare_data()
        
        np.random.seed(42)
        X_train_2, X_test_2, y_train_2, y_test_2, scaler_2, _ = train.download_and_prepare_data()
        
        # Data should be identical
        np.testing.assert_array_equal(X_train_1, X_train_2)
        np.testing.assert_array_equal(X_test_1, X_test_2)
        np.testing.assert_array_equal(y_train_1, y_train_2)
        np.testing.assert_array_equal(y_test_1, y_test_2)

    def test_data_quality(self):
        """Test data quality after preparation"""
        X_train, X_test, y_train, y_test, scaler, feature_names = train.download_and_prepare_data()
        
        # Check for NaN values
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_train).any()
        assert not np.isnan(y_test).any()
        
        # Check for infinite values
        assert not np.isinf(X_train).any()
        assert not np.isinf(X_test).any()
        assert not np.isinf(y_train).any()
        assert not np.isinf(y_test).any()
        
        # Check target values are positive (housing prices should be positive)
        assert (y_train > 0).all()
        assert (y_test > 0).all()

    @patch('train.os.makedirs')
    @patch('train.json.dump')
    @patch('builtins.open')
    def test_save_artifacts_error_handling(self, mock_open, mock_json_dump, mock_makedirs):
        """Test error handling in save_artifacts"""
        # Mock an exception during file operations
        mock_open.side_effect = IOError("Permission denied")
        
        model = Mock()
        scaler = Mock()
        feature_names = ['test']
        metrics = 0.5
        
        with pytest.raises(IOError):
            train.save_artifacts(model, scaler, feature_names, metrics)

    def test_model_parameters_validation(self):
        """Test that model has expected parameters"""
        model = train.create_xgb_model()
        
        # Test parameter ranges
        assert 0 < model.learning_rate <= 1.0
        assert model.max_depth > 0
        assert model.n_estimators > 0
        assert 0 < model.subsample <= 1.0
        assert 0 < model.colsample_bytree <= 1.0
        assert model.reg_alpha >= 0
        assert model.reg_lambda >= 0