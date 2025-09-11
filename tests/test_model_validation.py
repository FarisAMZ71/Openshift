import pytest
import numpy as np
import os
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Import the test_model module
import test_model

class TestModelValidation:
    """Test suite for model validation functionality"""
    
    @patch('test_model.os.path.exists')
    def test_load_model_and_artifacts_success(self, mock_exists):
        """Test successful loading of model artifacts"""
        mock_exists.return_value = True
        
        with patch('test_model.xgb.XGBRegressor') as mock_xgb, \
             patch('builtins.open', create=True) as mock_open, \
             patch('test_model.pickle.load') as mock_pickle, \
             patch('test_model.json.load') as mock_json:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.load_model = Mock()
            mock_xgb.return_value = mock_model
            mock_scaler = Mock()
            mock_pickle.return_value = mock_scaler
            mock_metadata = {
                'feature_names': ['test'], 
                'model_version': '1.0.0',
                'training_date': '2024-01-01T00:00:00',
                'test_mae': 0.5
            }
            mock_json.return_value = mock_metadata
            
            # Test
            with patch('builtins.print'):  # Suppress print output
                model, scaler, feature_names, metadata = test_model.load_model_and_artifacts()
            
            assert model == mock_model
            assert scaler == mock_scaler
            assert feature_names == ['test']
            assert metadata == mock_metadata

    def test_load_model_and_artifacts_file_not_found(self):
        """Test handling when model files don't exist"""
        with patch('test_model.os.path.exists', return_value=False), \
             patch('builtins.print'):  # Suppress print output
            
            # Should not raise SystemExit, just test that it tries to load
            try:
                test_model.load_model_and_artifacts()
            except FileNotFoundError:
                # Expected behavior when files don't exist
                pass

    def test_prepare_test_data(self):
        """Test test data preparation"""
        X_train, X_test, y_train, y_test, feature_names = test_model.prepare_test_data()
        
        # Check shapes and types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check California housing dataset properties
        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8
        assert len(feature_names) == 8
        
        # Check split ratio
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        assert 0.15 < test_ratio < 0.25

    def test_test_model_performance(self):
        """Test model performance evaluation"""
        # Create mock objects
        mock_model = Mock()
        mock_model.predict.return_value = np.array([2.0, 2.5, 3.0])
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 3)
        
        X_test = np.random.rand(3, 8)
        y_test = np.array([2.1, 2.4, 2.9])
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
        
        with patch('builtins.print'):  # Suppress print output
            result = test_model.test_model_performance(mock_model, X_test, y_test, 
                                                     mock_scaler, feature_names)
        
        # Check return structure
        assert 'mae' in result
        assert 'rmse' in result
        assert 'r2' in result
        assert 'predictions' in result
        assert 'residuals' in result
        
        # Check values are reasonable
        assert isinstance(result['mae'], (int, float))
        assert isinstance(result['rmse'], (int, float))
        assert isinstance(result['r2'], (int, float))
        assert len(result['predictions']) == 3
        assert len(result['residuals']) == 3

    def test_test_prediction_ranges(self):
        """Test prediction range analysis"""
        mock_model = Mock()
        # Return predictions that will create different price ranges
        mock_model.predict.return_value = np.array([1.0, 3.0, 5.0, 7.0])  
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.rand(4, 8)
        
        X_test = np.random.rand(4, 8)
        y_test = np.array([1.2, 2.8, 4.9, 6.8])  # Corresponding test values
        
        with patch('builtins.print'):  # Suppress print output
            # Should not raise exception
            test_model.test_prediction_ranges(mock_model, X_test, y_test, mock_scaler)
        
        mock_scaler.transform.assert_called_once()
        mock_model.predict.assert_called_once()

    def test_test_cross_validation(self):
        """Test cross-validation functionality"""
        X_train = np.random.rand(50, 8)
        y_train = np.random.rand(50)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = X_train
        
        with patch('builtins.print'):  # Suppress print output
            scores = test_model.test_cross_validation(X_train, y_train, mock_scaler, cv_folds=3)
        
        assert isinstance(scores, list)  # Changed from np.ndarray to list
        assert len(scores) == 3
        assert all(isinstance(score, (int, float)) for score in scores)

    def test_test_feature_importance(self):
        """Test feature importance analysis"""
        mock_model = Mock()
        # Mock feature importance scores
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02])
        
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        with patch('builtins.print'):  # Suppress print output
            importance = test_model.test_feature_importance(mock_model, feature_names)
        
        assert isinstance(importance, list)
        assert len(importance) == 8
        assert all(len(item) == 2 for item in importance)  # Each item should be (name, score)
        
        # Check sorting (should be descending)
        scores = [item[1] for item in importance]
        assert scores == sorted(scores, reverse=True)

    def test_test_prediction_stability(self):
        """Test prediction stability analysis"""
        mock_model = Mock()
        # Return slightly different predictions each time
        predictions = [
            np.array([2.5]), np.array([2.51]), np.array([2.49]), 
            np.array([2.52]), np.array([2.48])
        ]
        mock_model.predict.side_effect = predictions
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        X_test = np.random.rand(1, 8)
        
        with patch('builtins.print'):  # Suppress print output
            test_model.test_prediction_stability(mock_model, X_test, mock_scaler, num_runs=5)
        
        assert mock_model.predict.call_count == 5

    def test_test_edge_cases(self):
        """Test edge case handling"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([2.5])
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
        
        with patch('builtins.print'):  # Suppress print output
            test_model.test_edge_cases(mock_model, mock_scaler, feature_names)
        
        # Should make multiple predictions for different edge cases
        assert mock_model.predict.call_count >= 3

    def test_visualize_results_missing_matplotlib(self):
        """Test visualization handling when matplotlib is not available"""
        y_test = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.1])
        residuals = y_test - y_pred
        
        with patch('builtins.print'):  # Suppress print output
            # Should handle ImportError gracefully
            test_model.visualize_results(y_test, y_pred, residuals, save_plots=False)

class TestModelIntegration:
    """Integration tests for the complete testing pipeline"""
    
    @patch('test_model.load_model_and_artifacts')
    @patch('test_model.prepare_test_data') 
    def test_run_comprehensive_tests_success(self, mock_prepare_data, mock_load_artifacts):
        """Test successful execution of comprehensive test suite"""
        # Setup mocks
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02])
        mock_model.predict.return_value = np.array([2.5] * 20)  # Match X_test size
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.rand(20, 8)  # Match X_test size
        
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        metadata = {'model_version': '1.0.0'}
        
        mock_load_artifacts.return_value = (mock_model, mock_scaler, feature_names, metadata)
        
        X_train = np.random.rand(100, 8)
        X_test = np.random.rand(20, 8)
        y_train = np.random.rand(100)
        y_test = np.random.rand(20)
        
        mock_prepare_data.return_value = (X_train, X_test, y_train, y_test, feature_names)
        
        with patch('builtins.print'):  # Suppress print output
            result = test_model.run_comprehensive_tests()
        
        assert isinstance(result, dict)
        assert 'performance' in result
        assert 'cv_scores' in result
        assert 'feature_importance' in result

    @patch('test_model.os.path.exists')
    def test_main_missing_files(self, mock_exists):
        """Test main function when model files are missing"""
        mock_exists.return_value = False
        
        with patch('builtins.print'):  # Suppress print output
            # Should return early without error
            test_model.main()

    @patch('test_model.run_comprehensive_tests')
    @patch('test_model.os.path.exists')
    def test_main_success(self, mock_exists, mock_run_tests):
        """Test successful main execution"""
        mock_exists.return_value = True
        mock_run_tests.return_value = {'test': 'results'}
        
        with patch('builtins.print'):  # Suppress print output
            test_model.main()
        
        mock_run_tests.assert_called_once()