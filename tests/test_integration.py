import pytest
import requests
import json
import sys
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import RequestException, ConnectionError, Timeout

# Import the test_api module from application directory
import test_api

class TestAPITesting:
    """Test suite for API testing functionality"""
    
    def test_print_colored(self):
        """Test colored print function"""
        with patch('builtins.print') as mock_print:
            test_api.print_colored("Test message", test_api.GREEN)
            mock_print.assert_called_once()

    @patch('requests.get')
    def test_endpoint_get_success(self, mock_get):
        """Test successful GET endpoint testing"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {'status': 'ok'}
        mock_get.return_value = mock_response
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/health')
        
        assert result is True
        mock_get.assert_called_once_with('https://api.test.com/health')

    @patch('requests.post')
    def test_endpoint_post_success(self, mock_post):
        """Test successful POST endpoint testing"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {'prediction': 250000}
        mock_post.return_value = mock_response
        
        test_data = {'features': [1, 2, 3, 4, 5, 6, 7, 8]}
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/predict', 
                                          method='POST', data=test_data)
        
        assert result is True
        mock_post.assert_called_once_with('https://api.test.com/predict', json=test_data)

    @patch('requests.get')
    def test_endpoint_wrong_status_code(self, mock_get):
        """Test endpoint with unexpected status code"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {'content-type': 'text/plain'}
        mock_response.text = "Not found"
        mock_get.return_value = mock_response
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/nonexistent')
        
        assert result is False

    @patch('requests.get')
    def test_endpoint_connection_error(self, mock_get):
        """Test endpoint with connection error"""
        mock_get.side_effect = ConnectionError("Connection refused")
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/health')
        
        assert result is False

    @patch('requests.get')
    def test_endpoint_timeout(self, mock_get):
        """Test endpoint with timeout"""
        mock_get.side_effect = Timeout("Request timed out")
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/health')
        
        assert result is False

    @patch('requests.get')
    def test_endpoint_invalid_json(self, mock_get):
        """Test endpoint with invalid JSON response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_get.return_value = mock_response
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/health')
        
        assert result is False

    @patch('requests.get')
    def test_endpoint_non_json_response(self, mock_get):
        """Test endpoint with non-JSON response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.text = "<html>OK</html>"
        mock_get.return_value = mock_response
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/health')
        
        assert result is True

    def test_endpoint_unsupported_method(self):
        """Test endpoint with unsupported HTTP method"""
        with patch('builtins.print'):
            with pytest.raises(ValueError, match="Unsupported method: PUT"):
                test_api.test_endpoint('https://api.test.com', '/test', method='PUT')

    @patch('test_api.test_endpoint')
    @patch('requests.post')
    def test_run_tests_all_pass(self, mock_post, mock_test_endpoint):
        """Test run_tests function when all tests pass"""
        # Mock all test_endpoint calls to return True
        mock_test_endpoint.return_value = True
        
        # Mock performance test requests
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        with patch('builtins.print'), patch('time.time', side_effect=[0, 1]):  # 1 second elapsed
            result = test_api.run_tests('https://api.test.com')
        
        assert result is True
        assert mock_test_endpoint.call_count >= 7  # At least 7 different test endpoints

    @patch('test_api.test_endpoint')
    def test_run_tests_some_fail(self, mock_test_endpoint):
        """Test run_tests function when some tests fail"""
        # Alternate between True and False for different tests
        mock_test_endpoint.side_effect = [True, False, True, False, True, False, True, False]
        
        with patch('builtins.print'), \
             patch('requests.post') as mock_post, \
             patch('time.time', side_effect=[0, 1]):
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            result = test_api.run_tests('https://api.test.com')
        
        assert result is False

    @patch('requests.post')
    def test_performance_test_success(self, mock_post):
        """Test performance testing functionality"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        with patch('test_api.test_endpoint') as mock_test_endpoint:
            mock_test_endpoint.return_value = True
            
            with patch('builtins.print'), \
                 patch('time.time', side_effect=[0, 0.5]):  # 0.5 seconds for 100 requests
                
                result = test_api.run_tests('https://api.test.com')
        
        # Should pass performance test (< 2 seconds average)
        assert mock_post.call_count == 100

    @patch('requests.post')
    def test_performance_test_failure(self, mock_post):
        """Test performance test when requests are slow"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        with patch('test_api.test_endpoint') as mock_test_endpoint:
            mock_test_endpoint.return_value = True
            
            with patch('builtins.print'), \
                 patch('time.time', side_effect=[0, 250]):  # 250 seconds for 100 requests (very slow)
                
                result = test_api.run_tests('https://api.test.com')
        
        # Should fail overall due to slow performance
        assert result is False

class TestAPITestingMain:
    """Test suite for main execution logic"""
    
    def test_main_functionality_exists(self):
        """Test that the test_api module has the expected functions"""
        # Just verify the module has the required functions
        assert hasattr(test_api, 'test_endpoint')
        assert hasattr(test_api, 'run_tests')
        assert hasattr(test_api, 'print_colored')
        
        # Test that functions can be called with proper arguments
        assert callable(test_api.test_endpoint)
        assert callable(test_api.run_tests)
        assert callable(test_api.print_colored)

class TestAPIEndpointValidation:
    """Test validation of different API endpoint scenarios"""
    
    @patch('requests.post')
    def test_predict_endpoint_validation(self, mock_post):
        """Test prediction endpoint with various input validations"""
        # Test successful prediction
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {
            'prediction': 350000.0,
            'unit': 'USD',
            'model_version': '1.0.0'
        }
        mock_post.return_value = mock_response
        
        sample_features = [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/predict', 
                                          method='POST', data={'features': sample_features})
        
        assert result is True

    @patch('requests.post')
    def test_batch_predict_endpoint_validation(self, mock_post):
        """Test batch prediction endpoint"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {
            'predictions': [
                {'id': 'house_1', 'prediction': 350000.0, 'unit': 'USD'},
                {'id': 'house_2', 'prediction': 275000.0, 'unit': 'USD'}
            ],
            'total': 2,
            'model_version': '1.0.0'
        }
        mock_post.return_value = mock_response
        
        batch_data = {
            'batch': [
                {'id': 'house_1', 'features': [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]},
                {'id': 'house_2', 'features': [5.2, 15.0, 6.2, 1.0, 2500.0, 2.8, 37.77, -122.42]}
            ]
        }
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/batch_predict', 
                                          method='POST', data=batch_data)
        
        assert result is True

    @patch('requests.post')
    def test_error_handling_endpoint(self, mock_post):
        """Test error handling with invalid data"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {
            'error': 'Expected 8 features, got 2',
            'expected_features': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                                 'Population', 'AveOccup', 'Latitude', 'Longitude']
        }
        mock_post.return_value = mock_response
        
        invalid_data = {'features': [1.0, 2.0]}  # Too few features
        
        with patch('builtins.print'):
            result = test_api.test_endpoint('https://api.test.com', '/predict', 
                                          method='POST', data=invalid_data, expected_status=400)
        
        assert result is True