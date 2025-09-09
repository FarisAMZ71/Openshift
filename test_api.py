#!/usr/bin/env python3
"""
Test script for Housing Price Prediction API
Tests all endpoints with sample data
"""

import requests
import json
import sys
import time
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_colored(message, color):
    """Print colored output"""
    print(f"{color}{message}{RESET}")

def test_endpoint(url, endpoint, method="GET", data=None, expected_status=200):
    """Test a single endpoint"""
    full_url = f"{url}{endpoint}"
    print(f"\nTesting: {method} {endpoint}")
    print("-" * 50)
    
    try:
        if method == "GET":
            response = requests.get(full_url)
        elif method == "POST":
            response = requests.post(full_url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Check status code
        if response.status_code == expected_status:
            print_colored(f"✅ Status Code: {response.status_code}", GREEN)
        else:
            print_colored(f"❌ Status Code: {response.status_code} (expected {expected_status})", RED)
            
        # Print response
        if response.headers.get('content-type', '').startswith('application/json'):
            response_data = response.json()
            print(f"Response: {json.dumps(response_data, indent=2)}")
        else:
            print(f"Response: {response.text[:200]}...")
            
        return response.status_code == expected_status
        
    except requests.exceptions.RequestException as e:
        print_colored(f"❌ Request failed: {e}", RED)
        return False
    except json.JSONDecodeError as e:
        print_colored(f"❌ Invalid JSON response: {e}", RED)
        print(f"Raw response: {response.text[:200]}...")
        return False

def run_tests(base_url):
    """Run all API tests"""
    print_colored("\n🧪 Housing Price Prediction API - Test Suite", BLUE)
    print("=" * 60)
    
    # Remove trailing slash if present
    base_url = base_url.rstrip('/')
    
    all_tests_passed = True
    test_results = []
    
    # Test 1: Root endpoint
    print_colored("\n📍 Test 1: API Documentation", YELLOW)
    result = test_endpoint(base_url, "/")
    test_results.append(("API Documentation", result))
    
    # Test 2: Health check
    print_colored("\n📍 Test 2: Health Check", YELLOW)
    result = test_endpoint(base_url, "/health")
    test_results.append(("Health Check", result))
    
    # Test 3: Readiness check
    print_colored("\n📍 Test 3: Readiness Check", YELLOW)
    result = test_endpoint(base_url, "/ready")
    test_results.append(("Readiness Check", result))
    
    # Test 4: Metrics endpoint
    print_colored("\n📍 Test 4: Metrics", YELLOW)
    result = test_endpoint(base_url, "/metrics")
    test_results.append(("Metrics", result))
    
    # Test 5: Single prediction
    print_colored("\n📍 Test 5: Single Prediction", YELLOW)
    sample_data = {
        "features": [
            4.526,    # MedInc - Median income
            28.0,     # HouseAge - House age
            5.118,    # AveRooms - Average rooms
            1.073,    # AveBedrms - Average bedrooms  
            558.0,    # Population
            2.547,    # AveOccup - Average occupancy
            33.49,    # Latitude
            -117.16   # Longitude
        ]
    }
    result = test_endpoint(base_url, "/predict", method="POST", data=sample_data)
    test_results.append(("Single Prediction", result))
    
    # Test 6: Batch prediction
    print_colored("\n📍 Test 6: Batch Prediction", YELLOW)
    batch_data = {
        "batch": [
            {
                "id": "house_1",
                "features": [3.8, 25.0, 4.5, 1.1, 1200.0, 3.0, 34.05, -118.25]
            },
            {
                "id": "house_2", 
                "features": [5.2, 15.0, 6.2, 1.0, 2500.0, 2.8, 37.77, -122.42]
            },
            {
                "id": "house_3",
                "features": [2.5, 35.0, 3.9, 1.2, 800.0, 2.5, 32.72, -117.16]
            }
        ]
    }
    result = test_endpoint(base_url, "/batch_predict", method="POST", data=batch_data)
    test_results.append(("Batch Prediction", result))
    
    # Test 7: Invalid request
    print_colored("\n📍 Test 7: Error Handling - Invalid Features", YELLOW)
    invalid_data = {
        "features": [1.0, 2.0]  # Too few features
    }
    result = test_endpoint(base_url, "/predict", method="POST", data=invalid_data, expected_status=400)
    test_results.append(("Error Handling", result))
    
    # Test 8: Performance test
    print_colored("\n📍 Test 8: Performance Test", YELLOW)
    print("Making 10 rapid requests...")
    start_time = time.time()
    success_count = 0
    
    for i in range(10):
        try:
            response = requests.post(f"{base_url}/predict", json=sample_data, timeout=5)
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / 10
    
    print(f"Successful requests: {success_count}/10")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average response time: {avg_time:.3f} seconds")
    
    if success_count == 10 and avg_time < 2.0:
        print_colored("✅ Performance test passed", GREEN)
        test_results.append(("Performance", True))
    else:
        print_colored("❌ Performance test failed", RED)
        test_results.append(("Performance", False))
    
    # Summary
    print_colored("\n📊 Test Summary", BLUE)
    print("=" * 60)
    
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        color = GREEN if passed else RED
        print_colored(f"{test_name:.<30} {status}", color)
    
    passed_count = sum(1 for _, passed in test_results if passed)
    total_count = len(test_results)
    
    print("\n" + "=" * 60)
    if passed_count == total_count:
        print_colored(f"🎉 All tests passed! ({passed_count}/{total_count})", GREEN)
    else:
        print_colored(f"⚠️  {passed_count}/{total_count} tests passed", YELLOW)
        all_tests_passed = False
    
    return all_tests_passed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <API_URL>")
        print(f"Example: {sys.argv[0]} https://housing-api.apps.cluster.example.com")
        sys.exit(1)
    
    api_url = sys.argv[1]
    
    # Ensure URL has https:// prefix
    if not api_url.startswith(('http://', 'https://')):
        api_url = f'https://{api_url}'
    
    print(f"Testing API at: {api_url}")
    
    # Run tests
    success = run_tests(api_url)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)