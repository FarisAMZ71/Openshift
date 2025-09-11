pipeline {
    agent any
    
    environment {
        // OpenShift Configuration
        OPENSHIFT_PROJECT = 'farisamz71-dev'
        APP_NAME = 'housing-price-api'
        IMAGE_TAG = "${BUILD_NUMBER}"
        
        // Git Configuration
        GIT_BRANCH = "${env.BRANCH_NAME ?: 'main'}"
        
        // Test Configuration
        PYTEST_ARGS = '--verbose --tb=short --junitxml=test-results.xml --cov=application --cov-report=xml --cov-report=html'
        
        // Model Validation Thresholds
        MIN_MODEL_ACCURACY = '0.80'
        MAX_ACCEPTABLE_MAE = '50000'
    }
    
    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo "🔄 Checking out code from ${GIT_BRANCH} branch"
                checkout scm
                
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                    env.BUILD_VERSION = "${IMAGE_TAG}-${GIT_COMMIT_SHORT}"
                }
                
                echo "📋 Build Info:"
                echo "  - Branch: ${GIT_BRANCH}"
                echo "  - Commit: ${env.GIT_COMMIT_SHORT}"
                echo "  - Build Version: ${env.BUILD_VERSION}"
            }
        }
        
        stage('Setup Python Environment') {
            steps {
                echo "🐍 Setting up Python environment"
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    python --version
                    pip list | grep -E "(pytest|flask|scikit-learn|xgboost)"
                '''
            }
        }
        
        stage('Code Quality & Linting') {
            parallel {
                stage('Flake8 Linting') {
                    steps {
                        echo "🔍 Running Flake8 linting"
                        sh '''
                            . venv/bin/activate
                            flake8 application/ tests/ --max-line-length=120 --exclude=__pycache__ \
                                --format='%(path)s:%(row)d:%(col)d: %(code)s %(text)s' \
                                --output-file=flake8-report.txt || true
                            if [ -s flake8-report.txt ]; then
                                echo "⚠️  Flake8 issues found:"
                                cat flake8-report.txt
                            else
                                echo "✅ No Flake8 issues found"
                            fi
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'flake8-report.txt', allowEmptyArchive: true
                        }
                    }
                }
                
                stage('Pylint Analysis') {
                    steps {
                        echo "📊 Running Pylint analysis"
                        sh '''
                            . venv/bin/activate
                            pylint application/ --output-format=text --reports=yes \
                                --output=pylint-report.txt --exit-zero
                            echo "📈 Pylint analysis completed"
                            tail -10 pylint-report.txt || echo "No pylint report generated"
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'pylint-report.txt', allowEmptyArchive: true
                        }
                    }
                }
            }
        }
        
        stage('Run Tests') {
            steps {
                echo "🧪 Running pytest with coverage"
                sh '''
                    . venv/bin/activate
                    
                    # Create reports directory if it doesn't exist
                    mkdir -p reports
                    
                    # Run pytest with coverage and detailed output
                    pytest ${PYTEST_ARGS} tests/ application/
                    
                    # Display test results summary
                    if [ -f test-results.xml ]; then
                        echo "✅ Test results generated successfully"
                        python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('test-results.xml')
root = tree.getroot()
tests = int(root.get('tests', 0))
failures = int(root.get('failures', 0))
errors = int(root.get('errors', 0))
skipped = int(root.get('skipped', 0))
passed = tests - failures - errors - skipped
print(f'📊 Test Summary:')
print(f'  ✅ Passed: {passed}')
print(f'  ❌ Failed: {failures}')
print(f'  🚨 Errors: {errors}')
print(f'  ⏭️  Skipped: {skipped}')
print(f'  📈 Total: {tests}')
if failures > 0 or errors > 0:
    exit(1)
"
                    else
                        echo "❌ No test results file generated"
                        exit 1
                    fi
                '''
            }
            post {
                always {
                    // Publish test results
                    publishTestResults testResultsPattern: 'test-results.xml'
                    
                    // Publish coverage reports
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report',
                        reportTitles: 'Code Coverage'
                    ])
                    
                    // Archive artifacts
                    archiveArtifacts artifacts: 'test-results.xml,coverage.xml,htmlcov/**', allowEmptyArchive: true
                }
            }
        }
        
        stage('Model Validation') {
            steps {
                echo "🤖 Training and validating ML model"
                sh '''
                    . venv/bin/activate
                    
                    # Run model training and validation
                    python application/train.py
                    
                    # Validate model artifacts exist
                    if [ ! -f models/housing_model.json ]; then
                        echo "❌ Model file not found!"
                        exit 1
                    fi
                    
                    if [ ! -f models/scaler.pkl ]; then
                        echo "❌ Scaler file not found!"
                        exit 1
                    fi
                    
                    if [ ! -f models/metadata.json ]; then
                        echo "❌ Metadata file not found!"
                        exit 1
                    fi
                    
                    echo "✅ All model artifacts created successfully"
                    
                    # Validate model performance
                    python -c "
import json
import sys

try:
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print('📊 Model Performance Metrics:')
    for metric, value in metadata.get('performance_metrics', {}).items():
        print(f'  {metric}: {value:.4f}')
    
    # Check model accuracy/performance thresholds
    r2_score = metadata.get('performance_metrics', {}).get('r2_score', 0)
    mae = metadata.get('performance_metrics', {}).get('mae', float('inf'))
    
    print(f'\\n🎯 Validation Thresholds:')
    print(f'  Minimum R² Score: ${MIN_MODEL_ACCURACY}')
    print(f'  Maximum MAE: ${MAX_ACCEPTABLE_MAE}')
    print(f'  Current R² Score: {r2_score:.4f}')
    print(f'  Current MAE: {mae:.2f}')
    
    if r2_score < float('${MIN_MODEL_ACCURACY}'):
        print(f'❌ Model R² score {r2_score:.4f} below threshold ${MIN_MODEL_ACCURACY}')
        sys.exit(1)
    
    if mae > float('${MAX_ACCEPTABLE_MAE}'):
        print(f'❌ Model MAE {mae:.2f} above threshold ${MAX_ACCEPTABLE_MAE}')
        sys.exit(1)
    
    print('✅ Model validation passed!')
    
except Exception as e:
    print(f'❌ Error validating model: {e}')
    sys.exit(1)
"
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'models/**', allowEmptyArchive: true
                }
            }
        }
        
        stage('Build Container Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                    branch 'develop'
                }
            }
            steps {
                script {
                    echo "🐳 Building container image"
                    
                    // Login to OpenShift
                    sh '''
                        oc login --token=${OPENSHIFT_TOKEN} --server=${OPENSHIFT_SERVER} --insecure-skip-tls-verify=true
                        oc project ${OPENSHIFT_PROJECT}
                    '''
                    
                    // Start build from current directory
                    sh '''
                        echo "Starting OpenShift build..."
                        oc start-build ${APP_NAME} --from-dir=. --follow --wait
                        
                        # Tag the image with build version
                        oc tag ${APP_NAME}:latest ${APP_NAME}:${BUILD_VERSION}
                        
                        echo "✅ Container image built and tagged successfully"
                        echo "   Image: ${APP_NAME}:${BUILD_VERSION}"
                    '''
                }
            }
        }
        
        stage('Security Scan') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                    branch 'develop'
                }
            }
            steps {
                echo "🔒 Running security scan on container image"
                sh '''
                    # Run OpenShift image scan (if available)
                    oc describe is/${APP_NAME} | grep -A 10 "Image Vulnerabilities" || echo "No vulnerability scan results available"
                    
                    # Additional security checks could go here
                    echo "✅ Security scan completed"
                '''
            }
        }
        
        stage('Deploy to OpenShift') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                }
            }
            steps {
                script {
                    echo "🚀 Deploying to OpenShift"
                    
                    try {
                        // Apply OpenShift configurations
                        sh '''
                            echo "Applying OpenShift configurations..."
                            
                            # Apply all configurations
                            oc apply -f openshift/ -n ${OPENSHIFT_PROJECT}
                            
                            # Update deployment with new image
                            oc set image deployment/${APP_NAME} api=image-registry.openshift-image-registry.svc:5000/${OPENSHIFT_PROJECT}/${APP_NAME}:${BUILD_VERSION} -n ${OPENSHIFT_PROJECT}
                            
                            # Wait for rollout to complete
                            echo "Waiting for deployment rollout..."
                            oc rollout status deployment/${APP_NAME} -n ${OPENSHIFT_PROJECT} --timeout=300s
                            
                            # Verify deployment
                            oc get pods -l app=${APP_NAME} -n ${OPENSHIFT_PROJECT}
                            
                            echo "✅ Deployment completed successfully"
                        '''
                        
                        // Get application URL
                        script {
                            def route = sh(
                                script: "oc get route ${APP_NAME} -n ${OPENSHIFT_PROJECT} -o jsonpath='{.spec.host}'",
                                returnStdout: true
                            ).trim()
                            
                            if (route) {
                                env.APP_URL = "https://${route}"
                                echo "🌐 Application URL: ${env.APP_URL}"
                            }
                        }
                        
                    } catch (Exception e) {
                        echo "❌ Deployment failed: ${e.getMessage()}"
                        
                        // Rollback on failure
                        sh '''
                            echo "🔄 Rolling back deployment..."
                            oc rollout undo deployment/${APP_NAME} -n ${OPENSHIFT_PROJECT}
                            oc rollout status deployment/${APP_NAME} -n ${OPENSHIFT_PROJECT} --timeout=300s
                        '''
                        
                        throw e
                    }
                }
            }
        }
        
        stage('Post-Deployment Tests') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                }
            }
            steps {
                script {
                    if (env.APP_URL) {
                        echo "🧪 Running post-deployment health checks"
                        sh '''
                            # Wait a bit for the application to be fully ready
                            sleep 30
                            
                            # Health check
                            echo "Checking application health..."
                            curl -f -s ${APP_URL}/health || (echo "❌ Health check failed" && exit 1)
                            echo "✅ Health check passed"
                            
                            # Readiness check
                            echo "Checking application readiness..."
                            curl -f -s ${APP_URL}/ready || (echo "❌ Readiness check failed" && exit 1)
                            echo "✅ Readiness check passed"
                            
                            # Basic API functionality test
                            echo "Testing API functionality..."
                            curl -f -s -X POST ${APP_URL}/predict \\
                                -H "Content-Type: application/json" \\
                                -d '{"features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}' || (echo "❌ API test failed" && exit 1)
                            echo "✅ API functionality test passed"
                            
                            echo "🎉 All post-deployment tests passed!"
                        '''
                    } else {
                        echo "⚠️ No application URL available, skipping post-deployment tests"
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo "🧹 Cleaning up workspace"
            
            // Clean up virtual environment
            sh 'rm -rf venv || true'
            
            // Archive important artifacts
            archiveArtifacts artifacts: '**/*.log,test-results.xml,coverage.xml', allowEmptyArchive: true
        }
        
        success {
            echo "✅ Pipeline completed successfully!"
            
            script {
                def message = """
🎉 **Deployment Successful!**

**Build Info:**
- Build: #${BUILD_NUMBER}
- Branch: ${GIT_BRANCH}
- Commit: ${env.GIT_COMMIT_SHORT}
- Version: ${env.BUILD_VERSION}

**Application:**
- URL: ${env.APP_URL ?: 'Not available'}
- Project: ${OPENSHIFT_PROJECT}

**Pipeline Duration:** ${currentBuild.durationString}
                """.stripIndent()
                
                echo message
                
                // You can add notification steps here (Slack, email, etc.)
                // slackSend(channel: '#deployments', message: message)
            }
        }
        
        failure {
            echo "❌ Pipeline failed!"
            
            script {
                def message = """
🚨 **Deployment Failed!**

**Build Info:**
- Build: #${BUILD_NUMBER}
- Branch: ${GIT_BRANCH}
- Stage: ${env.STAGE_NAME ?: 'Unknown'}

**Error:** Check build logs for details.

**Pipeline Duration:** ${currentBuild.durationString}
                """.stripIndent()
                
                echo message
                
                // You can add notification steps here
                // slackSend(channel: '#deployments', color: 'danger', message: message)
            }
        }
        
        unstable {
            echo "⚠️ Pipeline completed with warnings"
        }
    }
}