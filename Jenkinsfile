pipeline {
    agent any
    
    environment {
        OPENSHIFT_PROJECT = 'farisamz71-dev'
        APP_NAME = 'housing-price-api'
        IMAGE_TAG = "${BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies & Test') {
            steps {
                sh '''
                    python -m venv venv
                    source venv/bin/activate
                    pip install -r requirements.txt
                    
                    # Run tests
                    python -m pytest tests/ -v --junitxml=test-results.xml
                    
                    # Code quality checks
                    flake8 app.py
                    pylint app.py
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }
        
        stage('Model Validation') {
            steps {
                sh '''
                    source venv/bin/activate
                    python validate_model.py
                '''
            }
        }
        
        stage('Build Container') {
            steps {
                script {
                    openshift.withCluster() {
                        openshift.withProject("${OPENSHIFT_PROJECT}") {
                            openshift.selector("bc", "${APP_NAME}").startBuild("--from-dir=.", "--wait=true")
                        }
                    }
                }
            }
        }
        
        stage('Deploy to Dev') {
            steps {
                script {
                    openshift.withCluster() {
                        openshift.withProject("${OPENSHIFT_PROJECT}") {
                            openshift.selector("dc", "${APP_NAME}").rollout().latest()
                        }
                    }
                }
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh '''
                    API_URL=$(oc get route ${APP_NAME} -o jsonpath='{.spec.host}')
                    python test_api.py $API_URL
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Production?', ok: 'Deploy'
                // Production deployment steps
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}