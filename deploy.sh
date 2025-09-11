#!/bin/bash

# Housing Price API - OpenShift Deployment Script
# This script automates the complete deployment process

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="farisamz71-dev"
APP_NAME="housing-price-api"
REGISTRY="image-registry.openshift-image-registry.svc:5000"

echo -e "${GREEN}🚀 Starting Housing Price API Deployment to OpenShift${NC}"
echo "=================================================="

# Step 1: Check if logged into OpenShift
echo -e "\n${YELLOW}Step 1: Checking OpenShift connection...${NC}"
if ! oc whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged into OpenShift. Please run: oc login <cluster-url>${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Connected to OpenShift as $(oc whoami)${NC}"

# Step 2: Create namespace if it doesn't exist
echo -e "\n${YELLOW}Step 2: Setting up namespace...${NC}"
if oc get namespace $NAMESPACE &> /dev/null; then
    echo "Namespace $NAMESPACE already exists"
else
    oc new-project $NAMESPACE --description="ML Workshop - Housing Price Prediction" \
        --display-name="ML Workshop"
    echo -e "${GREEN}✅ Created namespace: $NAMESPACE${NC}"
fi
oc project $NAMESPACE

# # Step 3: Create PVC for model storage
# echo -e "\n${YELLOW}Step 3: Creating persistent storage...${NC}"
# oc apply -f openshift/pvc.yaml
# echo -e "${GREEN}✅ PVC created/updated${NC}"

# Step 4: Create ConfigMap
echo -e "\n${YELLOW}Step 4: Creating ConfigMap...${NC}"
oc apply -f openshift/configmap.yaml
echo -e "${GREEN}✅ ConfigMap created/updated${NC}"

# Step 5: Build the container image
echo -e "\n${YELLOW}Step 5: Building container image...${NC}"
echo "This may take 5-10 minutes for the first build..."

# Option A: Build from local directory (if you have source code locally)
if [ -f "Dockerfile" ]; then
    echo "Building from local source..."
    # Create a binary build
    oc new-build --name=$APP_NAME --binary --strategy=docker || true
    # Start the build
    oc start-build $APP_NAME --from-dir=. --follow
else
    # Option B: Build from Git repository
    echo "Building from Git repository..."
    oc apply -f openshift/buildconfig.yaml
    oc start-build $APP_NAME --follow
fi
# Prompt user for a build number to wait for, or use latest
read -p "Enter build number to wait for (or press Enter to use latest): " BUILD_NUM
if [ -z "$BUILD_NUM" ]; then
    BUILD_NUM=$(oc get builds -o jsonpath="{.items[-1:].metadata.name}" | sed "s/.*-\([0-9]*\)$/\1/")
fi
BUILD_NAME="$APP_NAME-$BUILD_NUM"
# Wait for build to complete
echo -e "\n${YELLOW}Waiting for build to complete...${NC}"
oc wait --for=condition=Complete build/$BUILD_NAME --timeout=600s
echo -e "${GREEN}✅ Build completed successfully${NC}"

# Step 6: Deploy the application
echo -e "\n${YELLOW}Step 6: Deploying application...${NC}"
oc apply -f openshift/deployment.yaml
echo -e "${GREEN}✅ Deployment created/updated${NC}"

# Step 7: Create HPA for autoscaling
echo -e "\n${YELLOW}Step 7: Setting up autoscaling...${NC}"
oc apply -f openshift/hpa.yaml
echo -e "${GREEN}✅ HorizontalPodAutoscaler created/updated${NC}"

# Step 8: Wait for deployment to be ready
echo -e "\n${YELLOW}Step 8: Waiting for pods to be ready...${NC}"
oc rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=300s
echo -e "${GREEN}✅ Deployment is ready${NC}"

# Step 9: Get the route URL
echo -e "\n${YELLOW}Step 9: Getting application URL...${NC}"
ROUTE_URL=$(oc get route $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.host}')
echo -e "${GREEN}✅ Application deployed successfully!${NC}"

# Step 10: Display access information
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nApplication URL: ${GREEN}https://$ROUTE_URL${NC}"
echo -e "\nEndpoints:"
echo -e "  - API Docs:        https://$ROUTE_URL/"
echo -e "  - Health Check:    https://$ROUTE_URL/health"
echo -e "  - Metrics:         https://$ROUTE_URL/metrics"
echo -e "  - Predict:         https://$ROUTE_URL/predict"
echo -e "  - Batch Predict:   https://$ROUTE_URL/batch_predict"

# Step 11: Test the deployment
echo -e "\n${YELLOW}Step 11: Testing the deployment...${NC}"
echo "Waiting 10 seconds for service to stabilize..."
sleep 10

# Test health endpoint
echo -e "\nTesting health endpoint..."
if curl -s -o /dev/null -w "%{http_code}" https://$ROUTE_URL/health | grep -q "200"; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}⚠️  Health check failed. The service might need more time to start.${NC}"
fi

# Display monitoring commands
echo -e "\n${YELLOW}Useful commands for monitoring:${NC}"
echo "  oc get pods -n $NAMESPACE"
echo "  oc logs -f deployment/$APP_NAME -n $NAMESPACE"
echo "  oc describe hpa $APP_NAME-hpa -n $NAMESPACE"
echo "  oc get events -n $NAMESPACE --sort-by='.lastTimestamp'"

echo -e "\n${GREEN}Deployment script completed!${NC}"