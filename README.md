# Housing Price API - Quick Reference Guide

## 🚀 Quick Start (Complete Deployment)

```bash
# 1. Login to OpenShift
oc login <your-cluster-url>

# 2. Run the deployment script
chmod +x deploy.sh
./deploy.sh

# 3. Test the API
python test_api.py $(oc get route housing-price-api -n farisamz71-dev -o jsonpath='{.spec.host}')
```

## 📝 Step-by-Step Manual Deployment

### Initial Setup
```bash
# Create namespace
oc new-project farisamz71-dev

# Set current project
oc project farisamz71-dev
```

### Build Container Image
```bash
# Option 1: From local source
oc new-build --name=housing-price-api --binary --strategy=docker
oc start-build housing-price-api --from-dir=. --follow

# Option 2: From Git repository
oc new-app https://github.com/your-repo/housing-price-api.git \
  --name=housing-price-api \
  --strategy=docker
```

### Deploy Resources
```bash
# Apply all configurations
oc apply -f openshift/pvc.yaml
oc apply -f openshift/configmap.yaml
oc apply -f openshift/deployment.yaml
oc apply -f openshift/hpa.yaml
oc apply -f openshift/monitoring-dashboard.yaml
```

### Check Deployment Status
```bash
# View pods
oc get pods -n farisamz71-dev

# Check deployment
oc rollout status deployment/housing-price-api

# View logs
oc logs -f deployment/housing-price-api

# Get route URL
oc get route housing-price-api -o jsonpath='{.spec.host}'
```

## 🧪 Testing the API

### Using cURL
```bash
# Get API URL
API_URL=https://$(oc get route housing-price-api -o jsonpath='{.spec.host}')

# Health check
curl -X GET $API_URL/health

# Single prediction
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [3.8, 25.0, 4.5, 1.1, 1200.0, 3.0, 34.05, -118.25]
  }'

# Batch prediction
curl -X POST $API_URL/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "batch": [
      {"id": "1", "features": [3.8, 25.0, 4.5, 1.1, 1200.0, 3.0, 34.05, -118.25]},
      {"id": "2", "features": [5.2, 15.0, 6.2, 1.0, 2500.0, 2.8, 37.77, -122.42]}
    ]
  }'
```

### Using Python
```python
import requests

api_url = "https://housing-price-api-farisamz71-dev.apps.cluster.example.com"
data = {
    "features": [4.526, 28.0, 5.118, 1.073, 558.0, 2.547, 33.49, -117.16]
}
response = requests.post(f"{api_url}/predict", json=data)
print(response.json())
```

## 🔍 Monitoring & Troubleshooting

### View Metrics
```bash
# CPU and Memory usage
oc adm top pods -n farisamz71-dev

# HPA status
oc describe hpa housing-price-api-hpa

# Events
oc get events -n farisamz71-dev --sort-by='.lastTimestamp'
```

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Pods in Pending state | Check PVC: `oc describe pvc model-storage-pvc` |
| Pods in CrashLoopBackOff | Check logs: `oc logs <pod-name> --previous` |
| Route not accessible | Verify route: `oc describe route housing-price-api` |
| High response time | Scale up: `oc scale deployment housing-price-api --replicas=5` |
| Build fails | Check build logs: `oc logs build/housing-price-api-1` |

### Scaling Operations
```bash
# Manual scaling
oc scale deployment housing-price-api --replicas=5

# Edit HPA
oc edit hpa housing-price-api-hpa

# Pause autoscaling
oc patch hpa housing-price-api-hpa -p '{"spec":{"minReplicas":3,"maxReplicas":3}}'
```

## 🔄 Update and Rollback

### Update Application
```bash
# Trigger new build
oc start-build housing-price-api --follow

# Force redeployment
oc rollout restart deployment/housing-price-api

# Watch rollout
oc rollout status deployment/housing-price-api --watch
```

### Rollback
```bash
# View rollout history
oc rollout history deployment/housing-price-api

# Rollback to previous version
oc rollout undo deployment/housing-price-api

# Rollback to specific revision
oc rollout undo deployment/housing-price-api --to-revision=2
```

## 🧹 Cleanup

```bash
# Delete all resources
oc delete project farisamz71-dev

# Or delete specific resources
oc delete deployment,service,route,pvc,configmap,hpa -l app=housing-price-api
```

## 📊 Feature Reference

| Feature | Description | Expected Range |
|---------|-------------|----------------|
| MedInc | Median income in block group | 0.5 - 15.0 |
| HouseAge | Median house age in block group | 1.0 - 52.0 |
| AveRooms | Average number of rooms | 1.0 - 40.0 |
| AveBedrms | Average number of bedrooms | 0.5 - 5.0 |
| Population | Block group population | 3.0 - 35,000 |
| AveOccup | Average house occupancy | 1.0 - 10.0 |
| Latitude | Block group latitude | 32.0 - 42.0 |
| Longitude | Block group longitude | -124.0 to -114.0 |

## 🔗 Useful Links

- [OpenShift Documentation](https://docs.openshift.com)
- [TensorFlow Serving Guide](https://www.tensorflow.org/tfx/guide/serving)
- [Flask API Documentation](https://flask.palletsprojects.com/)
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

# 🔐 Security Quick Reference

## ✅ What's Now Safe for Public Repository

Your project is now configured to use Jenkins credentials securely. Here's what you can safely commit:

### Safe Files ✅
- `Jenkinsfile` - Uses credential references, no actual secrets
- `SECURITY.md` - Security documentation and best practices  
- `jenkins-config-template.yml` - Template with examples, no real credentials
- `.gitignore` - Prevents accidental commits of sensitive files
- All OpenShift YAML files - Configuration only, no secrets
- Application code and tests

### Never Commit ❌
- Actual OpenShift tokens
- Server URLs (if they contain sensitive cluster info)
- Service account keys or certificates
- Any files ending in `.token`, `.key`, `.pem`
- Environment files with real values (`.env`, `.env.production`)

## 🚀 Setup Steps

### 1. Repository Setup (Already Done)
```bash
git add .
git commit -m "Add secure Jenkins pipeline with credential management"
git push origin main
```

### 2. Jenkins Credentials Setup (You Need to Do)

#### Get Your OpenShift Info:
```bash
# Get your token
oc whoami -t

# Get your server URL
oc cluster-info | head -1
```

#### Create Jenkins Credentials:
1. Go to Jenkins → Manage Jenkins → Credentials
2. Add these two string credentials:
   - **ID**: `openshift-token` → **Value**: [your token from above]
   - **ID**: `openshift-server-url` → **Value**: [your server URL from above]

### 3. Optional: Create Dedicated Service Account
```bash
# More secure approach
oc create sa jenkins-deployer -n farisamz71-dev
oc policy add-role-to-user edit system:serviceaccount:farisamz71-dev:jenkins-deployer -n farisamz71-dev
oc sa get-token jenkins-deployer -n farisamz71-dev
# Use this token instead of your personal token
```

## 🔍 Security Features Implemented

1. **Jenkins Credentials Integration**: Uses `withCredentials()` block
2. **No Hardcoded Secrets**: All sensitive data referenced by ID only
3. **Comprehensive .gitignore**: Prevents accidental commits
4. **Branch Protection**: Only deploys from main/master branches
5. **Documentation**: Clear security guidelines
6. **Environment Separation**: Template for multiple environments

## 🧪 Testing Your Setup

### Local Test (Before Pushing):
```bash
# Check for any accidentally committed secrets
git log --patch | grep -i "token\|password\|secret" || echo "✅ No secrets found"

# Verify .gitignore is working
echo "test-token" > test.token
git add . 
git status | grep "test.token" && echo "❌ .gitignore not working" || echo "✅ .gitignore working"
rm test.token
```

### Jenkins Test:
1. Create a test branch
2. Push some changes
3. Check if pipeline runs without credential errors

## 🆘 Troubleshooting

If you get authentication errors in Jenkins:
1. Verify credential IDs match exactly: `openshift-token`, `openshift-server-url`
2. Test OpenShift CLI access manually with the same token
3. Check Jenkins logs for detailed error messages
4. Ensure service account has proper permissions

Your repository is now secure for public sharing! 🎉