# Security Configuration Guide

## Jenkins Credentials Setup

To securely use this pipeline, you need to configure the following credentials in Jenkins:

### Required Jenkins Credentials

1. **OpenShift Token** (String Credential)
   - **ID**: `openshift-token`
   - **Description**: OpenShift cluster authentication token
   - **Value**: Your OpenShift service account token
   - **How to get**: 
     ```bash
     oc whoami -t
     # or create a service account:
     oc create sa jenkins-deployer
     oc policy add-role-to-user edit system:serviceaccount:farisamz71-dev:jenkins-deployer
     oc sa get-token jenkins-deployer
     ```

2. **OpenShift Server URL** (String Credential)
   - **ID**: `openshift-server-url`
   - **Description**: OpenShift cluster server URL
   - **Value**: Your OpenShift cluster URL (e.g., `https://api.cluster-name.domain.com:6443`)
   - **How to get**: `oc cluster-info | grep 'Kubernetes control plane'`

### Setting up Jenkins Credentials

#### Via Jenkins UI:
1. Go to Jenkins Dashboard → Manage Jenkins → Credentials
2. Select the appropriate domain (usually Global)
3. Click "Add Credentials"
4. Choose "Secret text" for both credentials
5. Enter the ID, description, and secret value
6. Save

#### Via Jenkins CLI:
```bash
# Create OpenShift token credential
echo "YOUR_OPENSHIFT_TOKEN" | jenkins-cli create-credentials-by-xml system::system::jenkins < <(cat <<EOF
<com.cloudbees.plugins.credentials.impl.StringCredentialsImpl>
  <scope>GLOBAL</scope>
  <id>openshift-token</id>
  <description>OpenShift Authentication Token</description>
  <secret>$(cat)</secret>
</com.cloudbees.plugins.credentials.impl.StringCredentialsImpl>
EOF
)
```

## Alternative Authentication Methods

### 1. Service Account with Kubeconfig
Instead of tokens, you can use a kubeconfig file:

```groovy
withCredentials([file(credentialsId: 'openshift-kubeconfig', variable: 'KUBECONFIG')]) {
    sh 'oc get projects'
}
```

### 2. OpenShift OAuth Integration
For enterprise setups, integrate Jenkins with OpenShift OAuth:
- Configure OpenShift OAuth client
- Use OpenShift Jenkins image with built-in integration

## Environment-Specific Configuration

### Development Environment
```groovy
environment {
    OPENSHIFT_PROJECT = 'farisamz71-dev'
    // Other dev-specific settings
}
```

### Production Environment
```groovy
environment {
    OPENSHIFT_PROJECT = 'farisamz71-prod'
    // Additional security measures for prod
}
```

## Security Best Practices

### 1. Principle of Least Privilege
- Create dedicated service accounts for CI/CD
- Grant only necessary permissions
- Use project-scoped roles, not cluster-admin

### 2. Token Rotation
- Regularly rotate OpenShift tokens
- Set up automated token renewal if possible
- Monitor token usage in OpenShift

### 3. Audit and Monitoring
- Enable Jenkins audit logging
- Monitor OpenShift access logs
- Set up alerts for failed authentications

### 4. Branch Protection
- The pipeline only deploys from protected branches
- Require PR reviews before merging to main
- Use branch policies to prevent direct pushes

## Troubleshooting

### Common Issues:

1. **Authentication Failed**
   ```
   Error: error logging in: invalid token
   ```
   - Check if token is expired
   - Verify token has correct permissions
   - Ensure server URL is correct

2. **Permission Denied**
   ```
   Error: cannot create resource "deployments"
   ```
   - Service account needs 'edit' role in the project
   - Check RBAC permissions

3. **Project Not Found**
   ```
   Error: project "farisamz71-dev" not found
   ```
   - Verify project name
   - Check if service account has access to project

### Getting Help:
- Check OpenShift console for service account permissions
- Review Jenkins console logs for detailed error messages
- Test OpenShift CLI commands manually first

## Never Commit These:
- ❌ OpenShift tokens
- ❌ Server URLs (if they contain sensitive info)
- ❌ Service account keys
- ❌ Passwords or API keys
- ❌ Environment-specific secrets

## Safe to Commit:
- ✅ Project names (if not sensitive)
- ✅ Application names
- ✅ Jenkins credential IDs (references only)
- ✅ Build configurations
- ✅ This documentation