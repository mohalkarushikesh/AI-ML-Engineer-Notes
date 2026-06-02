# Kubernetes Cheatsheet

---

## Core Concepts

| Term | Description |
|------|-------------|
| **Pod** | Smallest unit in Kubernetes — one or more containers running together |
| **Node** | A physical or virtual machine that runs pods |
| **Cluster** | A set of nodes managed by Kubernetes |
| **Namespace** | Virtual isolation within a cluster (like folders) |
| **Deployment** | Manages replicas of a pod, handles rolling updates |
| **Service** | Exposes pods to network traffic (internal or external) |
| **ConfigMap** | Store non-secret config data (env vars, config files) |
| **Secret** | Store sensitive data (passwords, tokens) |
| **Ingress** | Manages external HTTP/HTTPS access to services |
| **Volume** | Persistent storage attached to a pod |
| **StatefulSet** | Like Deployment but for stateful apps (databases) |
| **DaemonSet** | Runs one pod per node (logging, monitoring agents) |
| **Job** | Runs a pod once to completion |
| **CronJob** | Runs a job on a schedule |

---

## kubectl — Basic Syntax

```bash
kubectl [command] [resource] [name] [flags]
```

---

## Cluster Info

| Command | Description |
|---------|-------------|
| `kubectl version` | Show kubectl and cluster version |
| `kubectl cluster-info` | Show cluster master and services |
| `kubectl get nodes` | List all nodes in the cluster |
| `kubectl get nodes -o wide` | List nodes with extra details (IP, OS, etc.) |
| `kubectl describe node <node>` | Detailed info about a node |
| `kubectl top nodes` | CPU and memory usage of nodes |

---

## Namespaces

| Command | Description |
|---------|-------------|
| `kubectl get namespaces` | List all namespaces |
| `kubectl create namespace myns` | Create a namespace |
| `kubectl delete namespace myns` | Delete a namespace |
| `kubectl config set-context --current --namespace=myns` | Set default namespace |
| `kubectl get pods -n myns` | List pods in a specific namespace |
| `kubectl get all -n myns` | List everything in a namespace |
| `kubectl get all --all-namespaces` | List everything across all namespaces |

---

## Pods

| Command | Description |
|---------|-------------|
| `kubectl get pods` | List all pods in current namespace |
| `kubectl get pods -o wide` | List pods with node and IP info |
| `kubectl get pods -A` | List pods in all namespaces |
| `kubectl describe pod <pod>` | Detailed info about a pod |
| `kubectl logs <pod>` | View pod logs |
| `kubectl logs -f <pod>` | Follow live logs |
| `kubectl logs <pod> -c <container>` | Logs from a specific container in pod |
| `kubectl exec -it <pod> -- bash` | Open shell inside a pod |
| `kubectl exec <pod> -- ls /` | Run a single command inside pod |
| `kubectl delete pod <pod>` | Delete a pod |
| `kubectl top pod <pod>` | CPU and memory usage of a pod |
| `kubectl run mypod --image=ubuntu --restart=Never` | Create a one-off pod |

---

## Deployments

| Command | Description |
|---------|-------------|
| `kubectl get deployments` | List all deployments |
| `kubectl describe deployment <name>` | Detailed info about a deployment |
| `kubectl create deployment myapp --image=nginx` | Create a deployment |
| `kubectl delete deployment myapp` | Delete a deployment |
| `kubectl scale deployment myapp --replicas=3` | Scale up or down |
| `kubectl set image deployment/myapp nginx=nginx:1.25` | Update image (rolling update) |
| `kubectl rollout status deployment/myapp` | Watch rolling update progress |
| `kubectl rollout history deployment/myapp` | View rollout history |
| `kubectl rollout undo deployment/myapp` | Rollback to previous version |
| `kubectl rollout undo deployment/myapp --to-revision=2` | Rollback to specific revision |
| `kubectl edit deployment myapp` | Edit deployment live |

---

## Services

| Command | Description |
|---------|-------------|
| `kubectl get services` | List all services |
| `kubectl describe service <name>` | Detailed info about a service |
| `kubectl expose deployment myapp --port=80 --type=ClusterIP` | Expose internally (default) |
| `kubectl expose deployment myapp --port=80 --type=NodePort` | Expose on a node port |
| `kubectl expose deployment myapp --port=80 --type=LoadBalancer` | Expose via cloud load balancer |
| `kubectl delete service myapp` | Delete a service |
| `kubectl port-forward svc/myapp 8080:80` | Forward local port to service |
| `kubectl port-forward pod/mypod 8080:80` | Forward local port to pod |

## Service Types

| Type | Description |
|------|-------------|
| `ClusterIP` | Internal only — accessible within the cluster |
| `NodePort` | Accessible on each node's IP at a static port |
| `LoadBalancer` | Exposed externally via cloud provider load balancer |
| `ExternalName` | Maps service to an external DNS name |

---

## ConfigMaps & Secrets

| Command | Description |
|---------|-------------|
| `kubectl get configmaps` | List all configmaps |
| `kubectl create configmap myconfig --from-literal=key=value` | Create configmap from value |
| `kubectl create configmap myconfig --from-file=config.txt` | Create configmap from file |
| `kubectl describe configmap myconfig` | View configmap details |
| `kubectl delete configmap myconfig` | Delete a configmap |
| `kubectl get secrets` | List all secrets |
| `kubectl create secret generic mysecret --from-literal=password=abc123` | Create a secret |
| `kubectl describe secret mysecret` | View secret details (values hidden) |
| `kubectl delete secret mysecret` | Delete a secret |

---

## Apply & Manage with YAML

| Command | Description |
|---------|-------------|
| `kubectl apply -f file.yaml` | Create or update resource from file |
| `kubectl apply -f ./folder/` | Apply all YAML files in a folder |
| `kubectl delete -f file.yaml` | Delete resource defined in file |
| `kubectl create -f file.yaml` | Create resource (fails if exists) |
| `kubectl diff -f file.yaml` | Show what would change before applying |
| `kubectl get deployment myapp -o yaml` | Export resource as YAML |
| `kubectl get deployment myapp -o json` | Export resource as JSON |

---

## Ingress

| Command | Description |
|---------|-------------|
| `kubectl get ingress` | List all ingress rules |
| `kubectl describe ingress <name>` | View ingress details |
| `kubectl apply -f ingress.yaml` | Create ingress from YAML |
| `kubectl delete ingress <name>` | Delete ingress |

---

## Volumes & Persistent Storage

| Command | Description |
|---------|-------------|
| `kubectl get pv` | List PersistentVolumes |
| `kubectl get pvc` | List PersistentVolumeClaims |
| `kubectl describe pvc <name>` | View PVC details |
| `kubectl delete pvc <name>` | Delete a PVC |

---

## Contexts & Config

| Command | Description |
|---------|-------------|
| `kubectl config view` | View kubeconfig |
| `kubectl config get-contexts` | List all contexts (clusters) |
| `kubectl config current-context` | Show active context |
| `kubectl config use-context <name>` | Switch to a different cluster |
| `kubectl config set-context --current --namespace=myns` | Change default namespace |

---

## Troubleshooting

| Command | Description |
|---------|-------------|
| `kubectl describe pod <pod>` | Most useful — shows events and errors |
| `kubectl logs <pod>` | View logs |
| `kubectl logs <pod> --previous` | Logs from a crashed/restarted pod |
| `kubectl get events` | View all cluster events |
| `kubectl get events --sort-by='.lastTimestamp'` | Events sorted by time |
| `kubectl exec -it <pod> -- bash` | Shell into pod to debug |
| `kubectl run debug --image=busybox -it --rm` | Spin up a temp debug pod |
| `kubectl top pods` | Check resource usage |
| `kubectl top nodes` | Check node resource usage |

---

## Common Flags

| Flag | Description |
|------|-------------|
| `-n myns` | Target a specific namespace |
| `-A` or `--all-namespaces` | All namespaces |
| `-o wide` | Extra output columns |
| `-o yaml` | Output as YAML |
| `-o json` | Output as JSON |
| `-f file.yaml` | Use a file |
| `--dry-run=client` | Simulate without applying |
| `-l app=myapp` | Filter by label |
| `--watch` or `-w` | Watch for live changes |
| `--force` | Force delete |

---

## Pod Lifecycle / Status

| Status | Meaning |
|--------|---------|
| `Pending` | Pod accepted but not yet scheduled to a node |
| `Running` | Pod is running on a node |
| `Succeeded` | All containers finished successfully |
| `Failed` | At least one container exited with error |
| `CrashLoopBackOff` | Container keeps crashing and restarting |
| `ImagePullBackOff` | Cannot pull the container image |
| `Terminating` | Pod is being deleted |
| `Unknown` | Node communication lost |

---

## Quick YAML Templates

### Pod
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mypod
  namespace: default
spec:
  containers:
  - name: mycontainer
    image: ubuntu
    command: ["bash"]
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: nginx:latest
        ports:
        - containerPort: 80
```

### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myconfig
data:
  APP_ENV: production
  APP_PORT: "8080"
```

### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: cGFzc3dvcmQxMjM=   # base64 encoded value
```
