# AI/ML CI/CD Cheatsheet

> Practical notes for building, testing, deploying, and monitoring ML/AI systems with CI/CD.

---

## 1. CI/CD for AI/ML

### Traditional Software CI/CD

```text
Code → Build → Test → Package → Deploy → Monitor
```

### ML CI/CD

```text
Code
  ↓
Data Validation
  ↓
Model Training / Validation
  ↓
Model Evaluation
  ↓
Model Registry
  ↓
Build Image
  ↓
Deploy
  ↓
Monitor
  ↓
Retrain
```

### Why ML CI/CD is different

ML systems have more than source code:

- Application code
- Training code
- Datasets
- Features
- Model artifacts
- Model configuration
- Dependencies
- Infrastructure
- Evaluation metrics

---

# 2. CI vs CD vs CT

| Term | Meaning | ML Example |
|---|---|---|
| CI | Continuous Integration | Test Python/model code on every PR |
| CD | Continuous Delivery/Deployment | Deploy validated model/API |
| CT | Continuous Training | Automatically retrain when data changes |

```text
CI = Code quality
CD = Deployment
CT = Model/data lifecycle
```

---

# 3. Typical MLOps Pipeline

```text
Git Push
   ↓
CI Pipeline
   ├── Lint
   ├── Unit Tests
   ├── Data Tests
   ├── Security Scan
   └── Build Docker Image
          ↓
     Training Pipeline
          ↓
     Evaluate Model
          ↓
   Register Model
          ↓
      Staging
          ↓
 Integration Tests
          ↓
     Production
          ↓
      Monitoring
          ↓
 Drift / Performance Alert
          ↓
       Retraining
```

---

# 4. Git Workflow

Recommended:

```text
main
 │
 ├── develop
 │
 ├── feature/model-training
 │
 ├── feature/rag-pipeline
 │
 └── bugfix/inference
```

Typical workflow:

```bash
git checkout -b feature/model
git add .
git commit -m "Add model training pipeline"
git push origin feature/model
```

Then:

```text
Pull Request
    ↓
CI Checks
    ↓
Code Review
    ↓
Merge
```

---

# 5. Project Structure

```text
ml-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── inference/
│   └── api/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── model/
│
├── configs/
│   └── config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
│
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 6. CI Pipeline

A good CI pipeline checks:

```text
Syntax
 ↓
Lint
 ↓
Unit Tests
 ↓
Data Validation
 ↓
Model Tests
 ↓
Security Scan
 ↓
Build
```

Example:

```yaml
name: ML CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint
        run: ruff check .

      - name: Test
        run: pytest

      - name: Build Docker
        run: docker build -t ml-api .
```

---

# 7. Testing in ML

## Unit Tests

Test individual functions.

```python
def test_preprocess():
    result = preprocess(data)
    assert result.shape[1] == 10
```

## Integration Tests

Test components together.

```text
API → Preprocessing → Model → Response
```

## Model Tests

Example:

```python
assert accuracy >= 0.85
assert precision >= 0.80
```

## Data Tests

Check:

- Missing values
- Schema
- Data types
- Range
- Duplicates
- Distribution
- Feature availability

Example:

```text
age ∈ [0, 120]
income >= 0
```

---

# 8. ML Quality Gates

Never automatically deploy every trained model.

Use gates:

```text
New Model
   ↓
Accuracy >= threshold?
   ↓ YES
Latency <= threshold?
   ↓ YES
Data validation passed?
   ↓ YES
Bias/fairness checks passed?
   ↓ YES
Deploy
```

Example:

```python
if accuracy >= 0.90 and latency_ms <= 200:
    deploy()
else:
    reject_model()
```

---

# 9. Reproducibility

A model should be reproducible.

Track:

```text
Code version
Dataset version
Model version
Dependencies
Hyperparameters
Random seed
Environment
Training timestamp
Evaluation metrics
```

Example:

```yaml
model:
  name: fraud-detector
  version: 1.3

training:
  dataset_version: v12
  random_seed: 42
  learning_rate: 0.001
  epochs: 20
```

---

# 10. Experiment Tracking

Track experiments using tools such as:

- MLflow
- Weights & Biases
- Neptune

Track:

```text
Parameters
Metrics
Artifacts
Models
Dataset versions
Git commit
```

Example:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 20)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_model(model, "model")
```

---

# 11. Model Registry

A model registry manages model versions.

```text
Model v1 → Staging
Model v2 → Staging
Model v2 → Production
Model v1 → Archived
```

Common stages:

```text
Development
    ↓
Staging
    ↓
Production
    ↓
Archived
```

Benefits:

- Version control
- Approval workflow
- Rollback
- Model lineage
- Deployment management

---

# 12. Docker for ML

Why Docker?

```text
Same environment
Same dependencies
Portable deployment
Isolation
Reproducibility
```

Example Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t ml-api:latest .
```

Run:

```bash
docker run -p 8000:8000 ml-api:latest
```

---

# 13. Docker Image Tagging

Avoid relying only on `latest`.

Bad:

```text
ml-api:latest
```

Better:

```text
ml-api:1.2.0
ml-api:abc123
ml-api:2026-09-14
```

Best practice:

```text
Image
 ↓
Git commit SHA
 ↓
Immutable deployment
```

---

# 14. CI/CD + Docker

```text
Git Push
   ↓
CI
   ↓
Tests
   ↓
Docker Build
   ↓
Security Scan
   ↓
Push to Registry
   ↓
Deploy
```

Container registries:

- AWS ECR
- Docker Hub
- GitHub Container Registry
- Google Artifact Registry
- Azure Container Registry

---

# 15. Model Artifact vs Docker Image

### Model Artifact

```text
model.pkl
model.joblib
model.pt
model.onnx
```

### Docker Image

Contains:

```text
Application
Dependencies
Runtime
Inference code
Configuration
```

Typical architecture:

```text
Docker Image
     +
Model Artifact
     ↓
Inference Service
```

---

# 16. Secrets Management

Never commit:

```text
API keys
AWS keys
Database passwords
OpenAI keys
Tokens
Private credentials
```

Bad:

```python
API_KEY = "sk-xxxxx"
```

Better:

```python
import os

API_KEY = os.getenv("API_KEY")
```

Use:

- GitHub Secrets
- AWS Secrets Manager
- AWS Systems Manager Parameter Store
- HashiCorp Vault
- Kubernetes Secrets

---

# 17. Environment Variables

Typical:

```bash
MODEL_NAME=fraud-detector
MODEL_VERSION=1.3
ENVIRONMENT=production
DATABASE_URL=...
API_KEY=...
```

Separate:

```text
.env.dev
.env.staging
.env.prod
```

Never commit real secrets.

---

# 18. Deployment Strategies

## Blue-Green

```text
Blue  → Current Production
Green → New Version

Test Green
   ↓
Switch Traffic
```

Rollback:

```text
Green → Blue
```

## Canary

Send small traffic first:

```text
v1 → 95%
v2 → 5%
```

If healthy:

```text
v1 → 50%
v2 → 50%
```

Then:

```text
v1 → 0%
v2 → 100%
```

## Rolling Deployment

Replace instances gradually.

```text
Old Old Old
   ↓
New Old Old
   ↓
New New Old
   ↓
New New New
```

---

# 19. Kubernetes

Core concepts:

```text
Pod
 ↓
Deployment
 ↓
Service
 ↓
Ingress
```

Example:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: ml-api

spec:
  replicas: 3

  selector:
    matchLabels:
      app: ml-api

  template:
    metadata:
      labels:
        app: ml-api

    spec:
      containers:
        - name: ml-api
          image: ml-api:1.2.0
          ports:
            - containerPort: 8000
```

Useful commands:

```bash
kubectl get pods
kubectl get deployments
kubectl get services
kubectl logs <pod>
kubectl describe pod <pod>
kubectl rollout status deployment/ml-api
kubectl rollout undo deployment/ml-api
```

---

# 20. Kubernetes for ML

Useful for:

- Scalable inference
- Multiple model replicas
- GPU workloads
- Rolling deployments
- Service discovery
- Autoscaling

Architecture:

```text
Load Balancer
      ↓
Kubernetes Service
      ↓
Inference Pods
 ┌────┼────┐
 ↓    ↓    ↓
Model Model Model
```

---

# 21. Cloud ML Deployment

## AWS Example

```text
GitHub
  ↓
GitHub Actions
  ↓
Docker
  ↓
ECR
  ↓
ECS / EKS / SageMaker
  ↓
Load Balancer
  ↓
Users
```

Useful AWS services:

| Need | AWS |
|---|---|
| Storage | S3 |
| Container Registry | ECR |
| Compute | EC2 |
| Containers | ECS |
| Kubernetes | EKS |
| ML platform | SageMaker |
| CI/CD | CodePipeline / CodeBuild |
| Secrets | Secrets Manager |
| Logs | CloudWatch |
| Monitoring | CloudWatch |
| Event scheduling | EventBridge |

---

# 22. FastAPI + ML Deployment

Typical architecture:

```text
Client
  ↓
FastAPI
  ↓
Preprocessing
  ↓
Model
  ↓
Prediction
```

Example:

```python
from fastapi import FastAPI

app = FastAPI()

model = load_model()

@app.post("/predict")
def predict(data: InputData):
    features = preprocess(data)
    prediction = model.predict(features)

    return {"prediction": prediction}
```

Health endpoint:

```python
@app.get("/health")
def health():
    return {"status": "healthy"}
```

---

# 23. AI/LLM CI/CD

For GenAI applications, test more than application code.

Pipeline:

```text
Code
 ↓
Unit Tests
 ↓
Prompt Tests
 ↓
RAG Evaluation
 ↓
Security Tests
 ↓
Docker Build
 ↓
Deploy
 ↓
LLM Monitoring
```

Test:

- Prompt templates
- Retrieval quality
- Context relevance
- Groundedness
- Hallucination rate
- Token usage
- Latency
- Cost
- Safety
- Tool/function calling

---

# 24. RAG CI/CD

Typical:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector DB
   ↓
Retriever
   ↓
LLM
```

CI tests:

```text
Chunking tests
Embedding dimension tests
Retriever tests
Prompt tests
Citation/grounding tests
LLM response tests
```

Evaluation metrics:

```text
Context Precision
Context Recall
Faithfulness
Answer Relevance
Retrieval Recall
Latency
Cost
```

---

# 25. LLM Evaluation

Don't test only:

```text
HTTP 200
```

Also evaluate:

```text
Question
   ↓
Retrieved Context
   ↓
Generated Answer
   ↓
Expected / Reference
```

Example threshold:

```yaml
evaluation:
  faithfulness: 0.90
  answer_relevance: 0.85
  retrieval_recall: 0.90
```

If thresholds fail:

```text
CI/CD Pipeline → FAIL
```

---

# 26. Prompt Versioning

Treat prompts like code.

Bad:

```text
Prompt stored manually in production
```

Better:

```text
prompts/
├── system_v1.txt
├── system_v2.txt
└── rag_v3.txt
```

Track:

```text
Prompt version
Model version
Temperature
Context
Evaluation score
```

---

# 27. AI Agent CI/CD

Agent pipeline:

```text
Code
 ↓
Unit Tests
 ↓
Tool Tests
 ↓
Prompt Tests
 ↓
Agent Evaluation
 ↓
Security Tests
 ↓
Docker
 ↓
Staging
 ↓
Production
```

Test:

- Tool selection
- Tool arguments
- Agent loops
- Maximum iterations
- Failure handling
- Prompt injection
- Unauthorized tool access
- Output schema

---

# 28. Data Validation

Example checks:

```python
assert df.shape[0] > 1000
assert df["age"].notnull().all()
assert df["age"].between(0, 120).all()
```

Better tools:

- Great Expectations
- Pandera
- Evidently

Data validation pipeline:

```text
Raw Data
   ↓
Schema Check
   ↓
Missing Value Check
   ↓
Distribution Check
   ↓
Quality Gate
   ↓
Training
```

---

# 29. Data Drift

Data distribution changes over time.

```text
Training Data
      ↓
Production Data
      ↓
Distribution changes
      ↓
Data Drift
```

Example:

```text
Training age:
mean = 35

Production age:
mean = 52
```

Possible tools:

- Evidently
- WhyLabs
- Arize
- Cloud monitoring tools

---

# 30. Model Drift

Model performance decreases because real-world behavior changes.

```text
Production
   ↓
Predictions
   ↓
Actual outcomes
   ↓
Performance drops
   ↓
Model Drift
```

Monitor:

```text
Accuracy
Precision
Recall
F1
AUC
MAE
RMSE
```

depending on the problem.

---

# 31. Monitoring

Monitor three layers.

### Infrastructure

```text
CPU
Memory
GPU
Disk
Network
```

### Application

```text
Request rate
Error rate
Latency
Throughput
```

### ML

```text
Prediction distribution
Data drift
Model performance
Feature drift
```

### LLM

```text
Token usage
Cost
Latency
Hallucination
Faithfulness
Safety
```

---

# 32. Observability

Use:

```text
Logs
Metrics
Traces
```

Known as:

```text
Three Pillars of Observability
```

Example:

```text
Request
 ↓
API
 ↓
Retriever
 ↓
LLM
 ↓
Response
```

Tracing helps identify where latency/errors occur.

---

# 33. Rollback

If production model is bad:

```text
v2 Production
     ↓
Issue detected
     ↓
Rollback
     ↓
v1 Production
```

Kubernetes:

```bash
kubectl rollout undo deployment/ml-api
```

Model registry:

```text
Production → Model v1
```

---

# 34. Retraining Pipeline

Trigger retraining when:

```text
New data available
OR
Data drift detected
OR
Model performance drops
OR
Scheduled interval reached
```

Pipeline:

```text
Trigger
  ↓
Collect Data
  ↓
Validate Data
  ↓
Train
  ↓
Evaluate
  ↓
Compare with Production
  ↓
Register
  ↓
Deploy if better
```

---

# 35. Champion vs Challenger

```text
Production Model = Champion
New Candidate    = Challenger
```

Compare:

```text
Champion:
F1 = 0.89

Challenger:
F1 = 0.92
```

If challenger satisfies all gates:

```text
Challenger → Champion
```

---

# 36. CI/CD Security

Security checks:

```text
SAST
DAST
Dependency Scan
Container Scan
Secret Scan
IaC Scan
```

Examples:

- Bandit
- Ruff
- Semgrep
- Trivy
- Gitleaks
- Dependabot

Pipeline:

```text
Code
 ↓
Secret Scan
 ↓
Dependency Scan
 ↓
SAST
 ↓
Docker Scan
 ↓
Deploy
```

---

# 37. Infrastructure as Code

Instead of manually creating infrastructure:

```text
Terraform
CloudFormation
Pulumi
```

Example:

```text
Terraform
   ↓
VPC
ECR
ECS/EKS
IAM
Load Balancer
S3
```

Benefits:

- Reproducibility
- Version control
- Automation
- Easy rollback
- Environment consistency

---

# 38. GitHub Actions Secrets

Example:

```yaml
env:
  AWS_REGION: ${{ secrets.AWS_REGION }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
```

Never:

```yaml
AWS_ACCESS_KEY_ID: "AKIA..."
```

Prefer short-lived credentials / OIDC where supported.

---

# 39. Environments

Maintain:

```text
Development
     ↓
Staging
     ↓
Production
```

### Development

Fast experimentation.

### Staging

Production-like testing.

### Production

Real users/traffic.

---

# 40. Approval Gates

For sensitive ML deployments:

```text
CI
 ↓
Training
 ↓
Evaluation
 ↓
Staging
 ↓
Manual Approval
 ↓
Production
```

For mature systems:

```text
Automated quality gates
        ↓
Automatic deployment
```

---

# 41. GitHub Actions Full Concept

```yaml
name: ML Pipeline

on:
  push:
    branches: [main]

jobs:

  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest

  build:
    needs: test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - run: docker build -t ml-api:${{ github.sha }} .

  deploy:
    needs: build
    runs-on: ubuntu-latest

    steps:
      - run: echo "Deploy model"
```

Core concept:

```text
needs:
  ↓
Job dependency
```

---

# 42. Jenkins

Common in enterprise environments.

Pipeline:

```text
Git
 ↓
Jenkins
 ↓
Build
 ↓
Test
 ↓
Docker
 ↓
Deploy
```

Typical Jenkinsfile:

```groovy
pipeline {
    stages {
        stage('Test') {
            steps {
                sh 'pytest'
            }
        }

        stage('Build') {
            steps {
                sh 'docker build -t ml-api .'
            }
        }

        stage('Deploy') {
            steps {
                sh './deploy.sh'
            }
        }
    }
}
```

---

# 43. CI/CD Tools Cheat Sheet

| Category | Tools |
|---|---|
| Git | Git, GitHub, GitLab |
| CI/CD | GitHub Actions, GitLab CI, Jenkins |
| Containers | Docker |
| Orchestration | Kubernetes |
| Registry | ECR, GHCR, Docker Hub |
| ML Tracking | MLflow, W&B |
| Data Validation | Great Expectations, Pandera |
| Monitoring | Prometheus, Grafana |
| ML Monitoring | Evidently, Arize |
| Cloud | AWS, GCP, Azure |
| IaC | Terraform |
| Security | Trivy, Gitleaks, Bandit |
| API | FastAPI |
| Experimentation | MLflow, W&B |

---

# 44. Production ML Architecture

```text
                 ┌───────────────┐
                 │    GitHub     │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │ CI/CD Pipeline│
                 └───────┬───────┘
                         ↓
              ┌─────────────────────┐
              │ Test / Build / Scan │
              └──────────┬──────────┘
                         ↓
                  ┌─────────────┐
                  │ Model Train │
                  └──────┬──────┘
                         ↓
                  ┌─────────────┐
                  │   Evaluate  │
                  └──────┬──────┘
                         ↓
                  ┌─────────────┐
                  │Model Registry│
                  └──────┬──────┘
                         ↓
                    ┌─────────┐
                    │ Staging │
                    └────┬────┘
                         ↓
                    ┌─────────┐
                    │Production│
                    └────┬────┘
                         ↓
                  ┌──────────────┐
                  │  Monitoring  │
                  └──────┬───────┘
                         ↓
                    Drift / Alert
                         ↓
                     Retraining
```

---

# 45. Golden CI/CD Checklist

Before production:

```text
[ ] Git version control
[ ] Code linting
[ ] Unit tests
[ ] Integration tests
[ ] Data validation
[ ] Model evaluation
[ ] Quality gates
[ ] Experiment tracking
[ ] Model registry
[ ] Docker image
[ ] Image security scan
[ ] Secrets management
[ ] Infrastructure as Code
[ ] Staging environment
[ ] Deployment strategy
[ ] Health checks
[ ] Logging
[ ] Metrics
[ ] Model monitoring
[ ] Data drift monitoring
[ ] Rollback strategy
[ ] Retraining strategy
```

---

# 46. Interview One-Liners

### What is CI/CD in ML?

> CI/CD for ML automates code testing, model validation, packaging, deployment, monitoring, and often retraining while accounting for data and model artifacts.

### CI vs CD?

> CI validates changes continuously; CD automates delivering validated changes to environments.

### What is Continuous Training?

> Continuous Training automatically retrains models when new data, drift, schedules, or performance triggers require it.

### Why is ML CI/CD harder?

> ML systems depend on code, data, features, models, experiments, and infrastructure, so reproducibility and validation involve more than source code.

### Why Docker?

> Docker packages the application and dependencies into a reproducible runtime environment.

### Why model registry?

> A model registry versions, tracks, approves, and manages models across development, staging, and production.

### What is model drift?

> Model drift occurs when model performance degrades as the relationship between inputs and outcomes changes.

### What is data drift?

> Data drift occurs when the statistical distribution of production inputs changes compared with training data.

### What is canary deployment?

> Canary deployment releases a new version to a small percentage of traffic before gradually increasing traffic.

### What is rollback?

> Rollback restores a previously stable model or application version when the new deployment fails quality or production checks.

---

# 47. ML CI/CD vs MLOps

```text
CI/CD
  └── Automation of build/test/deploy

MLOps
  ├── CI/CD
  ├── Data Management
  ├── Experiment Tracking
  ├── Model Registry
  ├── Model Deployment
  ├── Monitoring
  ├── Drift Detection
  └── Continuous Training
```

Think:

```text
MLOps = DevOps + ML lifecycle
```

---

# 48. GenAI/MLOps Stack to Remember

For an AI/ML Engineer, a practical stack is:

```text
Python
  ↓
Git/GitHub
  ↓
Pytest + Ruff
  ↓
Docker
  ↓
GitHub Actions / Jenkins
  ↓
MLflow
  ↓
FastAPI
  ↓
AWS
  ↓
ECR
  ↓
ECS/EKS/SageMaker
  ↓
Prometheus/Grafana
  ↓
Evidently
```

For GenAI:

```text
Python
 ↓
FastAPI
 ↓
LangChain / LlamaIndex
 ↓
Vector DB
 ↓
LLM
 ↓
Docker
 ↓
CI/CD
 ↓
Cloud
 ↓
LLM Evaluation + Observability
```

---

# 49. Final Mental Model

Remember this:

```text
                 CODE
                   ↓
                  CI
                   ↓
       ┌───────────┴───────────┐
       ↓                       ↓
   TEST CODE               TEST DATA
       ↓                       ↓
       └───────────┬───────────┘
                   ↓
             TRAIN / BUILD
                   ↓
                EVALUATE
                   ↓
             QUALITY GATE
                   ↓
             MODEL REGISTRY
                   ↓
                STAGING
                   ↓
              PRODUCTION
                   ↓
              MONITORING
                   ↓
            DRIFT / FAILURE
                   ↓
              RETRAIN / ROLLBACK
                   ↓
               PRODUCTION
```

## The 10 things to remember

```text
1. CI  → Test code/data/model
2. CD  → Deploy validated artifacts
3. CT  → Retrain automatically
4. Docker → Reproducible runtime
5. MLflow → Track experiments/models
6. Registry → Version model lifecycle
7. Quality gates → Don't deploy bad models
8. Monitoring → Observe production
9. Drift → Detect changing data/behavior
10. Rollback → Recover quickly
```
