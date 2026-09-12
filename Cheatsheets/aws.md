# AWS for AI/ML — In-Depth Cheat Sheet

A practical **AWS AI/ML cheat sheet** covering the services, architecture, CLI commands, deployment patterns, MLOps, GenAI, RAG, and interview concepts you’re most likely to need as an **AI/ML Engineer**.

---

# 1. AWS AI/ML Big Picture

```text
                         AWS
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
     Storage          Compute           AI/ML
        │                 │                 │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────────┐
   S3       EFS      EC2      ECS     SageMaker    Bedrock
   │          │       │        │          │            │
   │          │       │        │          │            │
 datasets   models   training  APIs    ML lifecycle   GenAI
```

### Services to know

| Area           | AWS Service         | AI/ML Usage                     |
| -------------- | ------------------- | ------------------------------- |
| Object Storage | **S3**              | Datasets, models, artifacts     |
| Compute        | **EC2**             | GPU/CPU training & inference    |
| ML Platform    | **SageMaker**       | Training, deployment, pipelines |
| GenAI          | **Bedrock**         | Foundation models, RAG, agents  |
| Containers     | **ECR**             | Store Docker images             |
| Containers     | **ECS**             | Deploy containers               |
| Kubernetes     | **EKS**             | Deploy ML systems               |
| Serverless     | **Lambda**          | Lightweight inference/API logic |
| API            | **API Gateway**     | Expose ML APIs                  |
| Database       | **RDS**             | Relational metadata             |
| NoSQL          | **DynamoDB**        | Fast application metadata       |
| Vector Search  | **OpenSearch**      | Vector DB / RAG                 |
| Monitoring     | **CloudWatch**      | Logs, metrics, alarms           |
| IAM            | **IAM**             | Authentication/authorization    |
| Networking     | **VPC**             | Private ML infrastructure       |
| Secrets        | **Secrets Manager** | API keys/secrets                |
| Registry       | **ECR**             | Docker images                   |
| Workflow       | **Step Functions**  | ML workflows                    |
| Queue          | **SQS**             | Async inference jobs            |
| Events         | **EventBridge**     | Event-driven pipelines          |
| IaC            | **CloudFormation**  | Infrastructure as code          |
| IaC            | **CDK**             | Infrastructure using code       |

---

# 2. S3 — Most Important AWS Service for ML

Think:

> **S3 = ML data lake + model/artifact storage**

Typical structure:

```text
s3://my-ml-bucket/
│
├── raw/
│   ├── images/
│   ├── pdfs/
│   └── csv/
│
├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── features/
│
├── models/
│   ├── model.pkl
│   └── model.tar.gz
│
├── embeddings/
│
└── experiments/
```

### CLI

```bash
aws s3 ls
```

```bash
aws s3 ls s3://my-bucket/
```

Upload:

```bash
aws s3 cp model.pkl s3://my-bucket/models/
```

Download:

```bash
aws s3 cp s3://my-bucket/models/model.pkl .
```

Sync:

```bash
aws s3 sync ./data s3://my-bucket/data/
```

Delete:

```bash
aws s3 rm s3://my-bucket/file.csv
```

Recursive:

```bash
aws s3 rm s3://my-bucket/data/ --recursive
```

### Python / boto3

```python
import boto3

s3 = boto3.client("s3")

s3.upload_file(
    "model.pkl",
    "my-bucket",
    "models/model.pkl"
)
```

Download:

```python
s3.download_file(
    "my-bucket",
    "models/model.pkl",
    "model.pkl"
)
```

Read:

```python
obj = s3.get_object(
    Bucket="my-bucket",
    Key="data/train.csv"
)

data = obj["Body"].read()
```

---

# 3. IAM

IAM controls:

```text
WHO → CAN DO WHAT → ON WHICH RESOURCE
```

Example:

```text
SageMaker Role
      │
      ├── Read S3
      ├── Write S3
      ├── Write CloudWatch
      └── Access ECR
```

### Key concepts

* User
* Group
* Role
* Policy
* Permission
* Resource
* Trust policy
* Access key

### Example policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::my-bucket/*"
    }
  ]
}
```

### Best practice

```text
❌ Hardcode AWS keys
❌ Store keys in GitHub

✅ IAM Role
✅ Environment variables for local development
✅ AWS Secrets Manager for secrets
```

---

# 4. EC2

**EC2 = virtual machine in AWS**

For AI/ML:

```text
CPU training
GPU training
GPU inference
FastAPI
Flask
Jupyter
Docker
Ollama
Custom ML servers
```

Typical architecture:

```text
User
 │
 ▼
EC2
 │
 ├── Docker
 │    └── FastAPI
 │          └── ML Model
 │
 └── S3
      └── model
```

### SSH

```bash
ssh -i key.pem ubuntu@<EC2-IP>
```

### GPU check

```bash
nvidia-smi
```

### Docker

```bash
docker build -t ml-api .
```

```bash
docker run -p 8000:8000 ml-api
```

### Useful EC2 concepts

| Concept        | Meaning                   |
| -------------- | ------------------------- |
| AMI            | Machine image             |
| Instance type  | CPU/GPU/RAM configuration |
| EBS            | Persistent block storage  |
| Security Group | Virtual firewall          |
| Key Pair       | SSH authentication        |
| Elastic IP     | Static public IP          |
| VPC            | Network                   |

---

# 5. EC2 Instance Types for ML

General idea:

```text
CPU
 ↓
General purpose
 ↓
Compute optimized
 ↓
GPU
 ↓
High-end GPU
```

Examples:

```text
t3 / t4g       → lightweight services
c7i            → CPU compute
g5             → NVIDIA GPU
g6             → newer GPU workloads
p-series       → heavy deep learning
```

For ML interviews, understand:

```text
CPU → traditional ML / APIs
GPU → deep learning / LLM inference
High-memory → large models / datasets
```

---

# 6. SageMaker

## What is SageMaker?

Managed platform for:

```text
Data
 ↓
Processing
 ↓
Training
 ↓
Evaluation
 ↓
Model Registry
 ↓
Deployment
 ↓
Monitoring
```

Think:

> **SageMaker = AWS's end-to-end ML platform**

---

# 7. SageMaker Architecture

```text
             S3
              │
              ▼
       Data Processing
              │
              ▼
          Training
              │
              ▼
       Model Artifact
              │
              ▼
      Model Registry
              │
              ▼
        Endpoint
              │
              ▼
        Predictions
              │
              ▼
        Monitoring
```

---

# 8. SageMaker Training

Training job:

```text
S3 Dataset
     │
     ▼
SageMaker Training Job
     │
     ├── Docker container
     ├── Training code
     ├── GPU/CPU
     └── Hyperparameters
     │
     ▼
model.tar.gz
     │
     ▼
S3
```

Example:

```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri="...",
    role=role,
    instance_type="ml.g5.xlarge",
    instance_count=1,
    output_path="s3://bucket/models/"
)

estimator.fit({
    "train": "s3://bucket/train/"
})
```

---

# 9. SageMaker Training Script

Typical:

```python
def train():

    # Load data
    X_train, y_train = load_data()

    # Train
    model = train_model(X_train, y_train)

    # Save
    save_model(model)
```

SageMaker expects artifacts to eventually be uploaded to S3.

Typical:

```text
/opt/ml/
│
├── input/
│   └── data/
│
├── model/
│
├── output/
│
└── code/
```

---

# 10. SageMaker Endpoint

Deployment:

```text
Model
 │
 ▼
SageMaker Endpoint
 │
 ▼
HTTPS API
 │
 ▼
Application
```

Python:

```python
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
```

Prediction:

```python
predictor.predict(data)
```

---

# 11. Real-Time vs Batch Inference

### Real-time

Use when:

```text
Request
  ↓
Model
  ↓
Response immediately
```

Examples:

* Fraud detection
* Recommendation
* Image classification
* Chatbot API

### Batch

```text
100,000 records
       ↓
Batch Job
       ↓
Predictions
       ↓
S3
```

Use for:

* Large datasets
* Daily predictions
* Offline scoring

### Asynchronous

```text
Client
 ↓
Request
 ↓
Queue
 ↓
Model
 ↓
Result
```

Useful for:

* Large files
* Long inference
* Image/video processing

---

# 12. SageMaker Model Registry

Purpose:

```text
Model versions
     ↓
Validation
     ↓
Approval
     ↓
Deployment
```

Example:

```text
model-v1 → accuracy 89%
model-v2 → accuracy 92%
model-v3 → accuracy 94% ← production
```

Useful for:

* Versioning
* Approval workflows
* CI/CD
* Governance

---

# 13. SageMaker Pipelines

ML pipeline:

```text
        Data
          ↓
      Processing
          ↓
      Validation
          ↓
       Training
          ↓
      Evaluation
          ↓
   Condition Check
       ↙      ↘
    Reject    Register
                ↓
             Deploy
```

Typical pipeline components:

```text
ProcessingStep
TrainingStep
EvaluationStep
ConditionStep
RegisterModel
Deploy
```

---

# 14. SageMaker Hyperparameter Tuning

Instead of:

```python
learning_rate = 0.001
```

Try:

```text
0.0001
0.0005
0.001
0.005
0.01
```

AWS can search automatically.

Typical parameters:

```text
learning_rate
batch_size
epochs
max_depth
num_layers
dropout
```

Goal:

```text
maximize accuracy
minimize loss
maximize F1
```

---

# 15. SageMaker Feature Store

Used for:

```text
Feature engineering
        ↓
Feature storage
        ↓
Training
        ↓
Inference
```

Example:

```text
customer_age
customer_income
transaction_count
avg_transaction
```

Why?

Avoid inconsistent features between:

```text
Training
    vs
Production inference
```

---

# 16. SageMaker Model Monitor

Monitor deployed models for:

### Data drift

```text
Training distribution
        ≠
Production distribution
```

Example:

```text
Training age:
20–60

Production:
60–90
```

### Model quality

Monitor:

```text
Accuracy
Precision
Recall
F1
RMSE
MAE
```

### Bias

Monitor changes in model behavior across relevant groups.

---

# 17. AWS Bedrock

For GenAI, know **Bedrock** extremely well.

Think:

> **Bedrock = managed access to foundation models + GenAI building blocks**

Typical:

```text
Application
     │
     ▼
Amazon Bedrock
     │
 ┌───┼───────────┐
 │   │           │
LLM  Embedding  Guardrails
 │
 ▼
Response
```

---

# 18. Bedrock Use Cases

```text
Chatbots
RAG
Summarization
Question answering
Agents
Content generation
Classification
Extraction
```

---

# 19. Bedrock Foundation Models

Bedrock provides access to models from multiple providers.

Conceptually:

```text
Amazon models
Anthropic models
Meta models
Mistral models
Cohere models
Google models
etc.
```

Model availability changes over time and by AWS Region.

---

# 20. Bedrock vs SageMaker

Very important interview question.

| Bedrock           | SageMaker                   |
| ----------------- | --------------------------- |
| GenAI focused     | Full ML platform            |
| Foundation models | Train/deploy ML models      |
| API-based         | More control                |
| RAG               | Custom training             |
| Agents            | ML pipelines                |
| Prompting         | Fine-tuning/training        |
| Easier            | More infrastructure control |

Simple rule:

```text
Need foundation model API?
        ↓
      Bedrock

Need custom ML lifecycle?
        ↓
    SageMaker
```

---

# 21. Bedrock RAG

Classic architecture:

```text
                 Documents
                     │
                     ▼
                Chunking
                     │
                     ▼
                Embeddings
                     │
                     ▼
              Vector Database
                     │
User Question ───────┤
                     ▼
                Retrieval
                     │
                     ▼
              Context + Query
                     │
                     ▼
               Foundation LLM
                     │
                     ▼
                   Answer
```

AWS components could include:

```text
S3
 ↓
Knowledge Bases for Amazon Bedrock
 ↓
Vector store
 ↓
Bedrock model
```

---

# 22. Embeddings

Text:

```text
"What is AWS?"
```

becomes:

```text
[0.12, -0.32, 0.78, ...]
```

Similar text:

```text
"What is Amazon Web Services?"
```

should have similar vectors.

Used for:

```text
Semantic search
RAG
Recommendation
Document retrieval
Clustering
```

---

# 23. Vector Database Options on AWS

Know these:

```text
Amazon OpenSearch Service
Amazon Aurora PostgreSQL + pgvector
Amazon RDS for PostgreSQL + pgvector
Amazon Neptune Analytics / graph-related workloads
Third-party vector databases
```

For interviews:

> OpenSearch is a common AWS-native choice for vector search.

---

# 24. OpenSearch RAG

```text
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Embedding Model
 ↓
OpenSearch
 ↓
Vector similarity
 ↓
Top-K documents
 ↓
LLM
 ↓
Answer
```

Similarity:

```text
cosine similarity
```

Formula:

```text
cos(A,B) = A·B / (||A|| ||B||)
```

---

# 25. Bedrock Agents

Agents allow an LLM to:

```text
Understand request
       ↓
Plan
       ↓
Choose tool
       ↓
Execute action
       ↓
Observe result
       ↓
Continue
       ↓
Final answer
```

Example:

```text
User:
"Check my order status"

Agent
 ↓
Order API
 ↓
Database
 ↓
Return result
```

Key concept:

> **LLM + tools + orchestration + knowledge**

---

# 26. Lambda

Serverless compute.

```text
Event
 ↓
Lambda
 ↓
Execute code
 ↓
Response
```

Good for:

* preprocessing
* lightweight inference
* triggering pipelines
* API logic
* S3 events
* event processing

Not ideal for:

```text
Huge models
Long GPU inference
Long-running training
```

---

# 27. Lambda + ML API

```text
User
 ↓
API Gateway
 ↓
Lambda
 ↓
SageMaker Endpoint
 ↓
Prediction
```

This is a very common architecture.

---

# 28. API Gateway

Expose APIs:

```text
POST /predict
GET /health
POST /chat
```

Architecture:

```text
Client
 ↓
API Gateway
 ↓
Lambda / ECS / SageMaker
 ↓
Response
```

---

# 29. ECS

**Elastic Container Service**

Used to run Docker containers.

```text
Dockerfile
    ↓
Docker Image
    ↓
ECR
    ↓
ECS
    ↓
Running Container
```

For an AI API:

```text
FastAPI
 ↓
Docker
 ↓
ECR
 ↓
ECS
 ↓
Load Balancer
 ↓
Users
```

---

# 30. ECR

**Elastic Container Registry**

Think:

> GitHub for Docker images — but AWS-native.

```bash
docker build -t my-model .
```

Authenticate:

```bash
aws ecr get-login-password --region <region> \
| docker login --username AWS --password-stdin <registry>
```

Tag:

```bash
docker tag my-model:latest <registry>/my-model:latest
```

Push:

```bash
docker push <registry>/my-model:latest
```

---

# 31. ECS vs EKS

| ECS                                | EKS                  |
| ---------------------------------- | -------------------- |
| AWS-native container orchestration | Kubernetes           |
| Easier                             | More complex         |
| AWS integration                    | Kubernetes ecosystem |
| Less operational overhead          | More flexibility     |

Rule:

```text
Want simpler AWS containers → ECS

Need Kubernetes → EKS
```

---

# 32. CloudWatch

Monitoring service.

Monitor:

```text
CPU
Memory
GPU
Latency
Errors
Requests
Logs
```

Typical ML API:

```text
FastAPI
   ↓
CloudWatch Logs

SageMaker
   ↓
CloudWatch Metrics
```

Useful CLI:

```bash
aws logs describe-log-groups
```

---

# 33. CloudWatch Alarm

Example:

```text
CPU > 80%
     ↓
CloudWatch Alarm
     ↓
Notification / Auto Scaling
```

ML example:

```text
Endpoint latency > threshold
       ↓
Alarm
       ↓
Investigate / scale
```

---

# 34. VPC

Virtual Private Cloud.

Think:

> Your private AWS network.

```text
VPC
│
├── Public Subnet
│
└── Private Subnet
      │
      ├── EC2
      ├── SageMaker
      └── Database
```

Important:

```text
Subnet
Route Table
Internet Gateway
NAT Gateway
Security Group
Network ACL
```

---

# 35. Security Groups

Virtual firewall.

Example:

```text
Inbound
22   SSH
80   HTTP
443  HTTPS
8000 API
```

Best practice:

```text
❌ 0.0.0.0/0 for everything

✅ Restrict source IP/security group
```

---

# 36. Secrets Manager

Never:

```python
API_KEY = "sk-xxxxxxxx"
```

Instead:

```text
Application
    ↓
Secrets Manager
    ↓
API Key
```

Python:

```python
import boto3

client = boto3.client("secretsmanager")

response = client.get_secret_value(
    SecretId="my-api-key"
)

secret = response["SecretString"]
```

---

# 37. Parameter Store

AWS Systems Manager Parameter Store.

Useful for configuration:

```text
MODEL_NAME
ENVIRONMENT
DATABASE_URL
API_ENDPOINT
```

Difference:

```text
Secrets Manager
→ secrets

Parameter Store
→ configuration + parameters
```

---

# 38. SQS

Simple Queue Service.

Useful for asynchronous ML.

```text
User
 ↓
API
 ↓
SQS
 ↓
Worker
 ↓
Model
 ↓
S3
```

Example:

```text
10,000 images
     ↓
Queue
     ↓
Workers
     ↓
Image classifier
```

---

# 39. Step Functions

Workflow orchestration.

Example:

```text
Start
 ↓
Load Data
 ↓
Validate
 ↓
Train
 ↓
Evaluate
 ↓
Accuracy > 90%?
 ↙          ↘
No          Yes
 ↓           ↓
Stop      Deploy
```

Useful for:

* ML workflows
* ETL
* inference workflows
* multi-step AI applications

---

# 40. EventBridge

Event-driven architecture.

Example:

```text
S3 upload
   ↓
EventBridge
   ↓
Lambda
   ↓
Start ML pipeline
```

Another:

```text
Model approved
     ↓
EventBridge
     ↓
Deployment
```

---

# 41. AWS Glue

Serverless data integration / ETL.

```text
Raw Data
   ↓
Glue
   ↓
Transform
   ↓
S3 / Data Warehouse
```

Useful for:

```text
ETL
Data catalog
Schema discovery
Data preparation
```

---

# 42. Athena

Query data directly in S3 using SQL.

```sql
SELECT *
FROM my_dataset
WHERE age > 30;
```

Architecture:

```text
S3
 ↓
Athena
 ↓
SQL
 ↓
Results
```

Very useful for ML data exploration.

---

# 43. Redshift

Data warehouse.

Use when you need:

```text
Large-scale analytics
BI
SQL analytics
Structured data warehouse
```

Simplified:

```text
S3 → Data Lake

Redshift → Data Warehouse
```

---

# 44. AWS ML Data Architecture

A realistic architecture:

```text
                    ┌─────────────┐
                    │ Applications│
                    └──────┬──────┘
                           │
                           ▼
                         S3
                           │
                    ┌──────┴──────┐
                    │             │
                  Glue          Athena
                    │
                    ▼
              Processed Data
                    │
                    ▼
                SageMaker
                    │
             ┌──────┴──────┐
             │             │
          Training       Evaluation
             │
             ▼
          Model
             │
             ▼
       Model Registry
             │
             ▼
        Endpoint
             │
             ▼
         Application
```

---

# 45. Complete ML Deployment Architecture

```text
Developer
   │
   ▼
GitHub
   │
   ▼
CI/CD
   │
   ▼
Docker
   │
   ▼
ECR
   │
   ▼
SageMaker / ECS
   │
   ▼
Model API
   │
   ▼
API Gateway / ALB
   │
   ▼
Users
```

Monitoring:

```text
                    ┌─────────────┐
                    │ CloudWatch  │
                    └──────▲──────┘
                           │
Users → API → Model → Logs/Metrics
```

---

# 46. Complete GenAI/RAG Architecture

For an AI/ML engineer, remember this architecture:

```text
                    DOCUMENT INGESTION

PDF / DOC / TXT
       │
       ▼
      S3
       │
       ▼
Text Extraction
       │
       ▼
   Chunking
       │
       ▼
  Embeddings
       │
       ▼
Vector Database
       │
       │
       │
       ▼
    RETRIEVAL
       ▲
       │
User Query
       │
       ▼
Query Embedding
       │
       ▼
Similarity Search
       │
       ▼
Top-K Chunks
       │
       ▼
   Prompt
       │
       ▼
Bedrock LLM
       │
       ▼
   Answer
```

---

# 47. AI Agent Architecture on AWS

```text
                 User
                   │
                   ▼
              API Gateway
                   │
                   ▼
             Agent / Lambda
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
        RAG       API      DB
          │        │        │
          └────────┼────────┘
                   ▼
                LLM
                   │
                   ▼
                Answer
```

Potential services:

```text
Bedrock
Lambda
API Gateway
S3
OpenSearch
DynamoDB
Step Functions
CloudWatch
IAM
```

---

# 48. AWS + FastAPI + ML

A practical deployment:

```text
FastAPI
   │
Docker
   │
ECR
   │
ECS / EC2
   │
Load Balancer
   │
API Gateway
   │
Frontend
```

FastAPI:

```python
@app.post("/predict")
def predict(request):

    result = model.predict(
        request.data
    )

    return {
        "prediction": result
    }
```

---

# 49. AWS + Docker + ML

```text
model.pkl
requirements.txt
app.py
Dockerfile
```

Dockerfile:

```dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app",
     "--host", "0.0.0.0",
     "--port", "8000"]
```

Build:

```bash
docker build -t ml-api .
```

Run:

```bash
docker run -p 8000:8000 ml-api
```

---

# 50. CI/CD for ML

Typical:

```text
Git Push
   ↓
GitHub Actions
   ↓
Tests
   ↓
Build Docker
   ↓
Push ECR
   ↓
Deploy
   ↓
Smoke Test
   ↓
Production
```

ML extension:

```text
Git Push
   ↓
Test
   ↓
Data validation
   ↓
Train
   ↓
Evaluate
   ↓
Register model
   ↓
Approval
   ↓
Deploy
```

---

# 51. MLOps Lifecycle

Memorize:

```text
DATA
 ↓
VALIDATE
 ↓
PREPROCESS
 ↓
TRAIN
 ↓
EVALUATE
 ↓
REGISTER
 ↓
DEPLOY
 ↓
MONITOR
 ↓
RETRAIN
 ↺
```

AWS mapping:

| Lifecycle  | AWS                                |
| ---------- | ---------------------------------- |
| Data       | S3                                 |
| ETL        | Glue                               |
| Query      | Athena                             |
| Training   | SageMaker                          |
| Registry   | SageMaker Model Registry           |
| Deployment | SageMaker/ECS/EKS                  |
| Monitoring | CloudWatch/SageMaker               |
| Workflow   | Step Functions/SageMaker Pipelines |
| Secrets    | Secrets Manager                    |
| Containers | ECR                                |

---

# 52. ML Monitoring

Monitor two things separately.

### Infrastructure

```text
CPU
RAM
GPU
Disk
Network
Latency
Throughput
```

### ML

```text
Data drift
Prediction drift
Feature drift
Model accuracy
Precision
Recall
F1
RMSE
```

---

# 53. Data Drift vs Concept Drift

### Data drift

Input changes:

```text
P(X)
```

Example:

```text
Training users:
age 20–40

Production:
age 50–70
```

### Concept drift

Relationship changes:

```text
P(Y|X)
```

Example:

```text
Customer behavior changes
```

Remember:

```text
Data drift → input distribution changes

Concept drift → relationship between input and target changes
```

---

# 54. Model Deployment Strategies

### Blue-Green

```text
Blue → Current production
Green → New model

Test Green
     ↓
Switch traffic
```

### Canary

```text
100% traffic
     ↓
95% old
5% new
     ↓
Monitor
     ↓
20% new
     ↓
50%
     ↓
100%
```

### Shadow

```text
User
 ├── Old model → actual response
 └── New model → test only
```

---

# 55. Cost Optimization

Important in real AWS work.

### S3

Use:

```text
Lifecycle policies
Compression
Appropriate storage class
```

### EC2/SageMaker

Avoid:

```text
GPU running 24/7 unnecessarily
```

Use:

```text
Spot Instances
Auto Scaling
Scheduled shutdown
Batch inference
Right-sized instances
```

### Architecture

```text
Real-time → endpoint

Occasional → serverless

Large offline → batch
```

---

# 56. Spot Instances

Spot = spare AWS capacity at potentially lower cost.

Good for:

```text
Training
Batch jobs
Experiments
Fault-tolerant workloads
```

Less suitable when:

```text
Work cannot tolerate interruption
```

---

# 57. Auto Scaling

Example:

```text
Traffic ↑
   ↓
Instances ↑

Traffic ↓
   ↓
Instances ↓
```

Useful metrics:

```text
CPU
Request count
Latency
Custom metrics
```

For ML endpoints:

```text
Inference traffic
      ↓
Auto Scaling
      ↓
More model instances
```

---

# 58. Serverless ML Architecture

For lightweight workloads:

```text
User
 ↓
API Gateway
 ↓
Lambda
 ↓
Model
 ↓
Response
```

For larger models:

```text
User
 ↓
API Gateway
 ↓
Lambda
 ↓
SageMaker Endpoint
 ↓
GPU
 ↓
Model
```

---

# 59. AWS CLI Cheat Sheet

Configure:

```bash
aws configure
```

Check identity:

```bash
aws sts get-caller-identity
```

List regions:

```bash
aws ec2 describe-regions
```

S3:

```bash
aws s3 ls
```

EC2:

```bash
aws ec2 describe-instances
```

IAM:

```bash
aws iam list-roles
```

ECR:

```bash
aws ecr describe-repositories
```

Lambda:

```bash
aws lambda list-functions
```

SageMaker:

```bash
aws sagemaker list-training-jobs
```

```bash
aws sagemaker list-endpoints
```

CloudWatch:

```bash
aws logs describe-log-groups
```

---

# 60. boto3 Cheat Sheet

Install:

```bash
pip install boto3
```

Create client:

```python
import boto3

s3 = boto3.client("s3")
```

SageMaker:

```python
sm = boto3.client("sagemaker")
```

Bedrock Runtime:

```python
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)
```

DynamoDB:

```python
dynamodb = boto3.resource("dynamodb")
```

Lambda:

```python
lambda_client = boto3.client("lambda")
```

---

# 61. AWS Credentials

Local development:

```bash
aws configure
```

Creates credentials under AWS configuration.

Better for production:

```text
EC2 IAM Role
SageMaker Execution Role
ECS Task Role
Lambda Execution Role
```

Avoid:

```python
boto3.client(
    "s3",
    aws_access_key_id="...",
    aws_secret_access_key="..."
)
```

---

# 62. Important ARN Concept

ARN = Amazon Resource Name.

Format:

```text
arn:aws:<service>:<region>:<account>:<resource>
```

Example:

```text
arn:aws:s3:::my-bucket
```

IAM policies frequently reference ARNs.

---

# 63. AWS Regions

Example:

```text
us-east-1
us-west-2
eu-west-1
ap-south-1
ap-south-2
```

Important:

> Not every AWS service/model is available in every Region.

For GenAI especially, always verify **model availability and quotas in the target Region**.

---

# 64. AWS AI/ML Decision Tree

```text
What are you building?
        │
        ├── Traditional ML?
        │       ↓
        │   SageMaker
        │
        ├── Train custom DL model?
        │       ↓
        │   SageMaker / EC2 GPU
        │
        ├── Foundation model application?
        │       ↓
        │   Bedrock
        │
        ├── RAG?
        │       ↓
        │   Bedrock + S3 + Vector DB
        │
        ├── AI Agent?
        │       ↓
        │   Bedrock Agents + tools
        │
        ├── Docker API?
        │       ↓
        │   ECR + ECS/EC2
        │
        └── Simple event function?
                ↓
              Lambda
```

---

# 65. Most Important Comparisons

### S3 vs EBS

```text
S3
→ Object storage
→ datasets/models
→ highly scalable

EBS
→ Block storage
→ attached to EC2
→ OS/application disk
```

### ECS vs EC2

```text
EC2
→ Manage virtual machine

ECS
→ Manage containers
```

### ECS vs EKS

```text
ECS → AWS container orchestration
EKS → Kubernetes
```

### Lambda vs EC2

```text
Lambda
→ serverless
→ short/event-driven workloads

EC2
→ full server
→ more control
→ long-running workloads
```

### Bedrock vs SageMaker

```text
Bedrock → GenAI/FM applications

SageMaker → ML lifecycle/custom models
```

### OpenSearch vs S3

```text
S3
→ store documents

OpenSearch
→ search/retrieve documents/vectors
```

---

# 66. Interview Architecture — ML Prediction API

**Question:** Design an ML prediction service.

Answer:

```text
Client
  ↓
API Gateway
  ↓
Load Balancer
  ↓
ECS/Fargate
  ↓
FastAPI
  ↓
ML Model
  ↓
Prediction
```

Supporting services:

```text
S3 → model/data
ECR → Docker image
CloudWatch → monitoring
IAM → permissions
Secrets Manager → secrets
```

---

# 67. Interview Architecture — RAG

```text
Documents
   ↓
S3
   ↓
Chunking
   ↓
Embedding Model
   ↓
OpenSearch
   ↓
Vector Search
   ↑
User Query
   ↓
Retrieved Context
   ↓
Bedrock LLM
   ↓
Answer
```

Mention:

```text
chunk size
chunk overlap
embedding model
top-k
similarity threshold
metadata filtering
reranking
prompt construction
hallucination evaluation
```

---

# 68. Interview Architecture — Production ML

```text
             GitHub
                │
                ▼
          CI/CD Pipeline
                │
                ▼
              ECR
                │
                ▼
             Training
                │
       ┌────────┴────────┐
       ▼                 ▼
      S3           Model Registry
                         │
                         ▼
                    Deployment
                         │
                         ▼
                      Endpoint
                         │
                         ▼
                       Users
                         │
                         ▼
                    CloudWatch
```

---

# 69. Security Checklist

For production:

```text
☑ IAM roles
☑ Least privilege
☑ Private subnets where appropriate
☑ Security groups
☑ Encryption at rest
☑ HTTPS/TLS
☑ Secrets Manager
☑ CloudTrail
☑ CloudWatch
☑ S3 bucket policies
☑ No credentials in source code
☑ No public S3 unless required
```

---

# 70. AI/ML Engineer AWS Skill Priority

If your goal is **AI/ML Engineer + GenAI**, don't try to learn every AWS service equally.

### 🔴 Must Know

```text
S3
IAM
EC2
Docker
ECR
SageMaker
Bedrock
Lambda
API Gateway
CloudWatch
VPC basics
```

### 🟠 Strongly Recommended

```text
OpenSearch
DynamoDB
SQS
Step Functions
EventBridge
Secrets Manager
ECS
Athena
Glue
```

### 🟡 Later

```text
EKS
Redshift
EMR
Kinesis
Neptune
Lake Formation
Advanced networking
```

---

# 71. Your AI/ML Learning Order

For an AI/ML engineer, I'd learn AWS in this sequence:

```text
1. AWS fundamentals
       ↓
2. IAM
       ↓
3. S3
       ↓
4. EC2
       ↓
5. Docker + ECR
       ↓
6. FastAPI deployment
       ↓
7. SageMaker
       ↓
8. Bedrock
       ↓
9. RAG + OpenSearch
       ↓
10. Lambda + API Gateway
       ↓
11. CloudWatch
       ↓
12. VPC basics
       ↓
13. MLOps
       ↓
14. CI/CD
       ↓
15. Cost + security
```

---

# 72. One Project to Learn Almost Everything

Build:

## **Production RAG AI Assistant on AWS**

```text
                         ┌──────────────┐
                         │   Frontend   │
                         └──────┬───────┘
                                │
                                ▼
                         API Gateway
                                │
                                ▼
                             Lambda
                                │
                                ▼
                       RAG Application
                         ┌──────┴──────┐
                         │             │
                         ▼             ▼
                    OpenSearch       Bedrock
                         ▲             │
                         │             ▼
                         └────────── Answer

Documents
    │
    ▼
   S3
    │
    ▼
Processing
    │
    ▼
Embeddings
    │
    ▼
OpenSearch
```

Add:

```text
Docker
ECR
IAM
CloudWatch
Secrets Manager
CI/CD
```

That one project gives you practical exposure to a **large portion of the AWS stack relevant to modern AI/ML/GenAI roles**.

---

# 73. Final AWS Mental Model

Memorize this:

```text
S3
│
├── Data
├── Models
└── Artifacts

EC2
│
├── CPU
├── GPU
└── Custom workloads

SageMaker
│
├── Training
├── Deployment
├── Pipelines
├── Registry
└── Monitoring

Bedrock
│
├── Foundation Models
├── RAG
├── Agents
├── Embeddings
└── Guardrails

ECR
│
└── Docker Images

ECS/EKS
│
└── Containers

Lambda
│
└── Serverless Functions

API Gateway
│
└── APIs

OpenSearch
│
└── Vector Search

CloudWatch
│
└── Logs + Metrics

IAM
│
└── Permissions

VPC
│
└── Networking

Secrets Manager
│
└── Secrets
```

### The 10 services I'd prioritize for your AI/ML transition

**S3 → IAM → EC2 → Docker/ECR → SageMaker → Bedrock → OpenSearch → Lambda → API Gateway → CloudWatch**

If you can **build and explain one production-style project using these**, rather than just memorizing AWS service definitions, you'll have much stronger material for AI/ML interviews.
