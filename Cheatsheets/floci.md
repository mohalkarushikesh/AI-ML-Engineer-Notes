# 🚀 Floci AWS Emulator Cheatsheet

This in-depth **Floci Cheatsheet** covers CLI commands, environment variables, data persistence, direct AWS service configurations, and SDK integrations for the **Floci AWS Emulator**.

**References:**

* [Floci AWS](https://floci.io/aws/)
* [Floci CLI GitHub](https://github.com/floci-io/floci-cli)
* [Floci Getting Started](https://floci.io/floci/getting-started/quick-start/)

---

## 🚀 1. Core Lifecycle Commands

Manage the lifecycle of the local AWS emulator using the `floci` binary.

| Command               | Description                                                                  |
| --------------------- | ---------------------------------------------------------------------------- |
| `floci doctor`        | Runs local environment diagnostics (Docker access, paths).                   |
| `floci start`         | Launches the Floci AWS emulator container on background port `4566`.         |
| `floci env`           | Prints the dummy local AWS environment configurations.                       |
| `eval $(floci env)`   | Sets the active shell environment variables to the local endpoint.           |
| `floci status`        | Displays container state, server runtime stats, and health.                  |
| `floci logs --follow` | Streams active Docker logs from the emulator.                                |
| `floci stop`          | Tears down and cleanly removes the active emulator container.                |
| `floci wait`          | Blocks execution until services pass health checks; useful for CI pipelines. |

### ⚡ Typical Startup

```bash
floci doctor
floci start
eval $(floci env)
floci status
```

---

## ☁️ 2. Multi-Cloud Expansion

Floci supports multi-cloud environments under the same binary interface using cloud-specific commands.

### Google Cloud Platform

```bash
floci gcp start
eval $(floci gcp env)
```

**Default port:** `4588`

### Microsoft Azure

```bash
floci az start
eval $(floci az env)
```

**Default port:** `4577`

### Oracle Cloud Infrastructure

```bash
floci oci start
floci oci setup
eval $(floci oci env)
```

**Default port:** `4599`

---

## 💾 3. Persistence & Snapshots

By default, Floci stores state in memory for fast development.

### Enable Persistent Storage

```bash
floci start --persist ./data
```

### Save a Snapshot

```bash
floci snapshot save microservices-v1
```

### Restore a Snapshot

```bash
floci snapshot restore microservices-v1
```

### 📌 Persistence Workflow

```bash
floci start --persist ./data
# Create your resources...

floci snapshot save microservices-v1

# Later...
floci snapshot restore microservices-v1
```

---

## 🛠️ 4. AWS CLI Integration

After running:

```bash
eval $(floci env)
```

you can use the standard AWS CLI commands against the local Floci environment.

If you don't use the environment hook, manually specify:

```bash
--endpoint-url http://localhost:4566
```

---

### 🪣 Amazon S3

#### Create a Bucket

```bash
aws s3 mb s3://my-test-bucket
```

#### Upload a File

```bash
aws s3 cp document.json s3://my-test-bucket/
```

#### List Bucket Contents

```bash
aws s3 ls s3://my-test-bucket/
```

---

### 📬 Amazon SQS

#### Create a Queue

```bash
aws sqs create-queue \
  --queue-name processing-queue
```

#### Send a Message

```bash
aws sqs send-message \
  --queue-url http://localhost:4566/000000000000/processing-queue \
  --message-body '{"status": "pending", "id": 1045}'
```

---

### 🗄️ Amazon DynamoDB

#### Create a Table

```bash
aws dynamodb create-table \
  --table-name UserProfile \
  --attribute-definitions AttributeName=UserId,AttributeType=S \
  --key-schema AttributeName=UserId,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

---

## 🧩 5. Code Client Initialization

You can connect your applications directly to Floci by overriding the AWS service endpoint.

### 🐍 Python — Boto3

```python
import boto3


def get_floci_client(service_name):
    return boto3.client(
        service_name,
        endpoint_url="http://localhost:4566",
        region_name="us-east-1",
        aws_access_key_id="mock-key",
        aws_secret_access_key="mock-secret"
    )


s3_client = get_floci_client("s3")
```

---

### 🟨 Node.js — AWS SDK v3

```javascript
const { S3Client } = require("@aws-sdk/client-s3");

const s3 = new S3Client({
  endpoint: "http://localhost:4566",
  region: "us-east-1",
  credentials: {
    accessKeyId: "mock",
    secretAccessKey: "mock"
  },
  forcePathStyle: true
});
```

> **Note:** `forcePathStyle: true` is used for local S3 endpoint resolution.

---

## 🌐 6. Visual Console

Floci provides a local Web Console UI for inspecting active resources.

### Primary Console

```text
http://localhost:4566/_floci/ui
```

### Fallback Console

```text
http://localhost:4500/console/aws
```

Open the URL in your browser after starting Floci to inspect your local AWS resources.

---

## ⚡ 7. Quick Start

For a basic local AWS development environment:

```bash
# Check environment
floci doctor

# Start Floci
floci start

# Configure AWS CLI
eval $(floci env)

# Wait for services
floci wait

# Check status
floci status
```

Then test S3:

```bash
aws s3 mb s3://my-test-bucket
aws s3 cp document.json s3://my-test-bucket/
aws s3 ls s3://my-test-bucket/
```

---

## 📌 Quick Reference

| Category         | Command / URL                     |
| ---------------- | --------------------------------- |
| Diagnose         | `floci doctor`                    |
| Start            | `floci start`                     |
| Environment      | `eval $(floci env)`               |
| Status           | `floci status`                    |
| Logs             | `floci logs --follow`             |
| Stop             | `floci stop`                      |
| Health Check     | `floci wait`                      |
| AWS Endpoint     | `http://localhost:4566`           |
| Web Console      | `http://localhost:4566/_floci/ui` |
| Persistent Data  | `floci start --persist ./data`    |
| Snapshot Save    | `floci snapshot save <name>`      |
| Snapshot Restore | `floci snapshot restore <name>`   |

---

## 🔗 References

* [Floci AWS](https://floci.io/aws/)
* [Floci CLI](https://github.com/floci-io/floci-cli)
* [Floci GitHub](https://github.com/floci-io/floci)
* [Floci Quick Start](https://floci.io/floci/getting-started/quick-start/)
* [Floci AWS Setup](https://floci.io/floci/getting-started/aws-setup/)
