This in-depth Floci Cheatsheet covers CLI commands, environment variables, data persistence, and direct service configurations for the Floci AWS Emulator. [1] (https://floci.io/aws/), [2] (https://github.com/floci-io/floci-cli), [3] (https://floci.io/aws/)
🚀 1. Core Lifecycle Commands
Manage the lifecycle of the local AWS emulator using the floci binary. [1] (https://github.com/floci-io/floci-cli)
Command	Description
floci doctor	Runs local environment diagnostics (Docker access, paths).
floci start	Launches the Floci AWS emulator container on background port 4566.
floci env	Prints the dummy local AWS environment configurations.
eval $(floci env)	Crucial: Sets the active shell environment variable pointers to your local endpoint.
floci status	Displays container state, server runtime stats, and health.
floci logs --follow	Streams active Docker logs from the emulator.
floci stop	Tears down and cleanly removes the active emulator container.
floci wait	Blocks execution loop until services pass health-checks (ideal for CI pipelines).
💾 2. Multi-Cloud Expansion
Floci isn't limited to AWS; it provides multi-cloud cross-compatibility under the same binary interface using prefixed targeting flags. [1] (https://github.com/floci-io/floci-cli)
bash
# Google Cloud Platform (Port 4588)
floci gcp start && eval $(floci gcp env)

# Microsoft Azure (Port 4577)
floci az start && eval $(floci az env)

# Oracle Cloud Infrastructure (Port 4599)
floci oci start && floci oci setup && eval $(floci oci env)
Use code with caution.
📦 3. Persistence & Snapshots
By default, Floci drops everything to memory on termination for speed. Use these patterns to persist development data across stack teardowns. [1] (https://github.com/floci-io/floci), [2] (https://floci.io/aws/)
• Launch with disk storage:bash
floci start --persist ./data
Use code with caution.
• Capture a static mock dataset snapshot:bash
floci snapshot save microservices-v1
Use code with caution.
• Restore state immediately:bash
floci snapshot restore microservices-v1
Use code with caution.
🛠️ 4. Direct AWS CLI Integrations
Once eval $(floci env) is hooked, use standard aws syntax. If you are not utilizing the automated environment hook, manually append --endpoint-url http://localhost:4566 to your queries. [1] (https://floci.io/floci/getting-started/quick-start/), [2] (https://github.com/floci-io/floci-cli), [3] (https://floci.io/aws/)
Simple Storage Service (S3) [1] (https://floci.io/floci/getting-started/quick-start/)
bash
# Create local bucket
aws s3 mb s3://my-test-bucket

# Upload a local file 
aws s3 cp document.json s3://my-test-bucket/

# List contents
aws s3 ls s3://my-test-bucket/
Use code with caution.
Simple Queue Service (SQS) [1] (https://floci.io/floci/getting-started/quick-start/)
bash
# Create a standard queue
aws sqs create-queue --queue-name processing-queue

# Send an arbitrary JSON body payload
aws sqs send-message \
  --queue-url http://localhost:4566/000000000000/processing-queue \
  --message-body '{"status": "pending", "id": 1045}'
Use code with caution.
DynamoDB Tables [1] (https://floci.io/floci/getting-started/quick-start/)
bash
# Create table with a primary key hash
aws dynamodb create-table \
  --table-name UserProfile \
  --attribute-definitions AttributeName=UserId,AttributeType=S \
  --key-schema AttributeName=UserId,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
Use code with caution.
🧩 5. Code Client Initialization
To link code directly to Floci, override the target runtime URLs to target the emulator endpoint. [1] (https://floci.io/floci/getting-started/aws-setup/), [2] (https://github.com/floci-io/floci)
Python (boto3) [1] (https://floci.io/floci/getting-started/aws-setup/)
python
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
Use code with caution.
Node.js (AWS SDK v3)
javascript
const { S3Client } = require("@aws-sdk/client-s3");

const s3 = new S3Client({
  endpoint: "http://localhost:4566",
  region: "us-east-1",
  credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
  forcePathStyle: true // Mandatory flag for local S3 resolution
});
Use code with caution.
🌐 6. Visual Console
Floci has an embedded Web Console UI available right on your local port layer. Navigate your browser directly to inspect active resources interactively: [1] (https://floci.io/)
👉 http://localhost:4566/_floci/ui (or fallback address http://localhost:4500/console/aws) [1] (https://floci.io/)
Would you like me to write a Docker Compose template incorporating Floci, or do you need a custom Terraform provider block configuration to map infrastructure as code onto this local setup?
