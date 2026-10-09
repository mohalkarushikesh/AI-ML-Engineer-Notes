# Terraform Cheat Sheet

Defination: Terraform is an Infrastructure as Code (IaC) tool created by HashiCorp that lets you build, change, and version cloud and on-premises infrastructure safely and efficiently

## Core Workflow Commands
```bash
terraform init          # Initialize working directory, download providers & modules
terraform validate      # Check configuration syntax and structural correctness
terraform fmt           # Format all .tf files in the directory to standard style
terraform plan          # Preview execution plan (dry run) showing infrastructure changes
terraform apply         # Deploy configuration and apply changes to the cloud infrastructure
terraform apply -auto-approve  # Apply modifications without prompting for manual confirmation
terraform destroy       # Remove all managed infrastructure resources permanently
```

## State Management
```bash
terraform show          # Display human-readable text of the current state or a plan
terraform state list    # List all resources currently tracked by the state file
terraform state show <addr>  # Show detailed attributes of a specific tracked resource
terraform state rm <addr>    # Remove a resource from state without destroying real infrastructure
terraform state mv <old> <new>  # Rename a resource address or move it into a module
terraform state pull    # Download and output the state file to stdout
```

## Configuration & Maintenance
```bash
terraform output        # Read and print defined output values from the state file
terraform graph         # Generate a visual dependency graph of resources (Graphviz format)
terraform console       # Open an interactive command-line shell to evaluate HCL expressions
terraform get           # Download or update remote module source code
terraform providers     # Display a tree of providers used in the current configuration
```

## Advanced Operations & Troubleshooting
```bash
terraform import <addr> <id>  # Map an existing, unmanaged cloud resource into Terraform state
terraform plan -refresh-only  # Detect configuration drift against real infrastructure status
terraform force-unlock <id>   # Release a stuck lock on the state file using its Lock ID
export TF_LOG=TRACE          # Enable verbose debugging logs (TRACE, DEBUG, INFO, WARN, ERROR)
```

## Workspace Management
```bash
terraform workspace list     # List all existing CLI workspaces
terraform workspace new <name>  # Create a brand new workspace environment
terraform workspace select <name>  # Switch current context to a different workspace
```

## HCL Syntax Snippets

### 1. Provider & Resource
```hcl
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}
```

### 2. Variables & Outputs
```hcl
variable "instance_type" {
  type        = string
  default     = "t2.micro"
  description = "EC2 instance size"
}

output "instance_ip" {
  value       = aws_instance.web.public_ip
  description = "The public IP of the web server"
}
```

### 3. Remote Backend (S3 with Locking)
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state-bucket"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-lock-table"
    encrypt        = true
  }
}
```
