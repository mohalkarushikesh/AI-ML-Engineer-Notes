# Killer Projects

A catalog of end-to-end AI, MLOps, and Data Engineering projects.

## Table of Contents

1. [Bank's Account Payables AI Agent System with Advanced AI Governance](#1-banks-account-payables-ai-agent-system-with-advanced-ai-governance)
2. [MedRAG with LlamaIndex: Clinical Guideline RAG from Ingestion to Deployment](#2-medrag-with-llamaindex-clinical-guideline-rag-from-ingestion-to-deployment)
3. [AI-Powered Web Application with LLM Fine-Tuning, CI/CD Automation and Vibe-Coding](#3-ai-powered-web-application-with-llm-fine-tuning-cicd-automation-and-vibe-coding)
4. [Enterprise Advanced RAG in LangGraph](#4-enterprise-advanced-rag-in-langgraph)
5. [Deployed Clinical Trial Intelligence using GCP and LangGraph](#5-deployed-clinical-trial-intelligence-using-gcp-and-langgraph)
6. [Complete End to End AI Governance Project](#6-complete-end-to-end-ai-governance-project)
7. [Multi-Agent AI Research Platform with AWS Guardrails](#7-multi-agent-ai-research-platform-with-aws-guardrails)
8. [Insurance Claim Support AI Agent with LangMem and RAG](#8-insurance-claim-support-ai-agent-with-langmem-and-rag)
9. [Travel Planning Multi Agent with LangGraph](#9-travel-planning-multi-agent-with-langgraph)
10. [End-to-End Realtime Flight Data Engineering with Airflow & Snowflake](#10-end-to-end-realtime-flight-data-engineering-with-airflow--snowflake)
11. [MLOPS Jenkins Shared Library CI-CD Project](#11-mlops-jenkins-shared-library-ci-cd-project)
12. [AI Powered Job Analyzer using Filebeat with ELK Stack and Kubernetes](#12-ai-powered-job-analyzer-using-filebeat-with-elk-stack-and-kubernetes)
13. [TaxPilot — Autonomous Tax Prep & Compliance Copilot with RAG, Audit-Risk Scoring & Guardrails]
---

## 1. Bank's Account Payables AI Agent System with Advanced AI Governance

🔗 https://www.krishnaik.in/project/sss
📚 21 lectures

Custodian is a governed multi-agent finance-ops platform: AI agents read invoices, score them for fraud/risk, route approvals, and auto-pay the safe ones. Six real governance layers wrap the agent runtime — Identity, Data, Model, Policy, Agent Runtime, and Operations — so every agent action is auditable and provable after the fact. It's a full Docker stack (Keycloak, SPIRE, Infisical, OpenMetadata, Presidio, LiteLLM, MLflow, Langfuse, Prometheus/Grafana) calling real LLMs (OpenAI + Groq via LiteLLM), not a mocked demo.

**What You Will Learn**
- How to build and orchestrate a real multi-agent AI system that makes autonomous decisions (invoice approval/payment) instead of just chatting.
- How to wrap AI agents in production-grade governance — identity, data privacy, policy enforcement, and auditability — so their actions are trustworthy and provable.
- How to run a full real-world microservices + observability stack (Docker, LiteLLM, MLflow, Langfuse, Prometheus/Grafana) integrating actual LLM providers, not a toy demo.

**What You'll Build**
- A multi-agent AI system that reads invoices, scores them for fraud/risk, routes approvals, and automatically pays the safe ones.
- Six governance layers — Identity, Data, Model, Policy, Agent Runtime, Operations — that watch every agent action and can prove after the fact exactly what happened and why.
- A full production-style stack (console UI, control plane, ledger, OCR/data services) wired to real LLMs via LiteLLM, with observability through MLflow, Langfuse, Prometheus, and Grafana.

---

## 2. MedRAG with LlamaIndex: Clinical Guideline RAG from Ingestion to Deployment

🔗 https://www.krishnaik.in/project/medrag-clinical-guideline-qa-with-retrieval-augmented-generation
📚 33 lectures

Build a production-ready clinical-guideline question-answering system with LlamaIndex powering document ingestion, indexing, retrieval, and grounded generation. Students use LlamaParse and PubMedReader to create LlamaIndex documents, then chunk and embed them into a Qdrant-backed VectorStoreIndex queried through an OpenAI-powered LlamaIndex query engine. The project completes the system with evaluations, FastAPI and Streamlit interfaces, Docker CI/CD deployment, and API-boundary input and output guardrails.

**What You Will Learn**
- Design a reusable, domain-agnostic RAG core with a MedRAG-specific project plugin.
- Build clinical-document ingestion, indexing, vector retrieval, and grounded answer generation workflows.
- Create shared evaluation modules to measure retrieval and generation quality.
- Expose one RAG service through CLI, FastAPI, and Streamlit interfaces.
- Containerize and deploy the application through CI/CD while implementing input and output safety guardrails.

**What You'll Build**
- A clinical-guideline ingestion and indexing pipeline backed by a vector database.
- A modular MedRAG retrieval and answer-generation service with source-grounded responses.
- CLI, FastAPI, and Streamlit interfaces powered by the same reusable RAG service.
- An evaluation harness with shared metrics and diagnostic workflows.
- A Dockerized CI/CD deployment with API-boundary input and output guardrails.

---

## 3. AI-Powered Web Application with LLM Fine-Tuning, CI/CD Automation and Vibe-Coding

🔗 https://www.krishnaik.in/project/bs
📚 5 lectures

Built an end-to-end Generative AI application by fine-tuning an LLM using Azure AI Foundry and integrating it into a React-based web interface. Implemented automated CI/CD deployment using AWS CodePipeline and hosted the application on S3 for scalable cloud delivery. Focused on production-ready AI workflows combining GenAI, cloud, and DevOps practices.

**What You Will Learn**
- How to build an end-to-end Generative AI application by fine-tuning an LLM using Azure AI Foundry and integrating it into a React-based frontend.
- Cloud deployment and DevOps practices by implementing CI/CD pipelines using AWS CodePipeline and deploying applications on S3.

**What You'll Build**
- A full-stack Generative AI web application with a React frontend integrated with a fine-tuned LLM using Azure AI Foundry.
- An automated cloud deployment pipeline using AWS CodePipeline to deploy and host the application on Amazon S3.

---

## 4. Enterprise Advanced RAG in LangGraph

*Enterprise Advanced RAG with Hybrid Search, ReRanking, HyDE, CRAG, Self-RAG, Text2SQL, Caching and Guardrails in LangGraph*

🔗 https://www.krishnaik.in/project/enterprise-advanced-rag-with-hybrid-search-reranking-hyde-crag-sel… *(link truncated in source — replace with full URL)*
📚 39 lectures

Build a production-grade Enterprise RAG system for Kubernetes IT operations using LangGraph, FastAPI, Qdrant, PostgreSQL, Redis caching, and advanced retrieval patterns. Learners start from baseline RAG, then add hybrid search, reranking, HyDE, CRAG, Self-RAG, Text2SQL with human approval, evaluation, and a 9-layer guardrails pipeline.

**What You Will Learn**
- Advanced RAG design, hybrid search, reranking, HyDE, CRAG, Self-RAG, Text2SQL.
- LangGraph orchestration, caching, evaluation, and guardrails.

**What You'll Build**
- A production-grade Kubernetes SRE copilot with FastAPI, LangGraph, Qdrant retrieval, PostgreSQL Text2SQL, Redis caching, Streamlit UI, Ragas evals, and security layers.

---

## 5. Deployed Clinical Trial Intelligence using GCP and LangGraph

🔗 https://www.krishnaik.in/project/deployed-clinical-trial-intelligence-using-gcp-and-langgraph
📚 39 lectures

MOSAIC is a production multi-agent clinical trial intelligence system built on GCP. It ingests real records from ClinicalTrials.gov and PubMed into Cloud SQL PostgreSQL with pgvector, then uses LangGraph to run six specialist agents in parallel under a supervisor — Broken Promises, Missing Results, Track Record, Pattern Finder, Side Effect Checker, Timeline Analyst — each hunting a different research-integrity problem. Findings go into a three-layer memory system (episodic, procedural, semantic), and low-confidence signals hit a human-in-the-loop gate whose corrections write back into procedural memory, so agents improve on the next run. The whole thing is served through a nine-endpoint FastAPI app, containerised, and deployed on Cloud Run, with Cloud Storage, Artifact Registry, Secret Manager, IAM service accounts and gcloud-driven provisioning behind it. It solves a genuine problem — ~30% of completed trials never publish results, and no human can read 400,000 records at once — and it produced real signals from live government data before the project was torn down to zero billing.

**What You Will Learn**
- Design multi-agent systems in LangGraph with a supervisor, parallel fan-out, and shared state.
- Build three-layer agent memory (episodic, procedural, semantic) and a human-in-the-loop learning loop.
- Run vector search on Cloud SQL Postgres with pgvector over real public API data.
- Wrap it all in FastAPI, containerise it, and deploy to Cloud Run with Secret Manager and IAM.
- Apply production patterns like `lru_cache` dependency injection, connection pooling, and state reducers.
- Audit and tear down cloud resources so the bill stays near zero.

**What You'll Build**
- A deployed Agentic AI system on GCP using government databases for Clinical Trials.

---

## 6. Complete End to End AI Governance Project

🔗 https://www.krishnaik.in/project/fvdd
📚 22 lectures

A complete end-to-end AI Governance System using all open-source alternatives, building a Multi-AI-Agent Chatbot with all governance layers — Identity Governance for humans and AI agents, Data Governance, Model Governance, Policy Governance, Agent Runtime Governance, Operations Governance, and Compliance Governance.

**What You Will Learn**
- How real AI governance controls work in practice — identity verification, policy enforcement, data/PII protection, model approval gating, guardrails, and audit logging — not just the theory.
- How a multi-agent system (LangGraph) enforces authorization at every layer, including agent-to-agent handoffs and individual tool calls, so unauthorized actions are blocked before they execute.

**What You'll Build**
- A working multi-agent AI system (order/billing/admin specialists) backed by a real Postgres database with least-privilege roles, orchestrated as an explicit LangGraph state graph.
- A full governance stack — Keycloak/SPIFFE identity, OPA policy engine, OpenMetadata data catalog, MLflow model registry, Guardrails AI, and Langfuse tracing.
- A live Streamlit dashboard showing all 7 governance checks pass/fail per request, plus a compliance report generator that reads real logs to prove enforcement.

---

## 7. Multi-Agent AI Research Platform with AWS Guardrails

*Multi-Agent AI Research Platform with AWS Guardrails, LLM Gateway, Red Teaming, STM, LTM & Semantic Caching*

🔗 https://www.krishnaik.in/project/jljmasl
📚 21 lectures

A production-grade autonomous research platform where a 4-agent LangGraph pipeline (Search → Summarize → Write → Verify) processes any topic end-to-end, with every request passing through AWS Bedrock Guardrails, a TensorZero LLM gateway with GPT-4o/Groq fallback, and a three-tier memory system — Redis session memory (STM), pgvector long-term memory (LTM), and semantic caching. Every report is automatically scored by an LLM-as-judge via LangSmith, while a PyRIT red team dashboard continuously stress-tests the system with jailbreak, XPIA, crescendo, and skeleton key attacks to prove the guardrails hold under real adversarial pressure. Full infrastructure on AWS, provisioned with Terraform, deployed via GitHub Actions CI/CD.

**What You Will Learn**
- How to build a production-grade multi-agent AI pipeline — a LangGraph 4-agent workflow (Search → Summarize → Write → Verify) with an LLM gateway, automatic model fallback, and layered memory (Redis STM, pgvector LTM, semantic cache).
- How to secure an AI system end-to-end — AWS Bedrock Guardrails for input/output safety, API authentication and rate limiting, and automated adversarial red team attacks (jailbreak, XPIA, crescendo, skeleton key) using PyRIT.
- How to ship AI infrastructure like a professional team — a complete AWS stack (ECS, RDS, ElastiCache, ALB, Secrets Manager, ECR, VPC) with Terraform, GitHub Actions CI/CD with automatic rollback, and monitoring via LangSmith tracing and LLM-as-judge evaluation.

**What You'll Build**
- A 4-agent autonomous research pipeline producing a fully written, verified report — exportable as text, PDF, or structured JSON.
- A layered AI memory and caching system — Redis (STM), PostgreSQL with pgvector (LTM), and a semantic cache that skips the full pipeline for similar topics, saving cost and latency.
- A live red team security dashboard — PyRIT-powered attack dashboard running four adversarial attack types against your running system, with weekly scheduling via EventBridge, and AWS Bedrock Guardrails defending every request in real time.

---

## 8. Insurance Claim Support AI Agent with LangMem and RAG

🔗 https://www.krishnaik.in/project/insurance-claims-copilot-with-memory-and-tool-calling
📚 19 lectures

Build an intelligent copilot application for insurance claims support, leveraging memory and tool-calling capabilities to provide personalized and context-aware assistance. This project enhances support efficiency by enabling the system to remember user interactions, retrieve relevant information, and utilize external tools for informed decision-making.

**What You Will Learn**
- Mastering LangChain for building memory-augmented applications.
- Implementing memory modules using LangMem.
- Utilizing tool calling to ground agent responses with structured data.
- Building modular and scalable applications with FastAPI.
- Deploying applications using Docker and AWS.

**What You'll Build**
- A copilot application for insurance claims support.
- Memory modules for personalized user interactions.
- Integration of external tools for data retrieval and decision-making.
- A CI/CD pipeline for automated deployment to AWS.

---

## 9. Travel Planning Multi Agent with LangGraph

🔗 https://www.krishnaik.in/project/travel-planning-multi-agent-with-langgraph
📚 9 lectures

Built a Travel Planning Multi-Agent System using LangGraph that intelligently coordinates multiple AI agents to create personalized travel plans. The system handles destination research, itinerary generation, recommendations, and planning through an agentic workflow.

**What You Will Learn**
- Understand the architecture of Multi-Agent AI systems.
- Build and orchestrate multiple AI agents using LangGraph.
- Implement agent communication and workflow management.
- Integrate external tools/APIs with AI agents.
- Manage shared state and generate coordinated final responses.

**What You'll Build**
- An end-to-end AI-powered Travel Planning Multi-Agent System.
- Specialized agents for different travel-planning tasks.
- A LangGraph-based agent orchestration workflow.
- A tool/API-integrated travel assistant.
- A complete system that generates personalized travel plans.

---

## 10. End-to-End Realtime Flight Data Engineering with Airflow & Snowflake

🔗 https://www.krishnaik.in/project/nkdn
📚 9 lectures

This project implements an end-to-end flight data engineering pipeline using Apache Airflow for orchestration and Snowflake as the data warehouse, following a medallion architecture pattern with bronze, silver, and gold data layers. The bronze layer ingests raw flight data from JSON files, the silver layer performs data cleansing and transformation, and the gold layer aggregates insights for analytics, all automated through scheduled Airflow DAGs. Built with Python scripts for each layer and Docker Compose for containerized deployment, this solution ensures scalable, reliable data processing from ingestion to warehousing, supporting real-time flight operations analytics.

**What You Will Learn**
- Design, implement, and manage end-to-end ETL/ELT pipelines using Apache Airflow, including task dependencies, scheduling, and error handling in a production-like environment.
- Hands-on experience with medallion architecture (bronze/silver/gold layers) and cloud data warehousing using Snowflake, including data ingestion, transformation, aggregation, and loading techniques.

**What You'll Build**
- An end-to-end ETL pipeline that ingests real-time flight data from the OpenSky Network API, processes it through bronze (raw), silver (cleaned), and gold (aggregated) layers, and loads it into Snowflake.
- A containerized data architecture using Docker Compose, featuring Apache Airflow for orchestration, PostgreSQL for metadata, and Snowflake as the cloud data warehouse, supporting scheduled batch processing and manual runs.

---

## 11. MLOPS Jenkins Shared Library CI-CD Project

🔗 https://www.krishnaik.in/project/nknk
📚 18 lectures

This project demonstrates a professional MLOps workflow by centralizing automation logic within a Jenkins Shared Library, ensuring that every build and deployment is standardized and reusable. By leveraging this library, the pipeline automatically handles everything from Docker image creation to Kubernetes orchestration on a GCP-hosted VM, turning complex manual steps into a single, scalable process.

**What You Will Learn**
- Master Jenkins Shared Libraries to write reusable, modular code that simplifies complex CI/CD pipelines.
- Bridge the gap between code and production by automating container builds and Kubernetes orchestration.

**What You'll Build**
- A centralized, reusable code repository to standardize and automate your entire CI/CD pipeline logic.
- A cloud-based environment on GCP featuring a functional Minikube cluster to orchestrate and host your containerized applications.
- A workflow that automatically handles code checkout, Docker image creation, and pushing to DockerHub.

---

## 12. AI Powered Job Analyzer using Filebeat with ELK Stack and Kubernetes

🔗 https://www.krishnaik.in/project/fmlml
📚 19 lectures

The AI Powered Job Analyzer is a cloud-native application that leverages GPT-4 to automatically screen resumes against job descriptions for hiring accuracy. Deployed on a Kubernetes cluster, it integrates a full ELK stack (Filebeat, Logstash, Elasticsearch, Kibana) to provide robust, real-time logging and system observability.

**What You Will Learn**
- Hands-on experience building a professional ELK Stack pipeline for centralized log management and real-time dashboard visualization.
- Containerizing applications and managing them within a Kubernetes cluster on a Virtual Machine.

**What You'll Build**
- An intelligent application that parses resumes and JDs to generate automated hiring decisions, match percentages, and gap analysis using GPT-4.
- A professional ELK Stack (Filebeat to Kibana) that captures application logs, processes them, and visualizes system health on a real-time dashboard.

---

## Here's a tax-domain AI/ML project designed to fit right into your catalog — same production-grade, governance-heavy, multi-agent style as Custodian and the RAG projects.

---

## 13. TaxPilot — Autonomous Tax Prep & Compliance Copilot with RAG, Audit-Risk Scoring & Guardrails

🔗 *(placeholder — e.g. https://www.krishnaik.in/project/taxpilot)*
📚 ~25 lectures

**Overview**

TaxPilot is a production-style multi-agent system that turns a shoebox of tax documents into a reviewed, explainable draft return. AI agents ingest financial documents (W-2, 1099-NEC/INT/DIV, K-1, receipts), extract and normalize the data via OCR, compute tax liability, discover eligible deductions and credits by grounding every answer in current IRS publications through RAG, score the return's audit risk, flag anomalies, and route anything uncertain to a human preparer. Because tax is high-stakes and regulated, every figure the system produces is **traceable to a source rule and a source document** — no ungrounded "the model said so."

It's a full stack — LangGraph orchestration, a vector store over IRS/tax-code corpora, a deterministic tax-calculation engine, a Streamlit/FastAPI console, and a guardrails + audit layer — not a chatbot that guesses numbers.

**Why it matters**

LLMs *cannot* be trusted to do arithmetic or to "know" tax law from memory — both change yearly and both carry legal liability. The whole design lesson is **separating what the LLM is good at (reading messy documents, retrieving and explaining rules, spotting anomalies) from what must be deterministic and grounded (the actual math and the cited authority).** That split is exactly the production skill this project teaches.

**System Architecture (agent pipeline)**

`Intake → Extractor → Classifier → Deduction Researcher → Calculator → Audit-Risk Scorer → Reviewer (human-in-the-loop) → Report`

- **Intake Agent** — accepts uploaded docs, OCRs them (Tesseract/textract), routes by type.
- **Extractor Agent** — pulls structured fields (wages, withholding, interest, contractor income) with confidence scores; low confidence → flag.
- **Classifier Agent** — determines filing status, dependents, income categories.
- **Deduction & Credit Researcher (RAG)** — retrieves relevant rules from an IRS-publication vector store, proposes deductions/credits *with citations* to the governing rule.
- **Calculator (deterministic, NOT the LLM)** — a Python tax-rules engine computes liability; the LLM never does the math.
- **Audit-Risk Scorer** — an ML model / heuristic layer scores the return against red-flag patterns (unusually high deduction ratios, mismatched 1099 totals, round-number expenses).
- **Reviewer Gate** — anything low-confidence, high-risk, or above a dollar threshold is escalated to a human; corrections write back to improve future runs.

**What You Will Learn**
- How to build a multi-agent LangGraph pipeline that *separates probabilistic reasoning from deterministic computation* — the LLM reads and explains, a rules engine does the math.
- How to ground high-stakes domain answers in authoritative sources with RAG so every de
