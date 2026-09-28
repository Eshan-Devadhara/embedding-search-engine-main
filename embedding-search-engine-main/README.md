# Bedrock Embedding Search Engine

A lightweight Python Flask application that performs **semantic search** using
[Amazon Bedrock](https://aws.amazon.com/bedrock/) (Titan Text Embeddings V2) and
cosine similarity.  Designed to run as a Docker container on
**Amazon ECS Fargate** behind an **Application Load Balancer** and **Route 53**.

---

## What it Does

1. At startup, loads `subfeatures_dataset.csv` and generates an embedding for
   every row using **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).
2. Caches those embeddings locally (`embeddings_cache.json`) so Bedrock is only
   called once per dataset row, not per search.
3. For every `/search` request it embeds the query (one Bedrock call) and ranks
   all dataset rows by cosine similarity, returning the top 5.

---

## Architecture

```
User
  ↓ HTTPS
Route 53  (DNS)
  ↓
Application Load Balancer  (port 443/80 → 5000)
  ↓
ECS Fargate Task  (Docker container, port 5000)
  ↓
Flask app (app.py / search.py)
  ↓
Amazon Bedrock Runtime  (IAM task role — no static keys)
  ↓
Titan Text Embeddings V2  →  cosine similarity  →  top-5 results
```

Authentication to Bedrock is provided by the **ECS task IAM role** via boto3's
standard credential provider chain.  No API keys are stored anywhere.

---

## File Structure

```
embedding-search-engine/
├── app.py                   Flask entry point + routes
├── search.py                Index init, cosine search logic
├── bedrock.py               Bedrock client + embedding function
├── subfeatures_dataset.csv  Dataset (43 rows)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── deploy.sh                ECR push helper
├── README.md
└── tests/
    └── test_app.py
```

---

## Prerequisites

| Tool | Notes |
|------|-------|
| Python 3.12+ | `python --version` |
| pip | bundled with Python |
| AWS CLI | for ECR / ECS operations |
| Docker | for building / running containers |
| AWS account | with Bedrock enabled in your target region |

---

## Local Setup

```bash
# 1. Clone
git clone <repo-url>
cd embedding-search-engine

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure AWS credentials for local development
#    Option A — AWS CLI profile (recommended for local dev)
aws configure

#    Option B — environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# 5. (Optional) copy .env.example and set variables
cp .env.example .env
# edit .env — set AWS_REGION at minimum
```

---

## AWS Configuration

### Enable Amazon Bedrock

1. Open the [AWS Console → Bedrock → Model access](https://console.aws.amazon.com/bedrock/home#/modelaccess).
2. Request access to **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).
3. Approval is usually instant for Titan models.

### IAM Permissions

The application needs exactly **one IAM permission** at runtime:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
    }
  ]
}
```

Attach this policy to the **ECS task role** (not the task execution role).

---

## Bedrock Model

| Setting | Value |
|---------|-------|
| Model ID | `amazon.titan-embed-text-v2:0` |
| Dimensions | 1024 (default) |
| Normalize | true |
| Override env var | `BEDROCK_MODEL_ID` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | *(boto3 auto-detects)* | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `PORT` | `5000` | Flask listen port |

---

## How to Run Locally

```bash
# Standard start (loads or generates embeddings, then serves)
python app.py

# Force rebuild of the embedding index (re-embeds all dataset rows)
python app.py --rebuild-index
```

---

## API Endpoints

### Health Check

```
GET /health
```

Returns HTTP 200 — used by the ALB target group health check.

```json
{"status": "healthy"}
```

### Semantic Search

```
GET /search?description=<query>
```

Returns the top 5 most semantically similar subfeatures.

**Example:**

```bash
curl "http://localhost:5000/search?description=How+can+I+extract+text+from+a+receipt"
```

**Response:**

```json
[
  {
    "name": "ocr",
    "score": 0.912,
    "description": "Extracts text from images using optical character recognition."
  },
  ...
]
```

**Error responses:**

| Situation | HTTP | Body |
|-----------|------|------|
| Missing description | 400 | `{"error": "description query parameter is required"}` |
| Empty description | 400 | `{"error": "description query parameter is required"}` |
| Bedrock failure | 500 | `{"error": "Search failed. Check server logs for details."}` |

---

## Testing

```bash
python -m pytest tests/ -v
```

Tests mock the Bedrock client — no AWS credentials needed to run tests.

---

## Docker

### Build

```bash
docker build -t bedrock-search-engine .
```

### Run locally (with AWS credentials)

```bash
docker run -p 5000:5000 \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id)" \
  -e AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key)" \
  bedrock-search-engine
```

> **In ECS Fargate** you do NOT pass credentials — the task IAM role handles it.

### Test the running container

```bash
curl http://localhost:5000/health
curl "http://localhost:5000/search?description=moderation"
```

---

## ECR Push

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REPOSITORY=bedrock-search-engine

# Create the ECR repo (first time only)
aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION

# Build + tag + push
bash deploy.sh
```

---

## ECS Fargate Deployment Overview

> Complete these steps manually in the AWS Console or with the CLI.

1. **Task Definition** — create a new ECS task definition:
   - Launch type: Fargate
   - Container image: `<account-id>.dkr.ecr.<region>.amazonaws.com/bedrock-search-engine:latest`
   - Port mapping: `5000`
   - Task role: IAM role with `bedrock:InvokeModel` permission
   - CPU / Memory: 0.5 vCPU / 1 GB is sufficient for the lab

2. **ECS Cluster** — create a cluster (Fargate / serverless).

3. **ECS Service** — create a service pointing to your task definition.
   - Attach to the VPC and subnets where your ALB lives.

4. **Application Load Balancer**:
   - Create a target group (protocol HTTP, port 5000, health check path `/health`).
   - Register the ECS service with the target group.
   - Create an ALB listener on port 80 (or 443 with an ACM certificate).

5. **Route 53** — create an A-record alias pointing your domain to the ALB DNS name.

### Why the ECS task role matters

When the Fargate container starts, boto3 automatically fetches temporary AWS
credentials from the ECS task metadata endpoint
(`http://169.254.170.2/...`).  These credentials belong to the **task role**
you specified in step 1.  No access keys are stored in the image or passed as
environment variables.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `AccessDeniedException` from Bedrock | Task role missing `bedrock:InvokeModel` | Add the IAM policy |
| `Could not connect to the endpoint URL` | Wrong or missing `AWS_REGION` | Set `AWS_REGION` env var |
| `ResourceNotFoundException` | Model not enabled in region | Enable Titan v2 in Bedrock Model Access |
| Container exits at startup | Dataset CSV missing | Ensure `subfeatures_dataset.csv` is in the image |
| `/health` returns 200 but `/search` errors | Bedrock index not built | Check container logs for embedding init errors |
| Slow first response after deploy | Embedding index is being generated | Wait — subsequent requests are fast |

---

## Local Quick-Start (copy-paste)

```bash
git clone <repo-url> && cd embedding-search-engine
pip install -r requirements.txt
export AWS_REGION=us-east-1   # or your region
python app.py
# in another terminal:
curl http://localhost:5000/health
curl "http://localhost:5000/search?description=extract+text+from+image"
```
