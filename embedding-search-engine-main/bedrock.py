"""
bedrock.py - Amazon Bedrock embedding client.

Generates text embeddings using Amazon Titan Text Embeddings V2 via
the Bedrock Runtime API.  Authentication is handled entirely by boto3's
standard credential provider chain (ECS task role, EC2 instance profile,
environment variables, ~/.aws/credentials) - no API keys are required or
accepted.
"""

import json
import logging
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BEDROCK_MODEL_ID: str = os.environ.get(
    "BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

AWS_REGION: str | None = os.environ.get("AWS_REGION")

# Titan v2 supports 256 / 512 / 1024 dimensions (default 1024).
EMBEDDING_DIMENSIONS: int = 1024

# ---------------------------------------------------------------------------
# Lazy client
# ---------------------------------------------------------------------------
_bedrock_client = None


def _get_client():
    """Return a cached Bedrock Runtime boto3 client."""
    global _bedrock_client
    if _bedrock_client is None:
        kwargs = {"service_name": "bedrock-runtime"}
        if AWS_REGION:
            kwargs["region_name"] = AWS_REGION
        _bedrock_client = boto3.client(**kwargs)
        logger.info(
            "Bedrock Runtime client initialised (model=%s, region=%s)",
            BEDROCK_MODEL_ID,
            AWS_REGION or "<boto3 auto-detect>",
        )
    return _bedrock_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embedding(text: str) -> list:
    """
    Generate a text embedding using Amazon Bedrock Titan Text Embeddings V2.

    Parameters
    ----------
    text : str
        The input text to embed. Must be non-empty.

    Returns
    -------
    list of float
        A list of floats representing the embedding vector
        (1024 dimensions for Titan v2).

    Raises
    ------
    ValueError
        If text is empty or None.
    RuntimeError
        If the Bedrock API call fails for any reason.
    """
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    # Titan v2 request body format:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
    request_body = json.dumps(
        {
            "inputText": text.strip(),
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True,
        }
    )

    try:
        client = _get_client()
        response = client.invoke_model(
            body=request_body,
            modelId=BEDROCK_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        raise RuntimeError(
            f"Bedrock API error [{error_code}]: check IAM permissions and model availability."
        ) from exc
    except BotoCoreError as exc:
        raise RuntimeError(
            f"Bedrock connection error: {type(exc).__name__}. "
            "Check AWS_REGION and network connectivity."
        ) from exc

    try:
        response_body = json.loads(response["body"].read())
        embedding = response_body["embedding"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unexpected Bedrock response format: {type(exc).__name__}"
        ) from exc

    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Bedrock returned an empty or malformed embedding.")

    return embedding
