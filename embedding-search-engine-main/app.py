"""
app.py - Flask entry point for the Bedrock Embedding Search Engine.

Environment variables
---------------------
PORT                 Flask listen port (default: 5000)
AWS_REGION           AWS region for Bedrock (boto3 auto-detects if unset)
BEDROCK_MODEL_ID     Bedrock model ID (default: amazon.titan-embed-text-v2:0)

Run locally
-----------
    python app.py

Rebuild the embedding index (force re-embed all rows):
    python app.py --rebuild-index
"""

import argparse
import logging
import os
import sys

from flask import Flask, jsonify, request

from bedrock import BEDROCK_MODEL_ID, AWS_REGION
from search import init_index, search_subfeature

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint used by the ALB target group.
    Must return HTTP 200 quickly - never calls Bedrock.
    """
    return jsonify({"status": "healthy"}), 200


@app.route("/search", methods=["GET"])
def search_endpoint():
    """
    Semantic search endpoint.

    Query parameters
    ----------------
    description : str  (required)
        Natural-language search query.

    Returns
    -------
    JSON list of top-5 results, each with ``name``, ``score``,
    and ``description``.
    """
    description = request.args.get("description", "").strip()

    if not description:
        return jsonify({"error": "description query parameter is required"}), 400

    try:
        results = search_subfeature(description)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        # Log full detail server-side; return a safe message to the client
        logger.error("Search failed: %s", exc)
        return jsonify({"error": "Search failed. Check server logs for details."}), 500
    except Exception as exc:
        logger.exception("Unexpected error during search: %s", exc)
        return jsonify({"error": "An unexpected error occurred."}), 500

    return jsonify(results), 200


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup(rebuild: bool = False) -> None:
    """Initialise the embedding index before the first request."""
    logger.info("=== Bedrock Embedding Search Engine starting ===")
    logger.info("Bedrock model ID : %s", BEDROCK_MODEL_ID)
    logger.info(
        "AWS region       : %s",
        AWS_REGION if AWS_REGION else "<boto3 auto-detect (ECS task region)>",
    )

    try:
        init_index(model_id=BEDROCK_MODEL_ID, force_rebuild=rebuild)
    except Exception as exc:
        logger.error("FATAL: Could not initialise embedding index: %s", exc)
        sys.exit(1)

    logger.info("Embedding index ready. Flask starting...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bedrock Embedding Search Engine")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete the local embedding cache and regenerate all embeddings from Bedrock.",
    )
    args = parser.parse_args()

    startup(rebuild=args.rebuild_index)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
