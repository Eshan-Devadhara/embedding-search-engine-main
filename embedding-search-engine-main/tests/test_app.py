"""
tests/test_app.py

Basic smoke tests for the Bedrock Embedding Search Engine.

These tests mock the Bedrock client so they run locally without real AWS
credentials.  They validate routing, error handling, and response shapes.

Run with:
    python -m pytest tests/ -v
"""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1024   # Titan v2 produces 1024-dim vectors


def make_fake_bedrock_response(embedding=None):
    """Return a mock object that mimics a Bedrock invoke_model response."""
    emb = embedding if embedding is not None else FAKE_EMBEDDING
    body_bytes = json.dumps({"embedding": emb}).encode()
    mock_response = {
        "body": MagicMock(read=MagicMock(return_value=body_bytes))
    }
    return mock_response


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestHealth(unittest.TestCase):
    """The /health endpoint must return 200 without touching Bedrock."""

    def setUp(self):
        # Pre-populate the in-memory index so app.py does not call init_index
        import search
        search._index = [
            {
                "name": "test_feature",
                "description": "A test description",
                "embedding": FAKE_EMBEDDING,
            }
        ]
        from app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_json_body(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertEqual(data.get("status"), "healthy")


class TestSearch(unittest.TestCase):
    """The /search endpoint."""

    def setUp(self):
        import search
        search._index = [
            {
                "name": "moderation",
                "description": "Text moderation scans text for offensive content.",
                "embedding": FAKE_EMBEDDING,
            },
            {
                "name": "ocr",
                "description": "Extracts text from images using optical character recognition.",
                "embedding": [0.2] * 1024,
            },
        ]
        from app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    @patch("bedrock._get_client")
    def test_search_returns_results(self, mock_get_client):
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = make_fake_bedrock_response()
        mock_get_client.return_value = mock_bedrock

        resp = self.client.get("/search?description=extract text from image")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    @patch("bedrock._get_client")
    def test_search_result_fields(self, mock_get_client):
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = make_fake_bedrock_response()
        mock_get_client.return_value = mock_bedrock

        resp = self.client.get("/search?description=text moderation")
        data = json.loads(resp.data)
        first = data[0]
        self.assertIn("name", first)
        self.assertIn("score", first)
        self.assertIn("description", first)

    def test_missing_description_returns_400(self):
        resp = self.client.get("/search")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    def test_empty_description_returns_400(self):
        resp = self.client.get("/search?description=")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    def test_blank_description_returns_400(self):
        resp = self.client.get("/search?description=   ")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    @patch("bedrock._get_client")
    def test_bedrock_failure_returns_500(self, mock_get_client):
        from botocore.exceptions import ClientError
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "InvokeModel",
        )
        mock_get_client.return_value = mock_bedrock

        resp = self.client.get("/search?description=something")
        self.assertEqual(resp.status_code, 500)
        data = json.loads(resp.data)
        self.assertIn("error", data)
        # Must NOT leak AWS credential info
        self.assertNotIn("key", data["error"].lower())
        self.assertNotIn("secret", data["error"].lower())


class TestGenerateEmbedding(unittest.TestCase):
    """Unit tests for bedrock.generate_embedding()."""

    @patch("bedrock._get_client")
    def test_returns_list_of_floats(self, mock_get_client):
        from bedrock import generate_embedding
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = make_fake_bedrock_response()
        mock_get_client.return_value = mock_bedrock

        result = generate_embedding("Hello world")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1024)
        self.assertIsInstance(result[0], float)

    def test_empty_text_raises_valueerror(self):
        from bedrock import generate_embedding
        with self.assertRaises(ValueError):
            generate_embedding("")

    def test_whitespace_text_raises_valueerror(self):
        from bedrock import generate_embedding
        with self.assertRaises(ValueError):
            generate_embedding("   ")

    def test_none_raises_valueerror(self):
        from bedrock import generate_embedding
        with self.assertRaises(ValueError):
            generate_embedding(None)


if __name__ == "__main__":
    unittest.main()
