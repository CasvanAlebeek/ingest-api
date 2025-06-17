import os
import sys
from pathlib import Path
import types
from unittest.mock import MagicMock

sys.modules.setdefault("httpx", types.ModuleType("httpx"))
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
import main

client = TestClient(main.app)


def mock_embedding(*args, **kwargs):
    class Result:
        data = [MagicMock(embedding=[0.0, 0.1])]
    return Result()


def test_ingest_endpoint(monkeypatch):
    monkeypatch.setattr(main.openai.embeddings, "create", mock_embedding)
    mock_index = MagicMock()
    monkeypatch.setattr(main, "index", mock_index)

    payload = {
        "title": "Test",
        "problem": "p",
        "solution": "s",
        "machine": "m",
        "type": "t",
        "project": "prj"
    }

    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert mock_index.upsert.called


def test_query_endpoint(monkeypatch):
    monkeypatch.setattr(main.openai.embeddings, "create", mock_embedding)
    mock_match = MagicMock(score=0.9, metadata={"title": "t", "problem": "p", "solution": "s"})
    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[mock_match])
    monkeypatch.setattr(main, "index", mock_index)

    response = client.post("/query", json={"query": "hi"})
    assert response.status_code == 200
    data = response.json()
    assert "antwoord" in data
    assert data["score"] == 0.9
