# Use mock to test, without actually calling the API
# The idea of mock is to replace "calling external services" with "pretending to call, returning preset results"

import pytest
from unittest.mock import patch, MagicMock

def test_calculate_fit_score_high():
    """Test high score scenario: 8+8+8 should be High Fit"""
    from app.tools import calculate_fit_score
    result = calculate_fit_score.invoke({
        "technical_match": 8,
        "domain_match": 8,
        "experience_match": 8
    })
    assert result["overall_score"] >= 80
    assert result["recommendation"] == "High Fit"

def test_calculate_fit_score_low():
    """Test low score scenario"""
    from app.tools import calculate_fit_score
    result = calculate_fit_score.invoke({
        "technical_match": 3,
        "domain_match": 3,
        "experience_match": 3
    })
    assert result["recommendation"] == "Low Fit"

def test_calculate_fit_score_boundary():
    """Test boundary value: calculate expected score based on actual weights"""
    from app.tools import calculate_fit_score
    # (8×0.6 + 5×0.2 + 5×0.2) × 10 = 68 → Moderate Fit
    result = calculate_fit_score.invoke({
        "technical_match": 8,
        "domain_match": 5,
        "experience_match": 5
    })
    assert result["overall_score"] == 68
    assert result["recommendation"] == "Moderate Fit"

def test_api_health(client):
    """Test /health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_analyze_empty_input(client):
    """Test empty input is correctly rejected"""
    response = client.post("/analyze", json={
        "company_name": "",
        "jd_text": "some text"
    })
    # Pydantic validator should return 422 Unprocessable Entity
    assert response.status_code == 422

@pytest.fixture
def client():
    """Create a test FastAPI client, without actually starting the server"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)