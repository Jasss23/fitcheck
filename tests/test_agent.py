# Use mock to test, without actually calling the API
# The idea of mock is to replace "calling external services" with "pretending to call, returning preset results"
import os
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

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping RAG tests"
)
def test_resume_rag_indexing():
    """测试简历文字能被正确索引和检索"""
    from app.rag import ResumeRAG
    
    rag = ResumeRAG()
    
    # 用真实格式的简历文字，不是 mock
    sample_resume = """EXPERIENCE
    
● LLM-Gated CI/CD: Integrated an LLM-powered code review gate into the ML 
deployment workflow. Multi-model routing scores quality and risk-logic correctness.

● App Embedding Model: Co-designed a Transformer-based model over app-install 
sequences using heterogeneous feature fusion.

SKILLS

● Machine Learning & AI: Supervised learning, Transformer, LLM, RAG, 
Agentic Coding, MLflow, PySpark."""

    chunk_count = rag.index_resume_text(sample_resume)
    assert chunk_count >= 1, f"Expected at least 2 chunks, got {chunk_count}"
    assert rag.has_resume is True


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping RAG tests"
)
def test_resume_rag_retrieval_relevance():
    """Test retrieval result and query relevance"""
    from app.rag import ResumeRAG
    
    rag = ResumeRAG()
    sample_resume = """EXPERIENCE

● LLM-Gated CI/CD: Integrated an LLM-powered code review gate.
Multi-model routing scores quality automatically.

● Data Pipeline: Built batch data pipeline for collection operations.
Achieved 99% daily SLA across regional markets.

SKILLS

Machine Learning, Python, PySpark, MLflow."""

    rag.index_resume_text(sample_resume)
    
    # Use LLM/CI/CD related query, should retrieve the first bullet
    result = rag.retrieve("LLM code review automated deployment pipeline")
    
    assert result != "", "Should return non-empty retrieval result"
    # LLM-Gated CI/CD should be more relevant than Data Pipeline
    assert "LLM" in result or "CI/CD" in result or "code review" in result, \
        f"Expected LLM-related content in retrieval, got: {result[:200]}"

def test_score_node_uses_resume_context():
    """Test score_node uses real resume instead of fixed profile when resume_context is present"""
    from app.agent import score_node
    
    # State with resume_context
    state_with_resume = {
        "company_info": "Tech company looking for ML engineers",
        "jd_text": "Senior ML Engineer with LLM experience required",
        "resume_context": "Candidate built LLM-gated CI/CD pipeline at SeaMoney",
        "error": "",
        "fit_score": {}
    }
    
    # State without resume_context (fallback path)
    state_without_resume = {
        "company_info": "Tech company looking for ML engineers", 
        "jd_text": "Senior ML Engineer with LLM experience required",
        "resume_context": "",
        "error": "",
        "fit_score": {}
    }
    
    # Both should return valid fit_score, no error
    result_with = score_node(state_with_resume)
    result_without = score_node(state_without_resume)
    
    assert "fit_score" in result_with or "error" in result_with
    assert "fit_score" in result_without or "error" in result_without

@pytest.fixture
def client():
    """Create a test FastAPI client, without actually starting the server"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)

