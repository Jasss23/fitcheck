from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
import os

load_dotenv()

from app.agent import run_analysis

# ─────────────────────────────────────────
# STEP 1: Create the FastAPI application instance
# title and description will appear in the automatically generated API documentation
# FastAPI automatically generates documentation: after starting, visit /docs to see all endpoints
# ─────────────────────────────────────────
app = FastAPI(
    title="FitCheck API",
    description="AI-powered job fit analyzer for ML/AI engineering roles",
    version="0.1.0"
)

# ─────────────────────────────────────────
# STEP 2: CORS middleware
# CORS = Cross-Origin Resource Sharing
##WARNING: During development, use allow_origins=["*"] to allow all origins, in production, tighten it
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change it to actual domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# STEP 3: Request/response data model
# Pydantic BaseModel does two things:
# 1. Define the structure and type of the data
# 2. Automatically validate: if a field is missing or the type is wrong, FastAPI returns a 422 error
# This is "API-level error handling" - bad data can't even get into your business logic
# ─────────────────────────────────────────
class AnalysisRequest(BaseModel):
    company_name: str
    jd_text: str
    
    # validator allows you to add custom validation rules
    # here we check that the input cannot be an empty string
    @validator('company_name')
    def company_name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Company name cannot be empty')
        return v.strip()
    
    @validator('jd_text')
    def jd_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('JD text cannot be empty')
        # JD is too long will cause token overflow, truncate at the input boundary
        if len(v) > 8000:
            return v[:8000]
        return v.strip()

class AnalysisResponse(BaseModel):
    company: str
    fit_score: dict
    report: str
    error: str

# ─────────────────────────────────────────
# Endpoint definition
# ─────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Health check endpoint, the deployment platform (Render/Cloud Run) will periodically ping this interface
    to check if the service is running normally. Returning 200 means "I'm still alive".
    This is a must-have for production services, without it the deployment platform doesn't know your service is down.
    """
    return {"status": "healthy", "version": "0.1.0"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Core endpoint: receive company name and JD, return fit analysis.
    
    async def makes this function asynchronous - when the agent is waiting for the LLM, FastAPI can handle other requests.
    """
    # Check if the API key exists
    # This is the "fail fast" principle: instead of letting the request run for 10 seconds and then telling the user the key is wrong, check it at the beginning and return a meaningful error immediately
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not configured"
        )
    if not os.environ.get("TAVILY_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="TAVILY_API_KEY not configured"
        )
    
    try:
        # run_in_threadpool puts the synchronous function in the thread pool so that the main thread (event loop) is not blocked, and other requests can be processed.
        result = await run_in_threadpool(
            run_analysis,
            request.company_name,
            request.jd_text
        )
        return AnalysisResponse(**result)
        
    except Exception as e:
        # Capture all unexpected errors
        # Note: We do not return the original error message directly to the user
        # Reason: The error message may contain internal implementation details (API key format, file paths, etc.)
        # This is a security practice: only say "internal error" to the user, detailed logs are kept on the server side
        print(f"Unexpected error in /analyze: {str(e)}")  # Server-side logging
        raise HTTPException(
            status_code=500,
            detail="Analysis failed due to an internal error. Please try again."
        )

# ─────────────────────────────────────────
# STEP 4: Static file service
# Serve the frontend HTML file directly with FastAPI
# This eliminates the need for a separate frontend server
# ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """
    When accessing the root path, return the frontend HTML file.
    The frontend and backend are in the same service, so only one server is needed for deployment.
    """
    return FileResponse("static/index.html")