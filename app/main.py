from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from app.rag import ResumeRAG
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
import os
from fastapi.responses import StreamingResponse
import json
import asyncio
import random
import time

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

limiter = Limiter(key_func=get_remote_address)


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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    session_id: str = ""  # optional, pass in when there is a resume, otherwise use the original fixed profile
    
    # validator allows you to add custom validation rules
    # here we check that the input cannot be an empty string
    @field_validator('company_name')
    @classmethod
    def company_name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Company name cannot be empty')
        return v.strip()
    
    @field_validator('jd_text')
    @classmethod
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
# Streaming endpoint
# SSE = Server-Sent Events
# The principle is that the HTTP connection is kept open, and the server can continuously push data to the client
# The format is fixed: each message starts with "data: " and ends with two newlines
# The frontend uses the EventSource API to receive, without WebSocket
# ─────────────────────────────────────────
@app.post("/analyze/stream")
@limiter.limit("5/hour")
async def analyze_stream(request: Request, body: AnalysisRequest):
    """
    Streaming version of /analyze.
    Returns Server-Sent Events so the frontend can show
    real-time progress as each agent node completes.
    """
    async def event_generator():
        # Use LangGraph's stream mode, yield once when each node completes
        # So the frontend can see the real agent execution progress
        resume_context = ""
        if body.session_id and body.session_id in resume_sessions:
            rag = resume_sessions[body.session_id]
            # Use JD text as query, retrieve the most relevant chunks in the resume
            # This is the core of RAG: use "question" to find "answer"
            resume_context = rag.retrieve(body.jd_text)



        
        from app.agent import build_agent
        
        agent = build_agent()
        initial_state = {
            "messages": [],
            "company_name": body.company_name,
            "jd_text": body.jd_text,
            "company_info": "",
            "jd_analysis": "",
            "fit_score": {},
            "error": "",    
            "resume_context": resume_context
        }
        
        # node name to frontend display text mapping
        # This is "separation of concerns": the backend should not know what text the frontend will display
        # But in this small project, it is a reasonable tradeoff to put it here
        step_labels = {
            "search": "Searching company intelligence...",
            "analyze_jd": "Parsing job requirements...",
            "score": "Calculating fit score...",
            "report": "Generating your report..."
        }

        # UX: the score node waits on the LLM for a long time. Emit synthetic
        # dimension steps every 2–3s until the real score step completes.
        score_dim_steps = [
            ("score_dim_technical", "Scoring Technical dimension..."),
            ("score_dim_domain", "Scoring Domain dimension..."),
            ("score_dim_experience", "Scoring Experience dimension..."),
        ]

        try:
            # agent.stream() yields once when each node completes
            # We use run_in_threadpool to wrap it, because stream() is synchronous
            def run_stream():
                return list(agent.stream(initial_state))
            
            # Run stream in the thread pool, yield SSE messages while running
            import threading
            results = []
            done = threading.Event()
            
            def worker():
                for step in agent.stream(initial_state):
                    results.append(step)
                done.set()
            
            thread = threading.Thread(target=worker)
            thread.start()

            sent = 0
            awaiting_score = False
            score_synth_i = 0
            next_score_synth_at = None

            while not done.is_set() or sent < len(results):
                if sent < len(results):
                    step = results[sent]
                    sent += 1

                    for node_name, node_output in step.items():
                        label = step_labels.get(node_name, f"Processing {node_name}...")

                        if node_name == "analyze_jd":
                            awaiting_score = True
                            score_synth_i = 0
                            next_score_synth_at = None
                        elif node_name == "score":
                            awaiting_score = False

                        # SSE 格式：data: {json}\n\n
                        event_data = {
                            "type": "step",
                            "node": node_name,
                            "label": label,
                            "output": node_output
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                else:
                    if awaiting_score:
                        now = time.monotonic()
                        if next_score_synth_at is None:
                            next_score_synth_at = now + random.uniform(2.0, 3.0)
                        elif now >= next_score_synth_at:
                            syn_node, syn_label = score_dim_steps[
                                score_synth_i % len(score_dim_steps)
                            ]
                            score_synth_i += 1
                            next_score_synth_at = now + random.uniform(2.0, 3.0)
                            synth_event = {
                                "type": "step",
                                "node": syn_node,
                                "label": syn_label,
                                "output": {},
                            }
                            yield f"data: {json.dumps(synth_event)}\n\n"
                    await asyncio.sleep(0.05)
            
            thread.join()
            
            # After all nodes are completed, send the final result
            final_result = {}
            for step in results:
                for node_name, node_output in step.items():
                    if node_output:
                        final_result.update(node_output)
            
            final_event = {
                "type": "complete",
                "result": {
                    "company": body.company_name,
                    "fit_score": final_result.get("fit_score", {}),
                    "report": final_result.get("messages", [{}])[-1].get("content", "") if final_result.get("messages") else "",
                    "error": final_result.get("error", "")
                }
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Tell nginx not to buffer, push directly
        }
    )

# ─────────────────────────────────────────
# Feedback collection endpoint
# ─────────────────────────────────────────
from datetime import datetime

class FeedbackRequest(BaseModel):
    company_name: str
    question: str

@app.post("/feedback")
async def collect_feedback(request: FeedbackRequest):
    """
    Collect questions from HR/hiring managers viewing the portfolio.
    Stored locally in feedback.jsonl (one JSON per line).
    JSONL format is append-friendly — no need to read the whole file to add a record.
    """
    try:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "company": request.company_name,
            "question": request.question
        }
        # JSONL: one JSON per line, append-friendly - no need to read the whole file to add a record
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"status": "ok", "message": "Thanks for your question!"}
    except Exception as e:
        # Feedback collection failure should not affect the user experience, so here we handle it silently
        print(f"Feedback collection failed: {e}")
        return {"status": "ok", "message": "Thanks!"}



# Use a simple dict to simulate session storage
# key is session_id (UUID generated by frontend), value is ResumeRAG instance
# In production, Redis will be used, here a memory dict is enough
resume_sessions: dict[str, ResumeRAG] = {}

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str = ""
):
    """
    Receive resume PDF, build vector index
    Return session_id for subsequent analysis call
    
    Why use session_id instead of global state?
    Because multiple users may upload resumes at the same time
    Each user's resume needs to be isolated, cannot be mixed together
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # File size limit: 5MB, resume should not be larger
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")
    
    # If there is no session_id, generate a new one
    import uuid
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Create the RAG instance for this session
    rag = ResumeRAG(session_id)
    chunk_count = rag.index_resume(contents)
    resume_sessions[session_id] = rag
    
    return {
        "session_id": session_id,
        "chunks_indexed": chunk_count,
        "message": f"Resume indexed successfully ({chunk_count} sections)"
    }

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