from dotenv import load_dotenv
load_dotenv()  # First thing to do

from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
import re

from app.tools import search_company, analyze_jd, calculate_fit_score

# ─────────────────────────────────────────
# STEP 1: Define the State
# ─────────────────────────────────────────
# State is the "shared memory" of the agent.
# TypedDict allows us to declare the type of each field, so LangGraph knows what the State looks like.
# The messages field is special: Annotated[list, operator.add] means "when multiple nodes write to the messages field, merge them (add) instead of overwriting".

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    company_name: str
    jd_text: str
    company_info: str       # The result of search_company is stored here
    jd_analysis: str        # The result of analyze_jd is stored here  
    fit_score: dict         # The result of calculate_fit_score is stored here
    error: str              # If any step fails, the error information is stored here
    resume_context: str     # New: retrieved resume chunks, empty string if none

# ─────────────────────────────────────────
# STEP 2: Define the node functions
# ─────────────────────────────────────────
# Each node function: receives state → does one thing → returns an update to the state
# Note: The return is an "update", not a complete new state

# Initialize the LLM
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

def search_node(state: AgentState) -> dict:
    """Search for company information"""
    try:
        result = search_company.invoke({"company_name": state["company_name"]})
        return {"company_info": result}
    except Exception as e:
        # Error handling: Any external API failure is captured
        # The exception is not propagated to the agent layer, but the error information is stored in the error field
        return {"error": f"Company search failed: {str(e)}", "company_info": ""}

def analyze_jd_node(state: AgentState) -> dict:
    """Analyze the JD"""
    # If the previous step failed, skip this step - the error should be propagated
    if state.get("error"):
        return {}
    try:
        result = analyze_jd.invoke({"jd_text": state["jd_text"]})
        return {"jd_analysis": result}
    except Exception as e:
        return {"error": f"JD analysis failed: {str(e)}", "jd_analysis": ""}

def score_node(state: AgentState) -> dict:
    """Score the candidate fit based on the results of the previous two steps"""
    if state.get("error"):
        return {}
    
    # We don't let the LLM go wild, but give it a structured prompt
    # It must output the scores in JSON format
    # Then our code parses the JSON and calls calculate_fit_score
    if state.get("resume_context"):
        candidate_profile = f"""Based on the candidate's actual resume (most relevant sections retrieved):

                            {state["resume_context"]}

                            Use the above as the ground truth for the candidate's background."""
    else:
        # Fallback: original fixed profile, keep the original experience for users who have not uploaded resumes
        candidate_profile = """Senior ML Engineer with 4 years at SeaMoney/Monee.
                            Built Transformer-based heterogeneous feature fusion models.
                            Deployed LLM-gated CI/CD pipelines. 
                            Experience with LangGraph, MLflow, PySpark, Airflow.
                            MS in CS (UPenn) + MS in Quantitative Economics (UW-Madison).
                            Singapore PR."""
    scoring_prompt = f"""Based on the following information, score the candidate fit:

COMPANY INFO (summary):
{state["company_info"][:800]}

JOB DESCRIPTION (key parts):
{state["jd_text"][:1500]}

CANDIDATE PROFILE:
{candidate_profile}

Return ONLY a JSON object. No markdown, no code blocks, no explanation.
Start with {{ and end with }}.

{{
    "technical_match": <int 0-10>,
    "domain_match": <int 0-10>,
    "experience_match": <int 0-10>,
    "reasoning": "<one sentence explaining the scores>"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=scoring_prompt)])
        content = response.content
        try:    
            scores = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                return {"error": f"Invalid JSON. Got: {content[:200]}"}
            scores = json.loads(json_match.group())

        fit_result = calculate_fit_score.invoke({
                "technical_match": scores["technical_match"],
                "domain_match": scores["domain_match"],
                "experience_match": scores["experience_match"]
            })
        
        # Now use deterministic code to calculate the final score, instead of letting the LLM do it
        fit_result = calculate_fit_score.invoke({
            "technical_match": scores["technical_match"],
            "domain_match": scores["domain_match"],
            "experience_match": scores["experience_match"]
        })
        
        # Add the LLM's reasoning to the result
        fit_result["reasoning"] = scores.get("reasoning", "")
        return {"fit_score": fit_result}
        
    except Exception as e:
        return {"error": f"Scoring failed: {str(e)}"}

def report_node(state: AgentState) -> dict:
    """Generate the final report"""
    # If there is an error, generate an error report instead of an analysis report
    if state.get("error"):
        error_msg = state["error"]
        return {"messages": [{"role": "assistant", 
                              "content": f"Analysis failed: {error_msg}"}]}
    
    report_prompt = f"""You are FitCheck, an AI job fit analyzer. Generate a concise, 
insightful fit analysis report based on:

Company: {state["company_name"]}
Fit Score: {state["fit_score"]}
Company Info Summary: {state["company_info"][:500]}...

Write a 3-paragraph report:
1. Overall fit assessment with the score and recommendation
2. Key strengths that match this role  
3. Potential gaps or areas to address in the application

Be direct and actionable. This is for a senior ML/AI engineer applying to tech companies."""

    response = llm.invoke([HumanMessage(content=report_prompt)])
    return {"messages": [{"role": "assistant", "content": response.content}]}

# ─────────────────────────────────────────
# STEP 3: Build the graph
# ─────────────────────────────────────────

def build_agent():
    # StateGraph accepts the type definition of State
    # It validates the input and output of each node based on this type
    graph = StateGraph(AgentState)
    
    # Add nodes: name → function
    graph.add_node("search", search_node)
    graph.add_node("analyze_jd", analyze_jd_node)
    graph.add_node("score", score_node)
    graph.add_node("report", report_node)
    
    # Add fixed edges: this agent is linear, no conditional branches
    # search → analyze_jd → score → report → END
    # 
    # Why don't we need conditional branches here?
    # Because error handling is propagated through the state["error"] field
    # Each node checks the error field, and if it fails, it skips its own logic
    # report_node handles both error and non-error cases
    graph.set_entry_point("search")
    graph.add_edge("search", "analyze_jd")
    graph.add_edge("analyze_jd", "score")
    graph.add_edge("score", "report")
    graph.add_edge("report", END)
    
    # recursion_limit prevents the agent from looping infinitely (not very meaningful for this linear agent,
    # but a good habit, very important for agents with loops)

    return graph.compile()

# ─────────────────────────────────────────
# STEP 4: Entry function (for FastAPI and testing)
# ─────────────────────────────────────────

def run_analysis(company_name: str, jd_text: str, resume_context: str = "") -> dict:
    """Run the complete fit analysis and return the structured result"""
    agent = build_agent()
    
    initial_state = {
        "messages": [],
        "company_name": company_name,
        "jd_text": jd_text,
        "company_info": "",
        "jd_analysis": "",
        "fit_score": {},
        "error": "",
        "resume_context": resume_context
    }

    for step in agent.stream(initial_state):
        print("=== STEP ===")
        for node_name, node_output in step.items():
            print(f"Node: {node_name}")
            print(f"Output: {node_output}")
        print("\n")
    
    result = agent.invoke(initial_state, config={"recursion_limit": 10})
    
    return {
        "company": company_name,
        "fit_score": result.get("fit_score", {}),
        "report": result["messages"][-1]["content"] if result["messages"] else "",
        "error": result.get("error", "")
    }