# app/tools.py
import os
from tavily import TavilyClient
from langchain_core.tools import tool

# @tool is a decorator from LangChain.
# Purpose: It turns a regular Python function into a tool that the LLM can call.
# The LLM reads the function's docstring to understand what this tool does and when to use it.
# Therefore, write the docstring clearly—it's for the LLM, not for humans.

@tool
def search_company(company_name: str) -> str:
    """Search for information about a company including their tech stack,
    culture, recent news, and funding status. Use this to understand
    the company before evaluating job fit."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    
    # Use two more precise queries separately instead of one broad query.
    # Reason: A single query trying to cover all dimensions often fails to capture any of them accurately.
    
    tech_results = client.search(
        query=f"{company_name} engineering tech stack machine learning infrastructure",
        max_results=2
    )
    culture_results = client.search(
        query=f"{company_name} company culture data science team hiring 2024 2025",
        max_results=2
    )
    
    # Include source information to help the LLM assess the credibility of the information.
    def format_results(results):
        formatted = []
        for r in results["results"]:
            formatted.append(
                f"Source: {r['url']}\n"
                f"Title: {r['title']}\n"  
                f"Content: {r['content']}\n"
            )
        return "\n---\n".join(formatted)
    
    return (
        "=== Tech Stack & Engineering ===\n"
        + format_results(tech_results)
        + "\n\n=== Culture & Team ===\n"
        + format_results(culture_results)
    )

@tool  
def analyze_jd(jd_text: str) -> str:
    """Extract and structure key requirements from a job description.
    Returns required skills, nice-to-have skills, and hidden signals 
    about team culture and expectations."""
    # 这个工具不需要外部 API——直接让 LLM 分析文本
    # 注意：这里我们返回原始 JD，让 agent 的 LLM 自己做分析
    # 而不是在工具里再调一次 LLM（避免嵌套调用，保持工具职责单一）
    return f"JD Content for analysis:\n{jd_text}"

@tool
def calculate_fit_score(
    technical_match: int, 
    domain_match: int, 
    experience_match: int
) -> dict:
    """Calculate overall fit score based on component scores (0-10 each).
    Returns overall score and recommendation level.
    Use this ONLY after you have analyzed both the JD and company info."""
    overall = round((technical_match * 0.6 + domain_match * 0.2 + experience_match * 0.2) * 10)
    
    # Why do we calculate this in the tool rather than let the LLM do it?
    # Because the score calculation is a deterministic business rule, and should not let the LLM "think it's about the same"
    if overall >= 80:
        recommendation = "High Fit"
    elif overall >= 65:
        recommendation = "Moderate Fit"  
    else:
        recommendation = "Low Fit"
        
    # Return the overall score, recommendation, and breakdown of the scores
    return {
        "overall_score": overall,
        "recommendation": recommendation,
        "breakdown": {
            "technical": technical_match,
            "domain": domain_match, 
            "experience": experience_match
        }

    }