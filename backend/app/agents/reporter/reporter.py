import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.state.research_state import ResearchState
from app.agents.shared.llm import gemini_client

logger = logging.getLogger(__name__)

async def reporter_node(state: ResearchState):
    """
    Synthesizes the final markdown report from all gathered research with strict token limit.
    """
    logger.info("Generating final investor-grade report...")

    # Compile the raw context
    findings_summary = "\n".join(state.get("findings", []))
    history = state.get("search_history", [])
    query = state.get("query", "")
    
    # Extract unique URLs from search history to pass as context
    unique_urls = list(set([item.get('url') for item in history if isinstance(item, dict) and 'url' in item]))
    sources_context = "\n".join([f"- {url}" for url in unique_urls])

    sys_msg = SystemMessage(content="You are a Wall Street financial analyst writing a concise investor report.")
    
    prompt = f"""
Write a structured report with EXACTLY these 6 sections.
Keep each section to 3-5 bullet points. Be concise and use real numbers.

## 1. Executive Summary
## 2. Financial Metrics
(Use exact numbers from the data. Write "Data not available" if missing. Do NOT guess.)
## 3. Competitive Position
## 4. Key Risks
## 5. Opportunities & Outlook
## 6. Sources
(List all Sources/URLs from the research data)

Total report must be under 800 words.
Do not add any sections beyond these 6.
Do not hallucinate any numbers.

User query: "{query}"

Research summary:
{findings_summary}

Available Sources Context:
{sources_context}
"""

    messages = [
        sys_msg,
        HumanMessage(content=prompt)
    ]

    try:
        response = await gemini_client.ainvoke(messages)
        final_report = response.content
        if isinstance(final_report, list):
            final_report = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in final_report])
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        final_report = f"Failed to generate final report: {e}"

    logger.info("Report generation complete.")
    
    # Update state
    return {
        "final_report": final_report,
        "current_research_step": "generation_complete"
    }
