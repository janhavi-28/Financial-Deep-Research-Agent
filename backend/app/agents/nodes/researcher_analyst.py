import asyncio
import logging
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from app.state.research_state import ResearchState
from app.tools.search.tavily_search import tavily_search_tool
from app.agents.shared.llm import gemini_client
from app.tools.rag.retriever import retrieve_financial_context

logger = logging.getLogger(__name__)

async def researcher_analyst_node(state: ResearchState) -> Dict[str, Any]:
    query = state.get("query", "")
    ticker = state.get("extracted_ticker", "NONE")
    next_queries = state.get("next_search_queries", [])
    financial_data = state.get("financial_data", {})
    search_history = state.get("search_history", [])
    
    logger.info("--- RESEARCHER + ANALYST NODE ---")
    
    # 1. RAG Retrieval
    rag_context, rag_sources = retrieve_financial_context(query=query, ticker=ticker)
    if not rag_context or "No local" in rag_context or "No relevant" in rag_context:
        rag_context = "None uploaded"
    else:
        for src in rag_sources:
            search_history.append({"query": "Internal PDF", "url": src})
        
    # 2. Web Search Execution
    raw_search_data = ""
    try:
        if next_queries:
            search_tasks = [tavily_search_tool.search(query=q, max_results=3) for q in next_queries]
            results_matrix = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for idx, result_list in enumerate(results_matrix):
                if isinstance(result_list, Exception):
                    continue
                for r in result_list:
                    search_history.append({"query": next_queries[idx], "url": r.url})
                    raw_search_data += f"\nSource: {r.url}\nContent: {r.content}\n"
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raw_search_data = "Web search failed."

    # 3. LLM Synthesis
    finance_str = json.dumps(financial_data, indent=2) if financial_data else "No ticker data available"
    
    prompt = f"""
You are a senior financial analyst.

User question: "{query}"

INTERNAL DOCUMENTS (from uploaded PDFs):
{rag_context}

WEB SEARCH RESULTS:
{raw_search_data[:12000]}

LIVE FINANCIAL DATA (Yahoo Finance):
{finance_str}

Your task:
- Ignore ads, irrelevant content, and repeated information
- Extract key financial facts, figures, and insights
- Cross-check web narrative against hard financial numbers
- Flag any discrepancies between what the web says and the numbers
- Summarize findings in clean bullet points under exactly these 4 headings:

### Market & Business Overview
### Financial Performance  
### Competitive Position
### Key Risks & Opportunities

Keep total output under 600 words. Use real numbers where available.
"""
    
    logger.info("Analyzing findings via Gemini...")
    try:
        response = await gemini_client.ainvoke([HumanMessage(content=prompt)])
        synthesis = response.content
        if isinstance(synthesis, list):
            synthesis = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in synthesis])
    except Exception as e:
        logger.error(f"Analyst failed: {e}")
        synthesis = f"Failed to generate analysis: {e}"

    return {
        "search_history": search_history,
        "findings": [synthesis],
        "current_research_step": "analysis_completed"
    }
