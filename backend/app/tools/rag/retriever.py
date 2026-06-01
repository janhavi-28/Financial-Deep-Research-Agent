import logging
from app.tools.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

def retrieve_financial_context(query: str, ticker: str = None, top_k: int = 5) -> tuple[str, list]:
    """
    Performs a semantic search against the localized ChromaDB to find the most
    relevant financial document chunks for a given query.
    Returns a tuple of (formatted_context_string, list_of_sources).
    """
    logger.info(f"Retrieving context for query: '{query}' (Ticker filter: {ticker})")
    
    vectorstore = get_vectorstore()
    
    # Configure retriever with metadata filtering if ticker is provided
    search_kwargs = {"k": top_k}
    if ticker:
        search_kwargs["filter"] = {"ticker": ticker.upper()}
        
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    try:
        results = retriever.invoke(query)
    except Exception as e:
        logger.warning(f"ChromaDB retrieval failed (is the DB empty?): {e}")
        return "No local RAG context available.", []
    
    if not results:
        return "No relevant local financial documents found.", []

    # Format the extracted context with citations
    context_blocks = []
    sources_list = []
    for i, doc in enumerate(results):
        source = doc.metadata.get("source_file", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        context_blocks.append(f"[Citation {i+1} - {source} Pg. {page}]:\n{doc.page_content}")
        sources_list.append(f"Internal Document: {source} (Page {page})")
        
    final_context = "\n\n".join(context_blocks)
    logger.info(f"Successfully retrieved {len(results)} relevant chunks.")
    return final_context, sources_list
