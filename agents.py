# agents.py - Two-Agent Orchestration System

from rag import (
    fetch_wikipedia, 
    split_into_chunks, 
    build_vectorstore_from_docs, 
    generate_article_from_retriever
)

def research_agent(topic: str, max_pages: int = 3) -> str:
    """
    Research Agent: Gathers information from Wikipedia
    
    Responsibility: Information retrieval and aggregation
    """
    print(f"🔍 Research Agent: Searching for '{topic}'...")
    text = fetch_wikipedia(topic, max_pages)
    
    if text:
        word_count = len(text.split())
        print(f"✅ Research Agent: Found {word_count} words from {max_pages} sources")
    else:
        print("⚠️ Research Agent: No content found")
    
    return text

def writer_agent(context_text: str, topic: str) -> str:
    """
    Writer Agent: Transforms research into newspaper article
    
    Responsibility: Content generation and formatting
    """
    print(f"✍️ Writer Agent: Generating article about '{topic}'...")
    
    # Prepare context
    docs = split_into_chunks(context_text)
    print(f"📄 Writer Agent: Processing {len(docs)} context chunks")
    
    # Build retriever
    retriever = build_vectorstore_from_docs(docs).as_retriever(
        search_kwargs={"k": 4}
    )
    
    # Generate article
    article = generate_article_from_retriever(retriever, topic)
    print("✅ Writer Agent: Article generated successfully")
    
    return article

def two_agent_pipeline(topic: str, max_pages: int = 3) -> str:
    """
    Orchestrates two-agent workflow
    
    Flow:
    1. Research Agent gathers information
    2. Writer Agent produces article
    3. Return final article
    """
    print("\n" + "="*50)
    print("🤖 TWO-AGENT SYSTEM ACTIVATED")
    print("="*50 + "\n")
    
    # Step 1: Research
    context = research_agent(topic, max_pages)
    
    if not context:
        return "Error: Research Agent could not find relevant information."
    
    # Step 2: Write
    article = writer_agent(context, topic)
    
    print("\n" + "="*50)
    print("✅ TWO-AGENT PIPELINE COMPLETE")
    print("="*50 + "\n")
    
    return article

def multi_agent_orchestration(topic: str, angle: str = None) -> dict:
    """
    Advanced: Multiple agents with specialized roles
    
    Agents:
    - Research Agent: Gathers data
    - Analysis Agent: Extracts key insights
    - Writer Agent: Creates article
    - Editor Agent: Reviews and refines
    """
    # TODO: Implement full multi-agent system
    # This is a placeholder for future enhancement
    pass