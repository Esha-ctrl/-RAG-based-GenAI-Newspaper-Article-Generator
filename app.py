# app.py - Enhanced Streamlit RAG Newspaper Generator

import streamlit as st
from rag import (
    load_pdf_text, split_into_chunks, fetch_wikipedia,
    build_vectorstore_from_docs, generate_article_from_retriever
)
from agents import two_agent_pipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GenAI Newspaper Generator", 
    page_icon="📰",
    layout="wide"
)

# Custom CSS for newspaper styling
st.markdown("""
<style>
    .newspaper-header {
        text-align: center;
        border-bottom: 4px double #333;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    .newspaper-title {
        font-size: 48px;
        font-weight: bold;
        font-family: 'Georgia', serif;
        color: #1a1a1a;
    }
    .article-content {
        background: white;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Georgia', serif;
        line-height: 1.8;
    }
    .article-headline {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1a1a1a;
    }
    .article-byline {
        font-style: italic;
        color: #666;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="newspaper-header">', unsafe_allow_html=True)
st.markdown('<div class="newspaper-title">📰 THE DAILY AI</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-style: italic; color: #666;">"All the News That\'s Fit to Generate"</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.title("GenAI Newspaper Article Generator")
st.markdown("**Powered by RAG, LangChain, OpenAI & Vector Store Technology**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in .env file!")
        st.info("Please add your OpenAI API key to the .env file")
    else:
        st.success("✅ OpenAI API Key configured")
    
    st.markdown("---")
    
    # Agent mode toggle
    use_agents = st.checkbox(
        "Use Two-Agent System",
        value=False,
        help="Research Agent + Writer Agent orchestration"
    )
    
    st.markdown("---")
    st.markdown("### 📚 How it works:")
    if use_agents:
        st.markdown("""
        **Agent Mode:**
        1. 🔍 **Research Agent**: Gathers information
        2. ✍️ **Writer Agent**: Generates article
        3. 🤝 **Orchestration**: Coordinates workflow
        """)
    else:
        st.markdown("""
        **RAG Mode:**
        1. **Retrieval**: Fetches relevant content
        2. **Vectorization**: Creates embeddings (FAISS)
        3. **Generation**: LLM writes article with context
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Tech Stack:")
    st.markdown("""
    - **Streamlit**: UI Framework
    - **LangChain**: RAG Pipeline
    - **FAISS**: Vector Store
    - **OpenAI**: GPT-4 & Embeddings
    - **Docker**: Containerization
    """)

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    mode = st.radio(
        "Choose data source:",
        ("Wikipedia search", "Upload documents"),
        horizontal=True
    )

with col2:
    if mode == "Wikipedia search":
        num_pages = st.slider("Max pages", 1, 10, 3)

topic = st.text_input(
    "📝 Article Topic *",
    value="Mercedes-Benz electric vehicles",
    placeholder="e.g., Artificial Intelligence, Climate Change, etc."
)

# Conditional file upload
uploaded_docs = None
if mode == "Upload documents":
    uploaded_docs = st.file_uploader(
        "📎 Upload PDF/TXT files",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    if uploaded_docs:
        st.success(f"✅ {len(uploaded_docs)} file(s) uploaded")

# Generate button
if st.button("🚀 Generate Article", type="primary", use_container_width=True):
    if not api_key:
        st.error("❌ Cannot generate article without OpenAI API key. Please configure .env file.")
    elif not topic:
        st.error("❌ Please enter a topic")
    elif mode == "Upload documents" and not uploaded_docs:
        st.error("❌ Please upload at least one file for document mode")
    else:
        # Use agent system if selected
        if use_agents and mode == "Wikipedia search":
            with st.spinner("🤖 Research Agent gathering information..."):
                try:
                    article = two_agent_pipeline(topic)
                    st.session_state.article = article
                    st.session_state.mode = "agents"
                except Exception as e:
                    st.error(f"Error in agent pipeline: {str(e)}")
        else:
            # Standard RAG pipeline
            with st.spinner("🔍 Building context and generating article..."):
                try:
                    docs = []
                    
                    if mode == "Wikipedia search":
                        wiki_text = fetch_wikipedia(topic, max_pages=num_pages)
                        if not wiki_text:
                            st.error("No Wikipedia content found for that topic.")
                        else:
                            st.success(f"✅ Fetched {num_pages} Wikipedia pages")
                            docs = split_into_chunks(wiki_text)
                    else:
                        # Handle uploaded files
                        all_text = ""
                        for up in uploaded_docs:
                            if up.type == "application/pdf":
                                # Save temporarily
                                temp_path = f"temp_{up.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(up.getbuffer())
                                txt = load_pdf_text(temp_path)
                                all_text += "\n\n" + txt
                                os.remove(temp_path)  # Clean up
                            else:
                                text = up.getvalue().decode("utf-8")
                                all_text += "\n\n" + text
                        docs = split_into_chunks(all_text)
                        st.success(f"✅ Processed {len(docs)} text chunks")
                    
                    if docs:
                        with st.spinner("🧮 Creating vector embeddings..."):
                            retriever = build_vectorstore_from_docs(docs).as_retriever(
                                search_type="similarity", 
                                search_kwargs={"k": 4}
                            )
                        
                        with st.spinner("✍️ Generating article with GPT-4..."):
                            article = generate_article_from_retriever(retriever, topic)
                            st.session_state.article = article
                            st.session_state.mode = "rag"
                
                except Exception as e:
                    st.error(f"Error generating article: {str(e)}")

# Display article
if 'article' in st.session_state:
    st.markdown("---")
    
    # Parse article (simple headline extraction)
    article_text = st.session_state.article
    lines = article_text.split('\n')
    headline = lines[0] if lines else "Generated Article"
    body = '\n'.join(lines[1:]) if len(lines) > 1 else article_text
    
    # Display in newspaper format
    st.markdown('<div class="article-content">', unsafe_allow_html=True)
    st.markdown(f'<div class="article-headline">{headline}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="article-byline">By AI News Desk | Generated with {st.session_state.mode.upper()}</div>', unsafe_allow_html=True)
    st.markdown(body)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download TXT",
            data=article_text,
            file_name=f"article_{topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
    with col2:
        if st.button("📋 Copy to Clipboard"):
            st.code(article_text, language=None)
    with col3:
        if st.button("🔄 Generate New"):
            del st.session_state.article

# Footer
st.markdown("---")
st.caption("💡 Tip: Try different topics or upload your own documents for custom articles!")