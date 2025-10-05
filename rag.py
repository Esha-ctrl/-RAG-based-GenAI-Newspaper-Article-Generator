# rag.py
import os
import wikipedia
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader

load_dotenv()

# Environment
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# --------------------------------------------------------
# 🔍 Wikipedia Fetch
# --------------------------------------------------------
def fetch_wikipedia(topic: str, max_pages: int = 3) -> str:
    """Fetch and concatenate multiple Wikipedia pages"""
    try:
        results = wikipedia.search(topic, results=max_pages)
        content = []
        for r in results:
            try:
                page = wikipedia.page(r)
                content.append(page.content)
            except Exception:
                continue
        return "\n\n".join(content)
    except Exception as e:
        print(f"❌ Wikipedia fetch failed: {e}")
        return ""

# --------------------------------------------------------
# 📄 PDF Reader
# --------------------------------------------------------
def load_pdf_text(path: str) -> str:
    """Extracts raw text from a PDF file"""
    text = ""
    with open(path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# --------------------------------------------------------
# ✂️ Chunking
# --------------------------------------------------------
def split_into_chunks(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.create_documents([text])
    return docs

# --------------------------------------------------------
# 🧠 Vectorstore Builder
# --------------------------------------------------------
def build_vectorstore_from_docs(docs: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --------------------------------------------------------
# 📰 Article Generator
# --------------------------------------------------------
def generate_article_from_retriever(retriever, topic: str) -> str:
    """Generate a concise newspaper article based on retrieved content."""
    llm = ChatOpenAI(model=MODEL, temperature=0.4)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    prompt = (
        f"You are a journalist writing a newspaper article about '{topic}'. "
        "Use the retrieved information to write a catchy headline followed by "
        "exactly three paragraphs (3-5 sentences each). Keep the tone informative and engaging."
    )

    response = qa_chain.run(prompt)
    return response
