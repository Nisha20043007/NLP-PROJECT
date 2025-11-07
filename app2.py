# app2.py
import streamlit as st
import os
import hashlib
import json
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

# -----------------------------
# CONFIGURATION AND LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

FAISS_INDEX_PATH = "faiss_index_user_docs"
INDEX_METADATA_PATH = "index_metadata.json"
DATA_DIR = "data"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def compute_bytes_hash(content_bytes: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    hasher = hashlib.sha256()
    hasher.update(content_bytes)
    return hasher.hexdigest()


def compute_file_hash(file_path: str) -> str | None:
    """Compute SHA256 hash of a file (path)."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error computing file hash for {file_path}: {str(e)}")
        return None


def load_existing_index(file_hashes: dict):
    """Load FAISS index if metadata matches uploaded files."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(INDEX_METADATA_PATH):
        logger.info("FAISS index or metadata not present.")
        return None

    try:
        with open(INDEX_METADATA_PATH, "r") as f:
            stored_metadata = json.load(f)
        if stored_metadata == file_hashes:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info("✅ Loaded existing FAISS index (no file changes).")
            return db
        else:
            logger.info("⚠️ Uploaded files differ. Rebuilding FAISS index.")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
    return None


def create_new_index_from_paths(file_paths: list[str], file_hashes: dict):
    """Create new FAISS index from file paths (PDFs already saved to disk)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    all_documents = []
    problems = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    for p in file_paths:
        try:
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                problems.append(f"{os.path.basename(p)} is missing or empty.")
                logger.error(f"Skipping {p} because it's missing or empty.")
                continue

            loader = PyPDFLoader(p)
            docs = loader.load()
            if not docs:
                problems.append(f"No readable pages found in {os.path.basename(p)}.")
                logger.error(f"No docs loaded for {p}")
                continue

            documents = text_splitter.split_documents(docs)
            all_documents.extend(documents)
        except Exception as e:
            problems.append(f"Error loading {os.path.basename(p)}: {str(e)}")
            logger.error(f"Error loading PDF {p}: {e}")

    if problems:
        for pr in problems:
            st.warning(pr)

    if not all_documents:
        st.error("No documents loaded. Please upload valid PDF files.")
        return None

    # Build FAISS DB
    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    # Save metadata (hashes)
    with open(INDEX_METADATA_PATH, "w") as f:
        json.dump(file_hashes, f)

    logger.info("✅ FAISS index created and saved successfully.")
    return db


# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Question Answering System", page_icon="🤖", layout="wide")
    st.title("📘 Question Answering System Over Custom Documents")
    st.markdown("""
    ### 🔍 Description
    Upload PDF documents and ask questions. The app creates an embeddings index (FAISS)
    and uses a retrieval + LLM chain to answer  queries from the uploaded documents.
    """)

    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = []

    # Sidebar for previous queries
    st.sidebar.header("🕓 Previous Queries")
    if st.session_state.previous_searches:
        for idx, query in enumerate(st.session_state.previous_searches):
            st.sidebar.write(f"{idx + 1}. {query}")
    else:
        st.sidebar.write("No previous searches yet.")

    # File upload
    uploaded_files = st.file_uploader("📤 Upload your PDF files here", type=["pdf"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Please upload one or more PDF documents to begin.")
        return

    # Ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Read each uploaded file ONCE, save bytes to disk, compute hash
    saved_paths = []
    file_hashes = {}
    for ufile in uploaded_files:
        try:
            # read bytes once
            ufile_bytes = ufile.read()
            if not ufile_bytes:
                st.error(f"{ufile.name} appears to be empty. Please re-upload a valid PDF.")
                logger.error(f"Empty upload: {ufile.name}")
                continue

            temp_path = os.path.join(DATA_DIR, ufile.name)

            # write bytes to disk
            with open(temp_path, "wb") as tmp:
                tmp.write(ufile_bytes)

            # compute hash from bytes (guaranteed consistent with file content)
            file_hash = compute_bytes_hash(ufile_bytes)
            file_hashes[temp_path] = file_hash
            saved_paths.append(temp_path)

            logger.info(f"Saved {ufile.name} -> {temp_path} (hash: {file_hash})")
        except Exception as e:
            st.error(f"Error saving uploaded file {ufile.name}: {e}")
            logger.error(f"Error saving {ufile.name}: {e}")

    if not saved_paths:
        st.error("No valid files were uploaded. Aborting.")
        return

    # Try to load existing index if file set hasn't changed
    db = load_existing_index(file_hashes)
    if db is None:
        with st.spinner("📚 Creating FAISS index..."):
            db = create_new_index_from_paths(saved_paths, file_hashes)
    if db is None:
        return

    retriever = db.as_retriever()

    # GROQ API key check
    try:
        groq_api_key = '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'
    except Exception:
        st.error("⚠️ GROQ_API_KEY not found in Streamlit secrets. Please configure it in your environment.")
        return

    # Initialize the LLM
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    except Exception as e:
        st.error(f"Error initializing ChatGroq LLM: {e}")
        logger.error(f"LLM init error: {e}")
        return

    # Create retrieval and document chains
    prompt_template = PromptTemplate(
        template="""
        You are an intelligent assistant for answering questions from uploaded documents.
        Use ONLY the context below to answer.
        If the context does not contain the answer, say "I couldn't find the answer in the provided documents."

        <context>
        {context}
        </context>

        Question: {input}
        """,
        input_variables=["context", "input"]
    )

    # Create the document chain and retrieval chain (legacy/classic style)
    try:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"Error creating chains: {e}")
        logger.error(f"Chain creation error: {e}")
        return

    # Query interface
    query = st.text_input("💬 Ask a question about your uploaded documents:")
    if query:
        st.session_state.previous_searches.append(query)
        with st.spinner("🤔 Searching and generating answer..."):
            try:
                response = retrieval_chain.invoke({"input": query})
                # The response shape may vary depending on the chain impl.
                # Try to display the most likely fields.
                if isinstance(response, dict):
                    if "answer" in response:
                        st.success("✅ Answer:")
                        st.write(response["answer"])
                    elif "output_text" in response:
                        st.success("✅ Answer:")
                        st.write(response["output_text"])
                    else:
                        # fallback: show the whole response for debugging
                        st.write(response)
                else:
                    # If the chain returned a string directly:
                    st.success("✅ Answer:")
                    st.write(response)
            except Exception as e:
                st.error(f"Error while processing your query: {str(e)}")
                logger.error(f"Query error: {str(e)}")


if __name__ == "__main__":
    main()

