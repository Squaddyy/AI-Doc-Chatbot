import streamlit as st
import fitz  # PyMuPDF
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(layout="wide")

# --- AI Model Caching ---
# Load the "brain" (embedding model) only once
@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model."""
    st.info("Loading AI 'brain' (embedding model)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("AI 'brain' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Load the model
model = load_embedding_model()

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        file_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
            
        pdf_document.close()
        return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def create_vector_store(text, model):
    """Creates a FAISS vector store from the text."""
    if not text or model is None:
        return None, None
        
    try:
        # Split text into manageable chunks (e.g., by paragraph)
        chunks = [para for para in text.split('\n') if para.strip()]
        if not chunks:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)] # Fallback chunking
        
        st.info(f"Indexing {len(chunks)} text chunks for the AI...")
        
        # --- This is the AI "reading" the document ---
        # 1. Embed the chunks
        embeddings = model.encode(chunks)
        
        # 2. Create the FAISS index (our vector database)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        
        st.success("Document has been 'read' and indexed by the AI.")
        
        # Return the index and the original text chunks
        return index, chunks
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None

def search_vector_store(query, model, index, chunks):
    """Searches the vector store for the most relevant text chunks."""
    try:
        # 1. Embed the user's question
        query_embedding = model.encode([query])
        
        # 2. Search the FAISS index
        k = 3  # Retrieve top 3 most relevant chunks
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        
        # 3. Get the actual text chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error searching vector store: {e}")
        return []

# --- Main App UI ---
st.title("🤖 AI Document Chatbot")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# Initialize session state variables
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = None
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = None

if uploaded_file is not None:
    # Process the file *once* and store in session state
    if st.session_state.doc_text is None:
        st.session_state.doc_text = extract_text_from_pdf(uploaded_file)
        
        if st.session_state.doc_text:
            # Create the vector store
            st.session_state.vector_index, st.session_state.text_chunks = create_vector_store(st.session_state.doc_text, model)
        else:
            st.error("Could not extract text from the PDF.")
else:
    # Clear session state if no file is uploaded
    st.session_state.doc_text = None
    st.session_state.vector_index = None
    st.session_state.text_chunks = None

# 3. Chat Interface
st.subheader("Chat with your document:")

if st.session_state.vector_index is not None:
    # Only show chat if the document is processed
    query = st.text_input("Ask a question about your document:")
    
    if query:
        # 1. Search the vector store
        relevant_chunks = search_vector_store(query, model, st.session_state.vector_index, st.session_state.text_chunks)
        
        st.subheader("AI Finding...")
        
        if relevant_chunks:
            # --- THIS IS THE "RETRIEVAL" PART ---
            # We display the relevant info we found.
            # In the *next* step, we'll feed this to a generator model.
            st.info("Found relevant information in your document:")
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"> {chunk}")
            
            # Placeholder for the final answer
            st.subheader("AI Answer (Coming Next):")
            st.warning("Next step: We will use a generative model to form a natural answer from this information.")
            
        else:
            st.warning("Couldn't find an answer in the document.")
else:
    st.info("Please upload a PDF to enable the chat.")