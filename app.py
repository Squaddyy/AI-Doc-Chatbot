import streamlit as st
import fitz  # PyMuPDF
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline  # --- NEW IMPORT ---

st.set_page_config(layout="wide")

# --- AI Model Caching ---

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model (for retrieval)."""
    st.info("Loading AI 'Retrieval' model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("AI 'Retrieval' model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_qa_model():
    """Loads the QA model (for generation)."""
    st.info("Loading AI 'Answer' model...")
    try:
        # Load a model specifically for Question Answering
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        st.success("AI 'Answer' model loaded!")
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        return None

# Load the models
retriever = load_embedding_model()
generator = load_qa_model()

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

@st.cache_resource(hash_funcs={type(retriever): id}) # Handle caching for the model object
def create_vector_store(_retriever, text): # Pass retriever as argument
    """Creates a FAISS vector store from the text."""
    if not text or _retriever is None:
        return None, None
        
    try:
        chunks = [para for para in text.split('\n') if para.strip()]
        if not chunks:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)] 
        
        st.info(f"Indexing {len(chunks)} text chunks for the AI...")
        embeddings = _retriever.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        st.success("Document has been 'read' and indexed by the AI.")
        
        return index, chunks
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None

def search_vector_store(query, _retriever, index, chunks):
    """Searches the vector store for the most relevant text chunks."""
    try:
        query_embedding = _retriever.encode([query])
        k = 3
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        
        # --- THIS IS THE KEY ---
        # Combine the relevant chunks into a single "context"
        context = " ".join(relevant_chunks)
        return context
    except Exception as e:
        st.error(f"Error searching vector store: {e}")
        return ""

# --- Main App UI ---
st.title("🤖 AI Document Chatbot")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if 'doc_text' not in st.session_state:
    st.session_state.doc_text = None
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = None

if uploaded_file is not None:
    # Process file *once*
    if st.session_state.doc_text is None:
        st.session_state.doc_text = extract_text_from_pdf(uploaded_file)
        
        if st.session_state.doc_text:
            st.session_state.vector_index, st.session_state.text_chunks = create_vector_store(retriever, st.session_state.doc_text)
        else:
            st.error("Could not extract text from the PDF.")
else:
    st.session_state.doc_text = None
    st.session_state.vector_index = None
    st.session_state.text_chunks = None

# 3. Chat Interface
st.subheader("Chat with your document:")

if st.session_state.vector_index is not None and generator is not None:
    query = st.text_input("Ask a question about your document:")
    
    if query:
        # 1. RETRIEVAL
        st.info("Finding relevant information...")
        context = search_vector_store(query, retriever, st.session_state.vector_index, st.session_state.text_chunks)
        
        if context:
            # 2. GENERATION
            st.info("Generating your answer...")
            with st.spinner("AI is thinking..."):
                result = generator(question=query, context=context)
                answer = result["answer"]
                
                st.subheader("AI Answer:")
                st.success(f"**{answer}**")
                
                # Optionally, show the source
                with st.expander("Show source context"):
                    st.markdown(f"> {context}")
            
        else:
            st.warning("Couldn't find an answer in the document.")
else:
    if retriever is None or generator is None:
        st.error("AI models failed to load. Please check the console.")
    else:
        st.info("Please upload a PDF to enable the chat.")