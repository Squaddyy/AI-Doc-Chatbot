import streamlit as st
import fitz  # This is the PyMuPDF library
import io

st.set_page_config(layout="wide")

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        # Open the PDF file from bytes
        file_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        full_text = ""
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
            
        pdf_document.close()
        return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# --- Main App UI ---
st.title("🤖 AI Document Chatbot")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    st.info("File uploaded successfully. Processing...")
    
    # 2. Extract Text
    document_text = extract_text_from_pdf(uploaded_file)
    
    if document_text:
        st.success("PDF processed. Text extracted successfully.")
        
        # Display a preview of the extracted text (e.g., first 500 chars)
        st.subheader("Extracted Text Preview:")
        st.text_area("Text", document_text[:500] + "...", height=150)
        
        # Placeholder for the chat
        st.subheader("Chat with your document:")
        st.text_input("Ask a question about your document:")
        
    else:
        st.error("Could not extract text from the PDF. Please try another file.")
else:
    st.info("Please upload a PDF to get started.")