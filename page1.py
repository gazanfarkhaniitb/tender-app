import streamlit as st
import fitz  # PyMuPDF
import openai
import numpy as np
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Load PDF and extract text."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"Error opening {pdf_path}: {e}")
        return ""
    
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, max_tokens=3000):
    """Split text into chunks of max_tokens size."""
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    chunks = []
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embeddings for the given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings for each chunk."""
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    return np.array(embeddings)

def extract_information_with_openai(text_chunk, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500  # Adjusting max tokens to handle the output size
    )
    return response['choices'][0]['message']['content'].strip()

def generate_summary_and_extract_info(chunks):
    summary_prompt = "I am a contractor and want to know if I can bid for this project or not. So please generate a 500-word technical summary of the project from the following text. The summary should include the name of the project, location of the project site, and important dates (start, end, other key dates) of the project. Make sure to give information in concise points."
    summary = ""
    for chunk in chunks:
        if not summary:
            summary = extract_information_with_openai(chunk, summary_prompt + chunk)
    return summary

def save_chat_to_pdf(chat_history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for entry in chat_history:
        paragraphs = entry.split('\n')
        for para in paragraphs:
            story.append(Paragraph(para, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

def app():
    st.header("Summarize and Chat with your PDF Tender Document 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your Tender here", type='pdf')

    if pdf is not None:
        pdf_path = pdf.name
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)

        # Generate and display summary
        if "summary" not in st.session_state:
            st.session_state.summary = generate_summary_and_extract_info(chunks)
        
        st.write("Summary of the Project:")
        st.write(st.session_state.summary)

        if st.button("Regenerate Summary"):
            st.session_state.summary = generate_summary_and_extract_info(chunks)
            st.write(st.session_state.summary)

        if st.button("Save Chat"):
            chat_history = [st.session_state.summary]
            pdf_buffer = save_chat_to_pdf(chat_history)
            st.download_button("Download Summary as PDF", data=pdf_buffer, file_name="summary.pdf", mime="application/pdf")
