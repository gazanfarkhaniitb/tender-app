import streamlit as st
from dotenv import load_dotenv
import pickle
import pdfplumber
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import os
import numpy as np
import openai
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Tender App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built for analyzing tenders using:
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Gak and Ashar')

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read PDF using PDFPlumber
def read_pdf(pdf_path):
    text = ""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            text += page_text
            pages.append(page_text)
    return text, pages

# Function to split text into chunks
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

# Function to generate embeddings using OpenAI
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

# Function to normalize embeddings for cosine similarity
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = normalize_embeddings(embeddings)
    index.add(normalized_embeddings)
    return index

# Function to query embeddings and find top 3 relevant chunks
def query_embeddings(query, index, chunks, embeddings, k=3):
    query_embedding = get_embedding(query)
    normalized_query_embedding = normalize_embeddings([query_embedding])[0]
    cosine_similarities = cosine_similarity([normalized_query_embedding], embeddings)[0]
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    relevant_chunks = [chunks[idx] for idx in top_k_indices]
    return relevant_chunks

# Streamlit App
def app():
    st.header("Chat with your PDF Tender Document 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your Tender here", type='pdf')

    if pdf is not None:
        pdf_path = pdf.name
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        text, pages = read_pdf(pdf_path)
        chunks = split_text(text)

        store_name = pdf.name[:-4]
        embeddings_file = f"{store_name}_embeddings.pkl"
        chunks_file = f"{store_name}_chunks.pkl"
        index_file = f"{store_name}_faiss.index"

        if os.path.exists(index_file) and os.path.exists(chunks_file) and os.path.exists(embeddings_file):
            index = faiss.read_index(index_file)
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            with open(embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
            st.write('Embeddings loaded from disk')
        else:
            embeddings = generate_embeddings_for_chunks(chunks)
            index = store_embeddings_in_faiss(embeddings)
            faiss.write_index(index, index_file)
            with open(chunks_file, "wb") as f:
                pickle.dump(chunks, f)
            with open(embeddings_file, "wb") as f:
                pickle.dump(embeddings, f)
            st.write('Embeddings generated and stored on disk')

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            top_chunks = query_embeddings(query, index, chunks, embeddings)
            
            # Wrap chunks in Document objects
            docs = [Document(page_content=chunk) for chunk in top_chunks]

            # Define a more sophisticated prompt
            sophisticated_prompt = f"""
            I run a firm. In order to get projects, I need to bid for tenders. Now, I want to know if I can bid for a project or not. And that depends on this tender. So when you read and answer, make sure to answer as if you're an expert assistant in the tendering field. Given the following documents extracted from a tender PDF, provide an accurate response to the question below. Be concise and to the point.
            Question: {query}
            Documents: {docs}
            """

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query, prompt=sophisticated_prompt)
                st.write(response)

            # Update chat history
            st.session_state.chat_history.append((query, response))

        # Display chat history
        if st.session_state.chat_history:
            st.write("### Chat History")
            for i, (user_query, bot_response) in enumerate(st.session_state.chat_history):
                st.write(f"**User:** {user_query}")
                st.write(f"**Chatbot:** {bot_response}")

            # Clear chat history button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.write("Chat history cleared.")

if __name__ == '__main__':
    app()
