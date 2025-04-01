import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

def load_and_vectorize_pdfs(pdf_folder):
    """Loads and vectorizes PDFs from the specified folder."""
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="Agentic AI Travel Assistant", page_icon="🌍", layout="centered")
    
    st.title("🌍 Agentic AI Travel Assistant")
    st.write("Hello! I'm your AI travel assistant, ready to help you with:")

    features = {
        "📍 Places": "Get details about cities, landmarks, and hidden gems.",
        "🌦️ Weather": "Real-time weather forecasts for any location.",
        "🍽️ Cuisines": "Discover local and international food specialties.",
        "🏝️ Destinations": "Explore top tourist attractions and experiences.",
        "🛫 Travel Packages": "Find the best travel deals from our brochures."
    }

    for icon, description in features.items():
        st.markdown(f"- {icon}: {description}")

    st.write("Ask me anything related to your travel needs!")
    user_query = st.text_input("What would you like to know?", "Best tourist spots in Paris")
    
    pdf_folder = "brochures"  # Folder where PDFs are stored
    vectorstore = load_and_vectorize_pdfs(pdf_folder)
    retriever = vectorstore.as_retriever()
    llm = OpenAI()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    
    if st.button("Get Information"):
        with st.spinner("Fetching details..."):
            response = qa_chain.run(user_query)
            st.success(f"**{user_query}**: {response}")
    
if __name__ == "__main__":
    main()
