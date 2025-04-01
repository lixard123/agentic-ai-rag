import streamlit as st
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Load API keys securely
openai_api_key = st.secrets.get("OPENAI_API_KEY_1", "")
weather_api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
flight_api_key = st.secrets.get("FLIGHT_API_KEY", "")

if not openai_api_key:
    st.error("Missing OpenAI API key. Please add it to Streamlit Secrets.")
    st.stop()  # Stop execution if no API key

# Function to load and vectorize PDFs
def load_and_vectorize_pdfs(pdf_folder):
    """Loads and vectorizes PDFs from the specified folder."""
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Pass API key explicitly
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Function to fetch live weather data
def get_weather(city):
    """Fetch real-time weather information using OpenWeatherMap API."""
    if not weather_api_key:
        return "Weather API key is missing."
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"{data['weather'][0]['description'].capitalize()}, Temperature: {data['main']['temp']}°C"
    else:
        return "Weather data not available."

# Function to fetch flight details
def get_flight_details(origin, destination, date):
    """Fetch real-time flight details using an external API."""
    if not flight_api_key:
        return "Flight API key is missing."

    url = f"https://api.flightapi.io/search/{flight_api_key}/{origin}/{destination}/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "flights" in data and data["flights"]:
            flight = data["flights"][0]  # Get the first available flight
            return f"Flight {flight['flight_number']} from {flight['departure']} to {flight['arrival']} on {flight['date']} at {flight['time']}"
        else:
            return "No flights found."
    else:
        return "Flight data not available."

def main():
    st.set_page_config(page_title="Agentic AI Travel Assistant", page_icon="🌍", layout="centered")
    
    st.title("🌍 Agentic AI Travel Assistant")
    st.write("Hello! I'm your AI travel assistant, ready to help you with:")

    features = {
        "📍 Places": "Get details about cities, landmarks, and hidden gems.",
        "🌦️ Weather": "Real-time weather forecasts for any location.",
        "🍽️ Cuisines": "Discover local and international food specialties.",
        "🏝️ Destinations": "Explore top tourist attractions and experiences.",
        "🛫 Travel Packages": "Find the best travel deals from our brochures.",
        "✈️ Flights": "Check flight details and availability."
    }

    for icon, description in features.items():
        st.markdown(f"- {icon}: {description}")

    st.write("Ask me anything related to your travel needs!")
    user_query = st.text_input("What would you like to know?", "Best tourist spots in Paris")
    
    pdf_folder = "brochures"  # Folder where PDFs are stored
    vectorstore = load_and_vectorize_pdfs(pdf_folder)
    retriever = vectorstore.as_retriever()

    # Initialize OpenAI LLM with API Key
    llm = OpenAI(openai_api_key=openai_api_key)
    qa_chain = RetrievalQA(llm=llm, retriever=retriever,chain_type="stuff")
    
    if st.button("Get Information"):
        with st.spinner("Fetching details..."):
            if "weather" in user_query.lower():
                city = user_query.split(" in ")[-1]
                response = get_weather(city)
            elif "flight" in user_query.lower():
                parts = user_query.split(" from ")[-1].split(" to ")
                if len(parts) == 2:
                    origin, destination = parts
                    date = "2025-04-01"  # Placeholder date; integrate user input if needed
                    response = get_flight_details(origin.strip(), destination.strip(), date)
                else:
                    response = "Please specify the flight origin and destination."
            else:
                response = qa_chain.run(user_query)
            st.success(f"**{user_query}**: {response}")

if __name__ == "__main__":
    main()
