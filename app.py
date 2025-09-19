import os
import streamlit as st
import pathway as pw
from openai import OpenAI
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

# Check if API keys are available
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()

# 1. Data Ingestion & Transformation (Backend)
def fetch_news_data(ticker):
    """
    Simulates a real-time news stream for a given stock ticker.
    In a real application, this would connect to a live API.
    """
    
    # Placeholder data for demonstration purposes
    data_map = {
        "AAPL": [
            {"title": "Apple stock soars on new iPhone announcement", "content": "The price of AAPL surged following a major product reveal.", "timestamp": time.time()},
            {"title": "Analysts raise price target for Apple", "content": "Several major financial firms have upgraded their forecasts for Apple's stock, citing strong sales.", "timestamp": time.time() + 5},
            {"title": "Apple acquires AI startup for $200M", "content": "The acquisition is seen as a strategic move to boost Apple's capabilities in generative AI.", "timestamp": time.time() + 10}
        ],
        "GOOGL": [
            {"title": "Google stock dips on regulatory news", "content": "GOOGL shares fell as new government regulations were proposed.", "timestamp": time.time()},
            {"title": "Google announces new powerful AI model", "content": "The company revealed a new, powerful AI model that could reshape the industry.", "timestamp": time.time() + 5}
        ]
    }
    
    if ticker.upper() in data_map:
        return pw.io.json.read_data(data_map[ticker.upper()], mode="append")
    else:
        # Return an empty stream if ticker is not in our placeholder data
        return pw.io.json.read_data([], mode="append")

# 2. Dynamic RAG Core with Pathway (Backend)
def create_rag_pipeline(ticker_stream):
    """
    Creates a Pathway pipeline for dynamic RAG.
    It ingests documents, embeds them, and builds a real-time index.
    """
    # Use the LLM xPack to handle embeddings and RAG
    documents = ticker_stream.select(
        id=pw.this.title,  # Using title as a unique ID
        text=pw.this.content
    )

    # Build the real-time vector index from the documents
    rag = pw.xpack.llm.RetrievalAugmentedGeneration(
        pw.xpack.llm.models.embeddings.OpenAIEmbedding(api_key=OPENAI_API_KEY),
        documents
    )
    return rag

# 3. Streamlit UI & Chatbot Logic (Frontend)
st.set_page_config(page_title="Real-Time Trading Co-Pilot", layout="wide")
st.title("📈 Real-Time Trading Co-Pilot")
st.markdown("Your live assistant for market intelligence. **Note:** This is a demo with simulated data for AAPL and GOOGL.")

# Session state to manage the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input for the stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, GOOGL):").upper()

if ticker:
    # Initialize the Pathway pipeline with the ticker
    ticker_stream = fetch_news_data(ticker)
    rag_pipeline = create_rag_pipeline(ticker_stream)

    # Display live news stream as it's ingested
    st.header(f"Live News Feed for {ticker}")
    live_news_placeholder = st.empty()

    # Pathway code that runs the stream and updates the UI
    @st.cache_resource
    def run_pathway_live():
        news_updates = ticker_stream.to_pandas_ddf().subscribe()
        return news_updates

    # Get the latest news from the stream
    news_feed = run_pathway_live()

    if not news_feed.empty:
        with live_news_placeholder.container():
            for _, row in news_feed.iterrows():
                st.info(f"**{row['title']}**\n\n{row['content']}")
    else:
        st.warning("No live news data found for this ticker in the demo.")

    # Chat interface
    st.header("Ask the Co-Pilot")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about this ticker..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Query the RAG pipeline
        response_stream = rag_pipeline.get_answer_as_streaming_response(
            prompt,
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini"
        )
        
        full_response = ""
        with st.chat_message("assistant"):
            response_container = st.empty()
            for chunk in response_stream:
                full_response += chunk.text
                response_container.markdown(full_response + "▌") # Typing indicator
            response_container.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})