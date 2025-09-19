import os
import pathway as pw
from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Check if API keys are available
if not OPENAI_API_KEY:
    print("Error: OpenAI API key not found. Please add it to your .env file.")
    exit()

# 1. Data Ingestion & Transformation
def fetch_news_data(ticker):
    """
    Simulates a real-time news stream for a given stock ticker.
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

    # Check if the ticker exists in our data map
    if ticker.upper() in data_map:
        data = data_map[ticker.upper()]
    else:
        # Provide an empty list
        data = []

    # Create a Pathway table from the in-memory list
    return pw.new(
        data=data,
        schema_model=pw.Schema(
            title=str,
            content=str,
            timestamp=float
        )
    )

# 2. Dynamic RAG Core with Pathway
def create_rag_pipeline(ticker_stream):
    """
    Creates a Pathway pipeline for dynamic RAG.
    It ingests documents, embeds them, and builds a real-time index.
    """
    documents = ticker_stream.select(
        id=pw.this.title,
        text=pw.this.content
    )

    rag = pw.xpack.llm.RetrievalAugmentedGeneration(
        pw.xpack.llm.models.embeddings.OpenAIEmbedding(api_key=OPENAI_API_KEY),
        documents
    )
    return rag

# 3. Main execution block (run from terminal)
if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g., AAPL, GOOGL): ").upper()
    query = input("Ask a question about this ticker: ")

    # Initialize the Pathway pipeline with the ticker
    ticker_stream = fetch_news_data(ticker)
    rag_pipeline = create_rag_pipeline(ticker_stream)

    # Get the answer from the RAG pipeline and print it
    print("\nThinking...\n")
    response_stream = rag_pipeline.get_answer_as_streaming_response(
        query,
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )

    full_response = ""
    for chunk in response_stream:
        full_response += chunk.text
        print(chunk.text, end="")
    
    print("\n")
