# Ingest API

This FastAPI application provides two endpoints for ingesting and querying text data that is stored in Pinecone and embedded with OpenAI.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with the following variables:
   ```bash
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX=your-index-name
   OPENAI_API_KEY=your-openai-api-key
   ```

3. Start the API using uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

The service will be available at `http://localhost:8000`.
