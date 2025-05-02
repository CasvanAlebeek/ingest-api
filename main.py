from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import openai
from pinecone import Pinecone
import uuid
import logging

# Laad .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# Init Pinecone en OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-large"

# Start app
app = FastAPI()

# --- Request modellen ---
class IngestRequest(BaseModel):
    title: str
    text: str

class QueryRequest(BaseModel):
    query: str

# --- Routes ---

@app.post("/ingest")
async def ingest(request: Request):
    try:
        body = await request.json()
        logging.info(f"Ontvangen data: {body}")
        data = IngestRequest(**body)
    except ValidationError as ve:
        logging.error(f"Validatiefout: {ve}")
        raise HTTPException(status_code=422, detail="Invoer is niet geldig. Vereist: 'title' en 'text' als strings.")
    except Exception as e:
        logging.error(f"Algemene fout bij lezen JSON: {e}")
        raise HTTPException(status_code=400, detail="Ongeldige JSON structuur.")

    # Embedding maken
    try:
        response = openai.embeddings.create(
            input=data.text,
            model=EMBED_MODEL
        )
        vector = response.data[0].embedding
        logging.info("Embedding succesvol aangemaakt.")
    except Exception as e:
        logging.error(f"Fout bij OpenAI embedding: {e}")
        raise HTTPException(status_code=500, detail="Fout bij het genereren van de embedding.")

    # Upsert naar Pinecone
    try:
        pinecone_id = str(uuid.uuid4())
        index.upsert([
            (pinecone_id, vector, {"title": data.title, "text": data.text})
        ])
        logging.info(f"Succesvol opgeslagen in Pinecone met ID {pinecone_id}")
        return {"status": "ok", "id": pinecone_id}
    except Exception as e:
        logging.error(f"Fout bij Pinecone upsert: {e}")
        raise HTTPException(status_code=500, detail="Fout bij opslaan in Pinecone.")


@app.post("/query")
async def query(req: QueryRequest):
    try:
        embedding_response = openai.embeddings.create(
            input=req.query,
            model=EMBED_MODEL
        )
        query_vector = embedding_response.data[0].embedding

        pinecone_response = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        matches = pinecone_response.get('matches', [])
        return {"matches": matches}

    except Exception as e:
        logging.error(f"Fout bij query: {e}")
        raise HTTPException(status_code=500, detail="Fout tijdens query.")
