import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import logging

# --- Setup ---
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- Pinecone setup ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# --- OpenAI setup ---
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-large"

# --- Request modellen ---
class IngestRequest(BaseModel):
    title: str
    content: str  # <-- aangepast van 'text'

class QueryRequest(BaseModel):
    query: str

# --- Ingest endpoint ---
@app.post("/ingest")
async def ingest(data: IngestRequest):
    try:
        logging.info(f"Ontvangen data: {data}")
        embedding = openai.embeddings.create(
            input=f"{data.title}\n{data.content}",
            model=EMBEDDING_MODEL
        ).data[0].embedding

        index.upsert(vectors=[{
            "id": data.title.replace(" ", "_"),
            "values": embedding,
            "metadata": {
                "title": data.title,
                "text": data.content
            }
        }])
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Fout tijdens ingest: {e}")
        return {"error": str(e)}

# --- Query endpoint ---
@app.post("/query")
async def query(data: QueryRequest):
    try:
        embedding = openai.embeddings.create(
            input=data.query,
            model=EMBEDDING_MODEL
        ).data[0].embedding

        result = index.query(vector=embedding, top_k=1, include_metadata=True)

        if result.matches:
            best = result.matches[0].metadata
            return {
                "antwoord": f'De melding "{best["title"]}" is inderdaad gevonden in de documentatie, maar er is geen verdere uitleg van detail over wat de storing inhoudt of hoe deze opgelost moet worden.\n\nHiervoor raad ik aan contact op te nemen met de MRP of een specialist die meer inzicht heeft.\n\nWil je dat ik ook zoek naar algemene meldingen met dit type storing?',
                "score": result.matches[0].score,
                "metadata": best
            }
        else:
            return {"antwoord": "Ik kon niets vinden in de vector data."}
    except Exception as e:
        logging.error(f"Fout tijdens query: {e}")
        return {"error": str(e)}

