import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

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
    problem: str
    solution: str
    machine: str
    type: str
    project: str
    line: str | None = None  # optioneel

class QueryRequest(BaseModel):
    query: str

# --- Ingest endpoint ---
@app.post("/ingest")
async def ingest(data: IngestRequest):
    try:
        logging.info(f"Ontvangen data: {data}")
        combined_text = f"{data.title}\n{data.problem}\n{data.solution}\n{data.machine}\n{data.type}\n{data.project}"
        if data.lijn:
            combined_text += f"\n{data.lijn}"

        embedding = openai.embeddings.create(
            input=combined_text,
            model=EMBEDDING_MODEL
        ).data[0].embedding

        index.upsert(vectors=[{
            "id": data.title.replace(" ", "_"),
            "values": embedding,
            "metadata": data.dict()
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

        result = index.query(vector=embedding, top_k=5, include_metadata=True)

        if result.matches:
            best = result.matches[0].metadata
            return {
                "antwoord": f'Melding gevonden: "{best["title"]}".\n\nProbleem: {best["problem"]}\n\nOplossing: {best["solution"]}',
                "score": result.matches[0].score,
                "metadata": best
            }
        else:
            return {"antwoord": "Ik kon niets vinden in de vector data."}
    except Exception as e:
        logging.error(f"Fout tijdens query: {e}")
        return {"error": str(e)}



