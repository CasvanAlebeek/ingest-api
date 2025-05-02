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
    problem: str
    solution: str
    machine: str
    type: str
    project: str

class QueryRequest(BaseModel):
    query: str

# --- Ingest endpoint ---
@app.post("/ingest")
async def ingest(data: IngestRequest):
    try:
        logging.info(f"Ontvangen data: {data}")

        embedding_input = f"Title: {data.title}\nProblem: {data.problem}\nSolution: {data.solution}\nMachine: {data.machine}\nType: {data.type}\nProject: {data.project}"

        embedding = openai.embeddings.create(
            input=embedding_input,
            model=EMBEDDING_MODEL
        ).data[0].embedding

        index.upsert(vectors=[{
            "id": data.title.replace(" ", "_"),
            "values": embedding,
            "metadata": data.model_dump()
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
                "antwoord": f"Beste match:\n\nTitel: {best['title']}\n\nProbleem: {best['problem']}\nOplossing: {best['solution']}\n\nMachine: {best['machine']} | Type: {best['type']} | Project: {best['project']}",
                "score": result.matches[0].score,
                "metadata": best
            }
        else:
            return {"antwoord": "Ik kon niets vinden in de vector data."}
    except Exception as e:
        logging.error(f"Fout tijdens query: {e}")
        return {"error": str(e)}


