from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS (voor test vanaf Zapier of browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestPayload(BaseModel):
    id: str
    title: str
    content: str

@app.post("/ingest")
async def ingest(payload: IngestPayload):
    full_text = f"{payload.title}\n{payload.content}"
    # Genereer embedding (test zonder opslaan)
    response = openai.Embedding.create(
        input=full_text,
        model="text-embedding-ada-002"
    )
    vector = response['data'][0]['embedding']
    print(f"Ingested {payload.title} ({len(vector)} dims)")
    return {"status": "ok", "vector_dim": len(vector)}
