from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# 🔐 API Key laden
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS = extern toegang toestaan (voor Zapier)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧾 Input-model
class IngestPayload(BaseModel):
    id: str
    title: str
    content: str

# 📥 POST endpoint
@app.post("/ingest")
async def ingest(payload: IngestPayload):
    full_text = f"{payload.title}\n{payload.content}"

    # 🧠 Embedding via OpenAI 1.x
    response = client.embeddings.create(
        input=full_text,
        model="text-embedding-ada-002"
    )

    vector = response.data[0].embedding

    print(f"Ingested '{payload.title}' → {len(vector)} dims")

    return {
        "status": "ok",
        "vector_dim": len(vector)
    }
