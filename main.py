from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import requests
import uuid

# Load environment variables
load_dotenv()

# Init FastAPI
app = FastAPI()

# Init Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create index if it doesn't exist
index_name = os.environ.get("PINECONE_INDEX")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # pas aan als je regio anders is
        )
    )

index = pc.Index(index_name)

# Init OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Models
class IngestRequest(BaseModel):
    title: str
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def read_root():
    return {"status": "up"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    # Embed de tekst
    embedding = client.embeddings.create(
        input=req.text,
        model="text-embedding-3-large"
    ).data[0].embedding

    # Voeg toe aan Pinecone
    index.upsert([{
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "title": req.title,
            "text": req.text
        }
    }])

    return {"status": "ingested", "title": req.title}

@app.post("/query")
def query(req: QueryRequest):
    # Embed de vraag
    embedding = client.embeddings.create(
        input=req.query,
        model="text-embedding-3-large"
    ).data[0].embedding

    # Zoek in Pinecone
    results = index.query(
        vector=embedding,
        top_k=req.top_k,
        include_metadata=True
    )

    # Bouw context op uit metadata
    context = "\n---\n".join([
        match["metadata"].get("text", "")
        for match in results["matches"]
    ])

    # Bouw prompt
    prompt = f"""Je bent een AI-assistent die helpt bij het beantwoorden van technische vragen op basis van projectnotities.
Gebruik alleen de onderstaande context om de vraag te beantwoorden. Als het antwoord niet in de context staat, zeg dan dat je het niet weet.

Context:
{context}

Vraag:
{req.query}

Antwoord:"""

    # Vraag GPT-4 om antwoord
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "answer": completion.choices[0].message.content,
        "context_used": context,
        "query": req.query
    }

