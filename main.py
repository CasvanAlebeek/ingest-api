from fastapi import FastAPI, Request
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# 🔄 .env variabelen laden
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# 🧠 OpenAI client aanmaken
client = OpenAI(api_key=openai_api_key)

# 🌲 Pinecone client (nieuwe SDK)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# 🚀 FastAPI app
app = FastAPI()

@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()
    item_id = data.get("id")
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title}\n{content}"

    # 🎯 Embedding maken via OpenAI
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=full_text
    )
    embedding = response.data[0].embedding

    # 💾 Upsert naar Pinecone
    index.upsert([
        {
            "id": item_id,
            "values": embedding,
            "metadata": {
                "title": title,
                "text": content
            }
        }
    ])

    return {"status": "ok"}

