from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
from openai import OpenAI
import pinecone

# 🌱 Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# 🤖 OpenAI client
client = OpenAI(api_key=openai_api_key)

# 📦 Pinecone setup
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index_name)

# ⚙️ FastAPI app
app = FastAPI()

@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()

    # ✅ Extract data
    item_id = data.get("id")
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = f"{title}\n{content}"

    # 🧠 Maak embedding aan via OpenAI
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=full_text
    )

    embedding = response.data[0].embedding

    # 📤 Verstuur embedding naar Pinecone
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

    return {"status": "ok", "vector_dim": len(embedding)}

