import os
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", override=True)  # load .env file, overriding stale shell values


agent = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
print(f"OPEN_API_KEY: {os.getenv('OPENAI_API_KEY')}")

response = agent.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful coding assistant.",
    input="How do I check if a Python object is an instance of a class?"
)

print("openAI models:", agent.models)

#print(os.getenv('QDRANT_URL'))
qdrant = QdrantClient(url="http://localhost:6333", api_key=os.getenv('QDRANT_API_KEY'))
#print("Qdrant URL:", os.getenv('QDRANT_URL'))
#qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

response = qdrant.get_collections()
# Extract and print names
for collection in response.collections:
    print(collection.name)

#print(response)
