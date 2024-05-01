import json
import chromadb
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from pydantic import BaseModel, Field

with open("photo_caption_embed_mapping.json", "rb") as f:
    images_with_embeddings = json.load(f)

embed_model = "Alibaba-NLP/gte-large-en-v1.5"
model = SentenceTransformer(embed_model, trust_remote_code=True)

chroma_client = chromadb.Client()
collection_name = "image_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

collection.add(
    documents=images_with_embeddings["captions"],
    metadatas=[{"path": p} for p in images_with_embeddings["paths"]],
    embeddings=images_with_embeddings["embeddings"],
    ids=["id1", "id2"]
)

vector_store = Chroma(client=chroma_client, collection_name=collection_name)

class RetrieveImage(BaseModel):
    """Call this function to find the best image the user is looking for with a description of an image"""
    query: str = Field(description="description of an image an user wants to find")

def retrieve_image(query: str) -> str:
    query_embedding = model.encode(query)
    result = vector_store.similarity_search_by_vector_with_relevance_scores(query_embedding.tolist(), k=1)
    score = result[0][1]
    path = result[0][0].metadata["path"]
    return path, score