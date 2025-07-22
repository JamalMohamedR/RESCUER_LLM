from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", device='cuda',trust_remote_code=True)

def build_faiss_index(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=32, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings
