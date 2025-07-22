import os, pickle
from utils.pdf_utils import extract_chunks_from_pdf
from utils.embed_utils import build_faiss_index
from utils.retrieval_utils import build_bm25_index
import faiss
import numpy as np

all_chunks = []
pdf_folder = "data"

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        chunks = extract_chunks_from_pdf(os.path.join(pdf_folder, pdf_file))
        all_chunks.extend(chunks)

faiss_index, dense_embeddings = build_faiss_index(all_chunks)
faiss.write_index(faiss_index, "embeddings/faiss_index.faiss")
np.save("embeddings/dense_embeddings.npy", dense_embeddings)
with open("embeddings/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

bm25, tokenized_chunks = build_bm25_index(all_chunks)
with open("embeddings/bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, tokenized_chunks), f)
