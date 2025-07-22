import os
import pickle
import faiss
import numpy as np
from utils.embed_utils import model as embed_model
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def clip_prompt(prompt: str, tokenizer, max_tokens: int = 4000):
    tokens = tokenizer.encode(prompt, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


# Enable expandable segments to reduce fragmentation OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load indexes and chunks
faiss_index = faiss.read_index("embeddings/faiss_index.faiss")
dense_embeddings = np.load("embeddings/dense_embeddings.npy")
chunks = pickle.load(open("embeddings/chunks.pkl", "rb"))
bm25, tokenized_chunks = pickle.load(open("embeddings/bm25_index.pkl", "rb"))

# Check CUDA availability and decide on quantization
device = "cpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with fallback (no quantization due to CUDA issues)
phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    attn_implementation="eager"  # Fix for flash attention warning
)

# Load tokenizer
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
if phi_tokenizer.pad_token is None:
    phi_tokenizer.pad_token = phi_tokenizer.eos_token

# Pipeline initialization - don't specify device when using accelerate
pipe = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer)

# Your existing functions remain the same...
def hybrid_retrieve(query, top_k=5):
    # Dense retrieval
    query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, dense_indices = faiss_index.search(query_emb, top_k)
    dense_results = [chunks[idx] for idx in dense_indices[0]]

    # BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [chunks[idx] for idx in top_bm25_indices]

    # Merge and deduplicate
    retrieved_chunks = list(dict.fromkeys(dense_results + bm25_results))
    return retrieved_chunks[:top_k]

def generate_structured_response(query):
    retrieved_chunks = hybrid_retrieve(query)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are an insurance domain AI assistant. Based on the query:
"{query}"

and the following policy document clauses:
{context}

Determine:
- Whether the procedure is covered.
- If covered, the payout amount if applicable.
- Reference the exact clause(s) for justification.

Return the result as:
{{
  "Decision": "Approved/Rejected",
  "Amount": "If applicable",
  "Justification": "Text referencing the clause"
}}
"""

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
        "pad_token_id": phi_tokenizer.eos_token_id,
    }

    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

# Entry point
if __name__ == "__main__":
    user_query = "46M, knee surgery, Pune, 3-month policy"
    response = generate_structured_response(user_query)
    print(response)