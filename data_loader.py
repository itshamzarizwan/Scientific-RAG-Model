import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_chunks(index_path="faiss_index.index", chunks_path="chunks.json.npy"):
    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedding_model
