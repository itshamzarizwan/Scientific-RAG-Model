import numpy as np

def retrieve_top_k(query, index, chunks, embedding_model, k=5):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [{
        "text": chunks[i]["text"],
        "paper_id": chunks[i]["paper_id"],
        "section": chunks[i]["section"],
        "authors": chunks[i]["authors"]
    } for i in I[0]]
