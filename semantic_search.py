import numpy as np
from sentence_transformers import SentenceTransformer

from runbooks import RUNBOOKS
from chunker import make_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = make_chunks(RUNBOOKS)
chunk_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)


def rank_semantic(query, top_k=3):
    q = model.encode([query], normalize_embeddings=True)[0]

    # YOU WRITE (1): one similarity score per chunk.
    # Because the vectors are normalized, cosine similarity is just the dot
    # product. Hint: chunk_vectors @ q
    scores = chunk_vectors @ q

    # YOU WRITE (2): the positions of the top_k highest scores, best first.
    # Hint: np.argsort(scores)[::-1][:top_k]
    top = np.argsort(scores)[::-1][:top_k]

    return [(chunks[i]["chunk_id"], round(float(scores[i]), 3)) for i in top]


if __name__ == "__main__":
    for q in [
        "recall drop threshold change",
        "missing values increased",
        "feature distribution shift psi",
        "column renamed upstream",
        "unit changed from seconds to milliseconds",
        "numbers suddenly 100 times larger",
        "nulls appearing in a feature",
    ]:
        print(q, "->", rank_semantic(q))