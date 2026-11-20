import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    texts = [
        "nulls appearing in a feature",
        "Missing values increased in production data",
        "Recall dropped after the decision threshold was raised",
        "numbers suddenly 100 times larger",
        "A unit or scale changes, for example rupees to paise",
    ]
    vectors = model.encode(texts)
    print("shape:", vectors.shape)
    print("first 5 numbers of vector 0:", vectors[0][:5])
    print("0 vs 1:", round(float(cosine(vectors[0], vectors[1])), 3))
    print("0 vs 2:", round(float(cosine(vectors[0], vectors[2])), 3))
    print("3 vs 4:", round(float(cosine(vectors[3], vectors[4])), 3))
    print("3 vs 1:", round(float(cosine(vectors[3], vectors[1])), 3))