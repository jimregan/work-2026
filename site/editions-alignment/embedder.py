import numpy as np
from sentence_transformers import SentenceTransformer


def embed(sentences: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)


def similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    return emb_a @ emb_b.T
