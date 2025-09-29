import numpy as np
from deepface import DeepFace

class Embedder:
    def __init__(self):
        pass

    def get_embedding(self, face_img):
        try:
            emb_list = DeepFace.represent(face_img, enforce_detection=False)
            if emb_list and len(emb_list) > 0:
                return np.array(emb_list[0]["embedding"], dtype=float)
            return None
        except Exception as e:
            print("Embedding failed:", e)
            return None

    def cos_sim(self, emb1, emb2):
        """Compute cosine similarity between two embeddings"""
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        if emb1 is None or emb2 is None:
            return -1
        # Cosine similarity formula
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(sim)
