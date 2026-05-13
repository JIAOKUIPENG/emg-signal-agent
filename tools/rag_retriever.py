import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from knowledge_base.emg_knowledge import EMG_KNOWLEDGE

class SimpleRAG:
    def __init__(self):
        self.docs = EMG_KNOWLEDGE
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        self.doc_vectors = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query: str, top_k: int = 2) -> str:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [self.docs[i] for i in top_idx if scores[i] > 0]
        if not results:
            return "未找到相关知识"
        return "\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))

rag = SimpleRAG()

def search_knowledge(query: str) -> str:
    return rag.retrieve(query)
