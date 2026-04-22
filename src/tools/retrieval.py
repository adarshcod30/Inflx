import os
from rank_bm25 import BM25Okapi
import re

def load_knowledge_base(file_path="data/knowledge_base.md"):
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Simple chunking by sections (splitting by ##)
    chunks = re.split(r'\n(?=## )', content)
    return [c.strip() for c in chunks if c.strip()]

def retrieve_knowledge(query: str, top_k=2):
    chunks = load_knowledge_base()
    if not chunks:
        return "Knowledge base not found."
    
    tokenized_corpus = [doc.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    top_chunks = bm25.get_top_n(tokenized_query, chunks, n=top_k)
    
    return "\n\n---\n\n".join(top_chunks)
