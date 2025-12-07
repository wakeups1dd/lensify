# Lensify — Upgraded Local Semantic Search Engine 

This upgraded Lensify prototype includes the following accuracy improvements:
- FAISS HNSW index for fast, high-recall retrieval
- Support for high-quality embedding models (BAAI/bge-*/sentence-transformers)
- Cross-Encoder re-ranker (ms-marco or similar) for final ranking
- Sentence-boundary chunking with overlap
- Text normalization and query expansion
- New CLI commands: doctor, stats, rebuild, query, export

IMPORTANT:
- Large models (BGE large, cross-encoders) are optional but recommended.
- To get best results install dependencies:
  pip install -r requirements.txt

Quickstart:
1. Install dependencies (recommended): pip install -r requirements.txt
2. Rebuild index: lensify rebuild /path/to/docs
3. Query: lensify query /path/to/docs "payment risk simulation" --k 5
4. Use doctor to check environment: lensify doctor

Notes:
- This is a prototype. For production usage ensure you have enough RAM and disk for large models.
