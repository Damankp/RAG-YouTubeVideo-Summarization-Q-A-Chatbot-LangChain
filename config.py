# HuggingFace models for text generation and embeddings
LLM_REPO_ID = "deepseek-ai/DeepSeek-V3.1"
EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", " ", ""]

# Retriever
NUM_RESULTS = 5