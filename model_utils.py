from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

load_dotenv()  # Load API keys

def init_chatmodel():
    """Initialize the language model."""
    llm = HuggingFaceEndpoint(
        repo_id=config.LLM_REPO_ID,
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)

# def init_embedding_model():
#     """Initialize the embedding model."""
#     return HuggingFaceEndpointEmbeddings(
#         repo_id=config.EMBEDDING_REPO_ID,
#         task="feature-extraction"
#     )

def init_embedding_model():
    """Initialize the embedding model (local)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_REPO_ID
    )