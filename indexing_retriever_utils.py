# retriever_utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
import config

def build_vectorstore(transcript: str, embedding_model):
    """Split transcript and build FAISS vectorstore."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS
    )
    chunks = text_splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

def build_retriever(vectorstore, chatmodel):
    """Create a multi-query retriever with MMR."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.NUM_RESULTS}
    )
    return MultiQueryRetriever.from_llm(retriever=retriever, llm=chatmodel)
