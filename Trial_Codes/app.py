# Q&A with Youtube Transcript using LangChain and HuggingFace

# Starting with a MVP (Minimum Viable Product) - Essential features only [Enhancemebts to be added later]

# Steps to be followed:
# 1. Importing the required libraries
# 2. Writing a function to fetch transcript from a given Youtube URL
# 3. Setting up the language model and embedding model using HuggingFace API
# 4. Text Splitting - Splitting the transcript into manageable chunks
# 5. Creating a vector store to hold the embeddings
# 6. Setting up the Retriever
# 7. Creating a chain
# 8. Creatin an UI using Streamlit

# 1. Importing the required libraries

from youtube_transcript_api import YouTubeTranscriptApi


# 2. Writing a function to fetch transcript from a given Youtube URL

def fetch_youtube_transcript(video_id: str) -> str:

    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id,
                  languages=['en'])  # Specify the language of the transcript, default is English. Should try to translate it to English at a later stage if transcript is not available in English

    # Converting the fetched transcript into a single string
    transcript = " ".join([snippet.text for snippet in fetched_transcript])
    return transcript

video_id = "UkN7j3To-uQ" # Replace with your YouTube video ID, Write a function later to extract the video ID from the URL

transcript = fetch_youtube_transcript(video_id)

# Checking the length of the transcript
# print(f"Length of the transcript: {len(transcript)} characters")

# # Printing the first and last 100 characters of the transcript
# print(f"First 100 characters of the transcript: {transcript[:100]}")
# print(f"Last 100 characters of the transcript: {transcript[-100:]}")

# 3. Setting up the language model and embedding model using HuggingFace API

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

# Setting up the language model
llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.1',
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Checking the model
# result = model.invoke("What is the capital of India")

# print(result.content)

# Setting up the embedding model
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    task="feature-extraction"
)
# # Checking the embedding model
# embedding = embedding_model.embed_query("Hello, how are you?")
# print(f"Embedding length: {len(embedding)}")
# print(f"Embedding: {embedding}")

# 4. Text Splitting - Splitting the transcript into manageable chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""])

chunks = text_splitter.create_documents([transcript])

# print(len(texts))
# print(f"Number of chunks: {len(chunks)}")
# print(f"First few chunks: {chunks[0]}")
# print(f"{chunks[1]}")
# print(f"{chunks[2]}")

# 5. Creating a vector store to hold the embeddings
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)
print(f"Number of vectors in the vector store: {vectorstore.index.ntotal}")

# First vector shape
# print(f"First vector shape: {vectorstore.index.reconstruct(0).shape}")
# # First vector
# print(f"First vector: {vectorstore.index.reconstruct(0)}")

# 6. Setting up the Retriever using multi-query retrieval with Maximal Marginal Relevance (MMR) as base-retriever

from langchain.retrievers import MultiQueryRetriever

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})


multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model)

# Checking the retriever on a sample query
multiquery_retriever.invoke("Most important points of the video?")

# 7. Creating a chain by combining the retriever and the language model

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a knowledgeable assistant answering questions about a YouTube video.

Use ONLY the information from the transcript context below.
- If the context does not contain the answer, reply with: "I don't know based on the transcript."
- Do NOT use outside knowledge.
- Prefer concise, factual answers.
- If multiple relevant points exist, summarize them in bullet points.
- If transcript timestamps are available in the context, include them in your answer.

---
Transcript context:
{context}
---
Question: {question}

""",
    input_variables=['context', 'question']
)

question = "is the topic of attention discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)
retrieved_docs

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': multiquery_retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

# result = main_chain.invoke('Can you summarize the video for a 15 year old?')

from IPython.display import Markdown
# Markdown(result)

# 8. Creatin an UI using Streamlit to interact with the chatbot

import streamlit as st

st.title("YouTube Video Q&A Chatbot")
question = st.text_input("Ask a question about the video:")
if question:
    result = main_chain.invoke(question)
    st.markdown(result)