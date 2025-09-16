import streamlit as st
from transcript_utils import fetch_youtube_transcript
from model_utils import init_chatmodel, init_embedding_model
from indexing_retriever_utils import build_vectorstore, build_retriever
from chain_utils import create_chain

def main():
    st.title("YouTube Video Q&A Chatbot")

    video_id = st.text_input("Enter YouTube video ID (e.g., UkN7j3To-uQ):", "UkN7j3To-uQ")

    if st.button("Load Video"):
        with st.spinner("Fetching transcript and setting up..."):
            transcript = fetch_youtube_transcript(video_id)
            chatmodel = init_chatmodel()
            embedding = init_embedding_model()
            vectorstore = build_vectorstore(transcript, embedding)
            retriever = build_retriever(vectorstore, chatmodel)
            st.session_state.chain = create_chain(chatmodel, retriever)

        st.success("Ready! Ask your questions below 👇")

    if "chain" in st.session_state:
        question = st.text_input("Ask a question about the video:")
        if question:
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke(question)
            st.markdown(result)

if __name__ == "__main__":
    main()
