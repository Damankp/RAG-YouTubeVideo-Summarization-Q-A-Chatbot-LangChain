# 🎥 YouTube Video Q&A Chatbot  

An interactive Streamlit app that lets you **summarize YouTube videos** and **ask questions directly from the transcript** using **LangChain** and **HuggingFace models**.  

---

## ✨ Features  

- ✅ **Streamlit Web App**: Interactive and user-friendly interface for uploading a YouTube video id and asking questions.

- ✅ **YouTube Transcript Retrieval**: Extracts video transcripts automatically from YouTube using youtube API.

- ✅ **Conversational Q&A**: Users can ask multiple follow-up questions about the video.

- ✅ **LangChain Integration**: Manages the pipeline of document loading, text splitting, embeddings, and retrieval.

   - **Recursive Character Text Splitter** for improved chunk splitting with respect to structure of the provided text, instead of splitting after fixed length.
   - Use of **Maximal Marginal Retrival** technique along with **Multi-query retrieval** to enhance the context text identification.

- ✅ **Retrieval-Augmented Generation (RAG)**: Uses FAISS vector database to store video transcript embeddings for efficient search.

- ✅ **Local Embedding**: Supports local models (e.g., HuggingFace embeddings) to overcome **timeout errors**.

- ✅ **LLM-Powered Responses**: Uses Chat models (configurable) for context-aware, high-quality answers.

---

## 📸 Demo  
[![Demo](assets\demo.gif)]

The demo video is 1 minute 42 seconds

---

## ⚡ Installation  

```bash
# Clone the repository
git clone https://github.com/your-username/youtube-qa-chatbot.git
cd youtube-qa-chatbot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage  

1. Start the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

2. Paste a **YouTube video URL** in the input box.  
3. The app will:  
   - Fetch transcript  
   - Generate a **summary** automatically  
   - Allow you to **ask questions interactively**  

---

## 📂 Project Structure  

```
youtube-qa-chatbot/
│
├── ui.py                         # Streamlit UI  
├── transcript_utils.py           # Fetch transcript & extract video ID  
├── model_utils.py                # Chatmodel & embeddings model setup  
├── chain_utils.py                # Chains (retrieval + Q&A)
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration details like Models used
├── indexing_retriever_utils.py   # building vectorstore and retriever functions
└── README.md                     # Project documentation
```

---

## 🚀 Future Improvements
- 🔹 Support for multi-language transcripts, translations
- 🔹 Export Q&A session
- 🔹 Multi-modal support - Adding multiple files - PDFs, videos, links etc. and question from all modalities
