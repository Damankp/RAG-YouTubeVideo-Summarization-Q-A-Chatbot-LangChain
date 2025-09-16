from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def create_chain(chatmodel, retriever):
    """Create the retrieval + generation chain."""

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

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    return parallel_chain | prompt | chatmodel | parser