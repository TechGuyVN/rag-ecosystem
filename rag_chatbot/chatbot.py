"""Simple RAG chatbot using OpenAI and Qdrant."""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant

def answer_question(vector_store: Qdrant, question: str) -> str:
    """Retrieve relevant context and generate an answer.

    Args:
        vector_store: The Qdrant vector store containing document embeddings.
        question: User question in plain text.

    Returns:
        LLM-generated answer grounded in retrieved context.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(
        "Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = ChatOpenAI(temperature=0)
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    return response.content

