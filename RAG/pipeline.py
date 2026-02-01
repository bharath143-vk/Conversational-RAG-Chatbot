from langchain_core.prompts import PromptTemplate

def create_rag_prompt():
    template = """
You are a professional AI assistant helping users understand documents.

Rules:
- Use ONLY the provided context to answer.
- Use the conversation history to understand follow-up questions.
- If the answer is not present in the context, say:
  "I don't have enough information in the provided documents."
- Be concise, clear, and factual.
- Do NOT make up information.

Conversation History:
{chat_history}

Context:
{context}

User Question:
{question}

Answer:
"""
    return PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template
    )

def build_conversational_rag(llm,vectorstore,memory):

    retriever=vectorstore.as_retriever(search_kwargs={"k":4})
    

    prompt=create_rag_prompt()

    chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt
        },
        verbose=True
    )

    return chain


