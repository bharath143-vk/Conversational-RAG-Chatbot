from loaders.loader import load_documents
from splitters.splitter import split_docs
from vectorstore.vectorstore import build_vector_store
from RAG.memory import create_memory
from RAG.pipeline import create_rag_prompt
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()

SMALL_TALK = [
    "hi", "hello", "hey", "how are you",
    "who are you", "what can you do",
    "good morning", "good evening"
    ]

def is_small_talk(query: str) -> bool:
    return query.lower().strip() in SMALL_TALK




def run_chatbot(text:str,pdf_path:list[str]):
    #load data
    docs=load_documents(text,pdf_path)

    #2 chunking
    chunks=split_docs(docs)

    #3)vectordb
    vector_store=build_vector_store(chunks)

    retriever=vector_store.as_retriever(search_kwargs={"k":4})

    prompt=create_rag_prompt()

    memory=[]

    llm=ChatGroq(model="llama-3.1-8b-instant")

    chain=prompt | llm






    while True:
        query=input("user ",)
        if query.lower()=="exit":
            print("Bye 👋")
            break
        memory.append(HumanMessage(query))

        if is_small_talk(query):
            result = llm.invoke(query)
            print("AI:", result.content)
            memory.append(AIMessage(content=result.content))
            continue


        docs=retriever.invoke(query)
        context="\n\n".join(doc.page_content for doc in docs)
       
        result=chain.invoke({"question":query,"chat_history":memory,"context":context})

        memory.append(result)
        print("AI:", result.content, "\n")
        # print(memory,"\n")
    

run_chatbot("virat kohli is a great leader",["D:/GEN AI Langchain/demo files/dl-curriculum.pdf"])