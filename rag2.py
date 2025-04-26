import os 
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
from langchain.prompts import ChatPromptTemplate
import warnings
load_dotenv()

warnings.filterwarnings("ignore")

api_key=os.getenv("GROQ_API_KEY")
#streamlit setup
st.title("RAG Chatbot")
st.subheader("Document: attention-is-all-you-need-Paper")

pdf_path="NIPS-2017-attention-is-all-you-need-Paper.pdf"
faiss_index_Attention="attention_FAISS"


if not os.path.exists(faiss_index_Attention):
    with st.spinner("loader and processing PDF...."):
        loader=PyPDFLoader(pdf_path)
        data=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
        splitter_data=text_splitter.split_documents(data)

        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectore_store=FAISS.from_documents(documents=splitter_data,embedding=embedding)

        vectore_store.save_local(faiss_index_Attention)

else:
    #load and use
    with st.spinner("Feaching the data from VectoreStore"):
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectore_store=FAISS.load_local(faiss_index_Attention,embeddings=embedding,allow_dangerous_deserialization=True)



#setup the rtriver and chain 

retriever=vectore_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})


#pre-trained model

LLM=ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

prompt=ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.
    if non relavent question ignore and says only ansewer this pdf or documents only and ask again ask any question related to thid document please

Context: {context}

Question: {question}"""


)

def format_doc(relevant_docs):
    return  " ".join(doc.page_content for doc in relevant_docs)


chain=(
    {"context":retriever|format_doc,"question":RunnablePassthrough()}
    | prompt
    |LLM
    |StrOutputParser()
)

question=st.chat_input("Ask Your Question")
if question:
    with st.spinner("Retrieving........."):
        response=chain.invoke(question)
        st.write(question)
        st.write(f'AI:{response}')

