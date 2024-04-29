import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import WebBaseLoader,UnstructuredFileLoader,DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

os.environ['GOOGLE_API_KEY'] = "AIzaSyBfRzQtaS_d6pDoAx-eU-IqCrfQUBr0_Jo"

def get_pdf_text():
    data = DirectoryLoader('.\content',loader_cls=UnstructuredFileLoader,loader_kwargs={'mode':"single","strategy":"fast"})

    text = data.load()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(csv_chunking):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_documents(csv_chunking,embeddings)

    vectorstore.save_local("./lizmotors/vector_embeddings", index_name="base_and_adjacent")
    vectorstore  =   FAISS.load_local("lizmotors/vector_embeddings", embeddings, index_name="base_and_adjacent",allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type = 'mmr',search_kwargs={'k':5})
    return retriever


def get_conversation_chain(retriever,query):
    model = ChatGoogleGenerativeAI(model='gemini-pro')

    text = """ You are an AI assistant who recieves blog links from the user. 
               You have to answer the questions based on the blog link.
               In answer no where use ** symbol.

               If the questions is not related to the answer you created, answer them with a polite sorry and tell user to enter questions related to the topic.

    User : {query}
    Assistant :
    """

    prompt = PromptTemplate(input_variables=["query"],template = text)

    

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=model,retriever = retriever)

    chat_history = []
    result = conversation_chain.invoke({"question": query,
                  "chat_history":chat_history})
    return result


data = get_pdf_text()

print(data)

chunks = get_text_chunks(data)

vectorstore = get_vectorstore(chunks)

chain = get_conversation_chain(vectorstore,"What are the technical skills of Yash Kumar")



print(chain['answer'])