from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain,LLMChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
import os



class preprocessing:


    def text_split(self,document):

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap = 100)

        splitted_content = splitter.split_documents(document)

        return splitted_content
    
    def gemini_embeddings(self, splitted_docs):
        
        os.environ['GOOGLE_API_KEY'] = "AIzaSyBfRzQtaS_d6pDoAx-eU-IqCrfQUBr0_Jo"

        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

        createdEmbeddings = FAISS.from_documents(splitted_docs,embedding=embeddings)

        createdEmbeddings.save_local("vector_embeddings")

