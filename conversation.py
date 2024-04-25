from langchain_community.document_loaders import UnstructuredFileLoader,DirectoryLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain,ConversationalRetrievalChain,RetrievalQA
from langchain.memory import VectorStoreRetrieverMemory,ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

os.environ['GOOGLE_API_KEY'] = "AIzaSyBfRzQtaS_d6pDoAx-eU-IqCrfQUBr0_Jo"

documents = WebBaseLoader('https://medium.com/@codethulo/most-important-interview-questions-of-transformer-5308764493b5')

data = documents.load()

splitter = RecursiveCharacterTextSplitter(chunk_overlap = 50, chunk_size = 200)

docs_splitted = splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

docs_embeddings = FAISS.from_documents(docs_splitted,embeddings)

docs_embeddings.save_local('docs_embeddings',index_name = 'created_embeddings')

new_docs_embeddings = FAISS.load_local('docs_embeddings',embeddings=embeddings,index_name="created_embeddings",allow_dangerous_deserialization=True)

query = "Retrieve basic information about transformer."

retriever = new_docs_embeddings.as_retriever(search_type = 'mmr', search_kwargs = {'k':10})

template = """ You are an AI bot which helps students with Examination questions. Students asks you questions.
You have to answer the questions. 


Question : {question}
Answer: 


"""

prompt = PromptTemplate(template=template,input_variables=['question'])

model = ChatGoogleGenerativeAI(model='gemini-pro')


chain = RetrievalQA.from_chain_type(llm = model, retriever=retriever)

response = chain(prompt)

print(response)

