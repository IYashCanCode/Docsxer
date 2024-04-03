from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
import os
from typing import IO, Any, Iterator, List, Optional, Sequence, Tuple, Union
import pptx
from pptx.presentation import Presentation
from pptx.shapes.autoshape import Shape
from langchain.chains import LLMChain
from ChatUI import fileUploaderReader
import warnings
warnings.filterwarnings("ignore")

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
zephyr = HuggingFaceHub(repo_id =  'HuggingFaceH4/zephyr-7b-alpha')


app = fileUploaderReader()
app.uploaderNread()


splitter = RecursiveCharacterTextSplitter(chunk_overlap = 100 , chunk_size = 200)
splitted_docs = splitter.split_documents(app.pptContent)
embedded_docs = FAISS.from_documents(splitted_docs,embedding = embeddings)


