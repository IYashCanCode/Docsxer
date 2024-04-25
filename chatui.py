import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredFileLoader,DirectoryLoader

path = '.\content'
os.makedirs(path,exist_ok=True)

class fileUploaderReader:

    def uploaderNread(self,uploaded_file):
        if uploaded_file is not None:
            for uFile in uploaded_file:
                with open(os.path.join(path,uFile.name),"wb") as file:
                    file.write(uFile.getvalue())
            pptContent = DirectoryLoader('.\content',loader_cls=UnstructuredFileLoader,loader_kwargs={'mode':"single","strategy":"fast"})
            pptContent = pptContent.load()
            return pptContent
