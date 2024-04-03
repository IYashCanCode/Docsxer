import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredFileLoader,DirectoryLoader

path = 'D:\HtmlFormfile\XamBuddy\chatbuddy\cbd\content'

class fileUploaderReader:

    def uploaderNread(self):
        uploaded_file = st.file_uploader("Choose a file",accept_multiple_files=True)
        if uploaded_file is not None:
            for uFile in uploaded_file:
                with open(os.path.join(path,uFile.name),"wb") as file:
                    file.write(uFile.getvalue())
        pptContent = DirectoryLoader('D:\HtmlFormfile\XamBuddy\chatbuddy\cbd\content')
        pptContent = pptContent.load()
        return pptContent

