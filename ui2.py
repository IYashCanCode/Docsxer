from chatui import fileUploaderReader
from embeddings import splitting
import streamlit as st


returnedContent = None
doc_split = None

with st.sidebar:
    st.header("upload you documnets here")
    uploaded_file = st.file_uploader("Choose a file",accept_multiple_files=True)

    if st.button("Process"):
        with st.spinner("Processing"):
            app = fileUploaderReader()
            returnedContent = app.uploaderNread(uploaded_file)
        
        doc_split = splitting()
        doc_split = doc_split.text_split(returnedContent)
#
        embeddings = splitting().gemini_embeddings(doc_split)

st.divider()
st.write(returnedContent)
st.divider()