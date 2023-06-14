import sys
import validators, streamlit as st
sys.path.append('..')
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from streamlit_js_eval import streamlit_js_eval
import common

# Set default model
repo_id = "IAJw/declare-flan-alpaca-large-18378" 
model_kwargs = {"temperature":0, "max_length":512}

st.header("Chat with a Web Page")
# Create URL input field
url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter web page URL")
    
if url:
  # Load URL data
  loader = UnstructuredURLLoader(urls=[url])
  data = loader.load()
  data_str = data[0].page_content.replace('\n', '')
  # Split into chunks
  text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0, length_function=len)
  chunks = text_splitter.split_text(data_str)  
  # Create embeddings
  embeddings = HuggingFaceHubEmbeddings()
  knowledge_base = FAISS.from_texts(chunks, embeddings)
  # Show user input
  user_question = st.text_input("Ask a question about this web page:")
  if user_question:
    try:
      with st.spinner("Please wait..."):
        docs = knowledge_base.similarity_search(user_question)
        llm = HuggingFaceHub(repo_id=common.repo_id, model_kwargs=common.model_kwargs)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.success(response)
    except Exception as e:
      st.error(e)
  if st.button("Chat with a new Web Page"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    