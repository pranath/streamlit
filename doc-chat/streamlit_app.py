import validators, streamlit as st
from pypdf import PdfReader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from streamlit_js_eval import streamlit_js_eval
from httpimport import *
from dotenv import load_dotenv, find_dotenv
import os

# Set Reference Links
st.set_page_config(
    initial_sidebar_state="expanded",
    menu_items={
        'About': "https://livingdatalab.com/",
        'Get Help': 'https://livingdatalab.com/about.html',
        'Report a bug': "https://livingdatalab.com/about.html",
    }
)

# Load API token
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Set default model
repo_id = "IAJw/declare-flan-alpaca-large-18378" 
model_kwargs = {"temperature":0, "max_length":512}

#-------------------------------------------------------------------

st.header("Chat with a Document")
# Create document input field
pdf = st.file_uploader("Upload PDF Document", type="pdf")
    
# Extract content
if pdf is not None:
  pdf_reader = PdfReader(pdf)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()
  # Split into chunks
  text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
  chunks = text_splitter.split_text(text)  
  # Create embeddingsc
  embeddings = HuggingFaceHubEmbeddings()
  knowledge_base = FAISS.from_texts(chunks, embeddings)
  # Show user input
  user_question = st.text_input("Ask a question about your document:")
  if user_question:
    try:
      with st.spinner("Please wait..."):
        docs = knowledge_base.similarity_search(user_question)
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.success(response)
    except Exception as e:
      st.error(e)
  if st.button("Chat with a new Document"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    