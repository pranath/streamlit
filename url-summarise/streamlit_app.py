import validators, streamlit as st
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
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

# Streamlit app
st.header('Summarise Web Page')
# Set model
repo_id = "IAJw/declare-flan-alpaca-large-18378" 
model_kwargs = {"temperature":0, "max_length":512}
# Create URL input field
url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter web page URL")

if url:
    try:
        with st.spinner("Please wait..."):
            # Load URL data
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            # Create model 
            llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
            # Run summarise chain on documents and output
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(data)
            st.success(summary)
    except Exception as e:
        st.error(e)