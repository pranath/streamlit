import sys
import validators, streamlit as st
sys.path.append('..')
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
import common

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
            llm = HuggingFaceHub(repo_id=common.repo_id, model_kwargs=common.model_kwargs)
            # Run summarise chain on documents and output
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(data)
            st.success(summary)
    except Exception as e:
        st.error(e)