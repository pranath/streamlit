import validators, streamlit as st
import tempfile
from langchain import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
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

# Init Streamlit web app
st.header('Summarise Document')
# Create document input field
source_doc = st.file_uploader("Upload PDF Document", type="pdf")

# If 'Summarise' button is clicked
if st.button("Summarise"):
    # Validate inputs
    if not source_doc:
        st.error("Please upload a PDF document.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(source_doc.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load_and_split()
                os.remove(tmp_file.name)
                # Create model 
                llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
                # Run summarise chain on documents and output
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = chain.run(pages)
                st.success(summary)
        except Exception as e:
            st.error(e)

