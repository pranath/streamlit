import validators, streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
import os
from dotenv import load_dotenv, find_dotenv

# Set Reference Links
st.set_page_config(
    initial_sidebar_state="expanded",
    menu_items={
        'About': "https://livingdatalab.com/",
        'Get Help': 'https://livingdatalab.com/about.html',
        'Report a bug': "https://livingdatalab.com/about.html",
    }
)

# Load Huggingface API Token
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# Streamlit app
st.subheader('Summarise URL')

url = st.text_input("URL", label_visibility="collapsed")

# If 'Summarise' button is clicked
if st.button("Summarise"):
    # Validate inputs
    if not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = UnstructuredURLLoader(urls=[url])
                data = loader.load()
                # Create model 
                # flan-alpaca-large is a text2text generation model https://huggingface.co/tasks/text-generation
                repo_id = "declare-lab/flan-alpaca-large" 
                llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.3, "max_length":512})
                # Run summarise chain on documents and output
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = chain.run(data)
                st.success(summary)
        except Exception as e:
            st.error(e)