import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap
from urllib.parse import urlparse
import validators, streamlit as st

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

# Init Streamlit web app
st.subheader('Summarise YouTube Video')
url = st.text_input("YouTube URL", label_visibility="collapsed")

# If 'Summarise' button is clicked
if st.button("Summarise"):
    domain = urlparse(url).netloc
    # Validate inputs
    if not validators.url(url) or domain != 'www.youtube.com':
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = YoutubeLoader.from_youtube_url(url)
                transcript = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
                docs = text_splitter.split_documents(transcript)
                # Create model 
                # Falcon is a text-generation model https://huggingface.co/tasks/text-generation
                repo_id = "tiiuae/falcon-7b-instruct"  
                llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500})
                # Run summarise chain on documents and output
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                output_summary = chain.run(docs)
                wrapped_text = textwrap.fill(output_summary, width=100, break_long_words=False, replace_whitespace=False)
                st.success(wrapped_text)
        except Exception as e:
            st.error(e)

