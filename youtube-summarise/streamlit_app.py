import sys
import validators, streamlit as st
sys.path.append('..')
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap
from urllib.parse import urlparse
import common

# Init Streamlit web app
st.header('Summarise YouTube Video')
# Create URL input field
url = st.text_input("YouTube URL", label_visibility="collapsed", placeholder="Enter YouTube URL")

if url:
    domain = urlparse(url).netloc
    # Validate inputs
    if not validators.url(url) or domain != 'www.youtube.com':
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                transcript = loader.load()
                st.write('**"' + str(transcript[0].metadata['title']) + '"**')
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
                docs = text_splitter.split_documents(transcript)
                # Create model 
                llm = HuggingFaceHub(repo_id=common.repo_id, model_kwargs=common.model_kwargs)
                # Run summarise chain on documents and output
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                output_summary = chain.run(docs)
                wrapped_text = textwrap.fill(output_summary, width=100, break_long_words=False, replace_whitespace=False)
                st.success(wrapped_text)
        except Exception as e:
            st.error(e)

