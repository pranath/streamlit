import sys
import validators, streamlit as st
sys.path.append('..')
from langchain import HuggingFaceHub
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from streamlit_js_eval import streamlit_js_eval
from urllib.parse import urlparse
import textwrap
import common

st.header("Chat with a YouTube Video")
# Create URL input field
url = st.text_input("YouTube URL", label_visibility="collapsed", placeholder="Enter YouTube URL")

if url:
    domain = urlparse(url).netloc
    # Validate inputs
    if not validators.url(url) or domain != 'www.youtube.com':
        st.error("Please enter a valid YouTube URL.")
    else:
        # Load URL data
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        transcript = loader.load()
        st.write('**"' + str(transcript[0].metadata['title']) + '"**')
        # Split into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0, length_function=len)
        chunks = text_splitter.split_text(transcript[0].page_content)  
        # Create embeddings
        embeddings = HuggingFaceHubEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        # Show user input
        user_question = st.text_input("Ask a question about this video:")
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
        if st.button("Chat with a new YouTube video"):
          streamlit_js_eval(js_expressions="parent.window.location.reload()")
    