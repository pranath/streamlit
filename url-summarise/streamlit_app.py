import validators, streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
import os
from dotenv import load_dotenv, find_dotenv

# Load Huggingface API Token
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# Streamlit app
st.subheader('Summarize URL')

url = st.text_input("URL", label_visibility="collapsed")

# If 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = UnstructuredURLLoader(urls=[url])
                data = loader.load()
                llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
                prompt_template = """Write a summary of the following in 200-250 words split into 2 paragraphs:

                    {text}

                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(data)
                st.success(summary)
        except Exception as e:
            st.error("I'm sorry something went wrong! Please try again with a different URL")
