from dotenv import load_dotenv, find_dotenv
import streamlit as st
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