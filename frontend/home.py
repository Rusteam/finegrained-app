"""Home page for streamlit app
"""
import streamlit as st

from utils.client import Client

client = Client()

st.set_page_config(page_title="finegrained - home")

st.title("Finegrained.AI app")

st.text("Select a page on the left sidebar to interact with models.")

st.header("Models")
st.write(client.list_models())

st.header("Vector data")
st.write(client.list_embeddings())

st.header("Pipelines")
st.write(client.list_pipelines())
