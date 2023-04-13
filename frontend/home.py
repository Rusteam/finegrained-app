"""Home page for streamlit app
"""
import streamlit as st

from utils.client import Client
from utils.st import handle_error

client = Client()

st.set_page_config(page_title="finegrained - home")

st.title("Finegrained.AI app")

st.text("Select a page on the left sidebar to interact with models.")


@handle_error
def show_section(title: str, callback: callable):
    st.subheader(title)
    st.write(callback())


show_section("Models", client.list_models)
show_section("Vectors", client.list_embeddings)
show_section("Pipelines", client.list_pipelines)
