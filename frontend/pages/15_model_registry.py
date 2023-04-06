import streamlit as st
import pandas as pd

from utils.client import Client

client = Client()

st.title("Finegrained.AI - Model registry")
st.subheader("List models available in the model registry.")

models = pd.DataFrame(client.list_registry_models())

st.table(models)

# TODO finish
# TODO remove user and password from the URL
