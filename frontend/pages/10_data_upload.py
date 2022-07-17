"""A page to upload data.
"""
import numpy as np
from typing import List

import pandas as pd
import streamlit as st

from utils.client import Client


def _index_text_embeddings(data: pd.DataFrame, data_field: str,
                           model_name: str, data_name: str):
    texts = data[data_field].tolist()
    prediction = client.predict(model_name, {"text": texts})
    resp = client.index(data_name, prediction["embeddings"],
                        data.fillna("").to_dict("records"))
    return resp


client = Client()

file = st.file_uploader(label="Upload your file")

with st.sidebar:
    st.write("# Sidebar")
    st.radio("on or off", options=["on", "off"])


if file:
    assert file.name.endswith(".csv"), "File must have .csv extension"
    data = pd.read_csv(file)

    with st.form("index_data"):
        data_field = st.selectbox(label="Select a data field",
                                  options=data.columns)
        model = st.selectbox(label="Select a model",
                             options=client.list_models())
        data_name = st.text_input(label="Enter a name for the data",
                                  max_chars=100)
        btn = st.form_submit_button("Submit")

    if btn:
        with st.spinner("Indexing data"):
            resp = _index_text_embeddings(data, data_field, model, data_name)
            st.write(resp)
