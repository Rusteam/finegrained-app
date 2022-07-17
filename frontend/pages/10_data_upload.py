"""A page to upload data.
"""
import numpy as np
from typing import List

import pandas as pd
import streamlit as st

from utils.client import Client
from utils.st import set_page


def _index_text_embeddings(
    data: pd.DataFrame, data_field: str, model_name: str, data_name: str
):
    texts = data[data_field].tolist()
    prediction = client.predict(model_name, {"text": texts})
    resp = client.index(
        data_name, prediction["embeddings"], data.fillna("").to_dict("records")
    )
    return resp


client = Client()

set_page(
    "Finegrained.AI - Data upload",
    "Upload data, pass through a model and index to a database.",
)

file = st.file_uploader(label="Upload your file")

if file:
    if not file.name.endswith(".csv"):
        st.error("File must have .csv extension")
    else:
        data = pd.read_csv(file)

        with st.form("index_data"):
            data_field = st.selectbox(
                label="Select a data field", options=data.columns
            )
            model = st.selectbox(
                label="Select a model", options=client.list_models()
            )
            data_name = st.text_input(
                label="Enter a name for the data", max_chars=100
            )
            btn = st.form_submit_button("Submit")

        if btn:
            with st.spinner("Indexing data"):
                resp = _index_text_embeddings(data, data_field, model, data_name)
                st.write(resp)
