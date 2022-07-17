"""Chatbot page to send messages and get replies.
"""

import streamlit as st

from utils.client import Client
from utils.st import display_labels, set_page

client = Client()

set_page("Finegrained.AI - Text matching",
         "Find top K most similar text samples.")


def _get_chatbot_answer(text_input, model_name, data_name, top_k):
    top_sim = client.predict_and_search(
        model_name, data_name, data={"text": text_input}, top_k=top_k
    )

    return top_sim[0]


left, right = st.columns([1, 1])

with left:
    st.markdown('### Configure')
    with st.form("send_message"):
        vectors = st.selectbox("Select vector data", client.list_embeddings())
        model = st.selectbox("Select a model", client.list_models())
        top_k = st.number_input(label="No. of top matches", min_value=1,
                                max_value=10, value=5)
        text = st.text_area(label="Enter a message", max_chars=300)
        btn = st.form_submit_button(label="Send")

with right:
    if btn:
        st.markdown('### Results')
        if len(text) <= 1:
            st.error("Message should be at least 2 chars")
        else:
            top_sim = _get_chatbot_answer(text, model, vectors, top_k)
            display_labels(top_sim)
