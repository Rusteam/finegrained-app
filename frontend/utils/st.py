"""Helper functions to display streamlit objects.
"""
from typing import List

import requests
import streamlit as st


def display_labels(labels: List[dict]):
    """Display data in a dict as a markdown.
    """
    for one in labels:
        messages = [
            f"**{k}:** {v:.2f}" if isinstance(v, float) else f"**{k}:** {v}"
            for k, v in one.items()
        ]
        messages.insert(0, '---')
        st.write("\n\n".join(messages))


def set_page(title: str, description: str):
    st.set_page_config(page_title=title.lower())
    st.title(title)
    st.text(description)


def handle_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Is it running?")
        except requests.exceptions.HTTPError as e:
            st.error(f"Error: {e}")
    return wrapper
