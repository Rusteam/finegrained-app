from datetime import datetime as dt

import streamlit as st
import pandas as pd

from utils.client import Client

client = Client()

st.title("Finegrained.AI - Model registry")
st.subheader("List models available in the model registry.")

models = client.list_registry_models()

_display_date = lambda x: dt.fromtimestamp(x/1000).strftime("%Y-%m-%d")

for m in models:
    with st.expander(m['name']):
        st.text(f"Description: {m['description']}")
        change_date = _display_date(m["last_updated_timestamp"])
        st.markdown(f"Last updated: [{change_date}]({m['link']})")

        versions = client.list_registry_model_versions(m["name"])
        st.markdown("##### Versions:")
        for v in versions:
            left, mid, right = st.columns(3)
            version_change = _display_date(v["last_updated_timestamp"])
            left.markdown(f"[Version: {v['version']}]({v['link']})")
            mid.text(f"Current stage: {v['current_stage']}")
            right.text(f"Last updated: {version_change}")

# TODO finish
