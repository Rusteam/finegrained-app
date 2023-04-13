from datetime import datetime as dt

import streamlit as st
import pandas as pd

from utils.client import Client
from utils.st import handle_error

client = Client()

st.title("Finegrained.AI - Model registry")
st.text("List models available in the model registry.")


_display_date = lambda x: dt.fromtimestamp(x/1000).strftime("%Y-%m-%d")


@handle_error
def _list_registry_models():
    models = client.list_registry_models()
    return models


@handle_error
def _deploy_model(model_name, version):
    global client
    with st.spinner("Deploying model..."):
        client.deploy_model_version_from_registry(model_name, version)
        st.success(f"Deployed {model_name} version {version}")


@handle_error
def _delete_model(model_name, version):
    global client
    with st.spinner("Deleting model..."):
        client.delete_model_version(model_name, version)
        st.success(f"Deleted {model_name} version {version}")


models = _list_registry_models()
if models:
    st.subheader("Registry models:")
    for i, m in enumerate(models):
        st.markdown("---")
        change_date = _display_date(m["last_updated_timestamp"])
        st.markdown(f"##### {i+1}. {m['name']} ({change_date})")
        st.markdown(f"{m['description']}")
        with st.expander("versions"):

            versions = handle_error(client.list_registry_model_versions)((m["name"]))
            for v in versions:
                st.markdown("---")

                left, right = st.columns([5, 2])

                version_change = _display_date(v["last_updated_timestamp"])
                left.markdown(f"[Version: {v['version']}]({v['link']})")
                left.text(f"Current stage: {v['current_stage']}")
                left.text(f"Last updated: {version_change}")

                deploy_clicked = right.button("Deploy", key=f"{m['name']}-{v['version']}",
                                              type="secondary",
                                              use_container_width=True,
                                              on_click=_deploy_model,
                                              args=(m['name'], v['version']))
                delete_clicked = right.button("Delete", key=f"{m['name']}-{v['version']}-delete",
                                              on_click=_delete_model,
                                              args=(m['name'], v['version']),
                             type="primary", use_container_width=True)

                # if deploy_clicked:
                #     st.success(f"Deployed {m['name']} version {v['version']}")
                # if delete_clicked:
                #     st.success(f"Deleted {m['name']} version {v['version']}")
