"""A Gradio app for searching similar texts.
"""
from typing import Tuple

import os

import gradio as gr
import numpy as np
import pandas as pd

from .client import Client


client = Client()
DATA_NAME = "retrieval_chatbot"


def _greet(name):
    return "Hello " + name + "!!"


def greet():
    demo = gr.Interface(fn=_greet, inputs="text", outputs="text")
    demo.launch()


def _embed_text(file, text_field, model_name):
    data = pd.read_csv(file.name)
    text = data[text_field].tolist()

    # get embeddings
    # TODO handle batch size in the backend
    batch_size = 8
    embeddings = []
    for i in range(0, len(text), batch_size):
        batch = text[i : i + batch_size]
        predicted = client.predict(model_name, {"text": batch})
        embeddings.append(predicted["embeddings"])
    embeddings = np.vstack(embeddings).tolist()

    # index data
    resp = client.index(DATA_NAME, embeddings, data.to_dict("records"))

    return resp


def _get_chatbot_answer(text_input, history, model_name):
    predictions = client.predict(model_name, {"text": text_input})
    top_sim = client.search_similar(
        DATA_NAME, predictions["embeddings"], top_k=1
    )

    top_match = top_sim[0][0]
    match_score = top_match["similarity"]
    reply = f"> {top_match['question']}  >>> {top_match['answer']} ({match_score:.2f})"
    history.append((text_input, reply))

    return history, ""


def _load_credentials() -> Tuple[str, str]:
    user = os.getenv("GRADIO_USER", "gradio")
    pswd = os.getenv("GRADIO_PASSWORD", "io.grad")
    return user, pswd


def chatbot():
    models = client.list_models()
    fields = ["question", "answer"]

    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Retrieval chatbot")
        gr.Markdown("Upload your data and starting chatting")
        with gr.Tabs():
            with gr.TabItem("Upload data"):
                gr.Markdown(
                    f"Upload a CSV file with the {fields} "
                    "columns, select field and model and submit."
                )

                with gr.Box():
                    file = gr.File(label="CSV file")
                    text_field = gr.Radio(
                        choices=fields,
                        value=fields[0],
                        label="Which field to use",
                    )
                    model_name = gr.Dropdown(
                        choices=models,
                        value=models[0],
                        label="Which model to use",
                    )
                    file_btn = gr.Button("Submit")

                data = gr.JSON(
                    label="My data",
                )

            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(label="Dialog")
                with gr.Row():
                    with gr.Box():
                        text_input = gr.Textbox(
                            lines=2,
                            max_lines=10,
                            placeholder="Enter a message",
                            label="Message",
                        )
                        text_btn = gr.Button("Send")

        file_btn.click(
            _embed_text,
            inputs=[file, text_field, model_name],
            outputs=[data],
        )
        text_btn.click(
            _get_chatbot_answer,
            inputs=[text_input, chatbot, model_name],
            outputs=[chatbot, text_input],
        )

    launch_kwargs = dict(
        debug=True, inbrowser=False, auth=_load_credentials(),
        server_name="0.0.0.0",
    )
    links = demo.launch(**launch_kwargs)
    print(links)
