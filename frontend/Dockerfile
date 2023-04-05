FROM python:3.10-slim-buster

COPY frontend/requirements.txt .
RUN pip install --no-cache -r requirements.txt

WORKDIR /app
COPY frontend ./frontend

EXPOSE 8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=False
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=True
ENV STREAMLIT_LOGGER_LOG_LEVEL=warning
ENV STREAMLIT_GLOBAL_DEVELOPMENT_MODE=False

CMD ["streamlit", "run", "frontend/home.py"]