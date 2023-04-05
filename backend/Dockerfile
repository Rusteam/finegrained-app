FROM python:3.10-slim-buster

COPY backend/requirements.txt .
RUN pip install --no-cache -r requirements.txt

WORKDIR /app
COPY backend ./backend

CMD ["uvicorn", "backend.main:app", "--host",  "0.0.0.0", "--port", "8100"]