FROM python:3.9.6-slim

COPY backend/requirements.txt .
RUN pip install --no-cache -r requirements.txt

WORKDIR /app
COPY backend ./backend
COPY main.py .

CMD ["uvicorn", "main:app", "--host",  "0.0.0.0", "--port", "8100"]