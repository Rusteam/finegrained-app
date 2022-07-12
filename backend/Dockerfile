FROM python:3.9.6-slim

COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

WORKDIR /app
COPY . .

CMD ["uvicorn", "backend.main:app", "--host",  "0.0.0.0", "--port", "8100"]