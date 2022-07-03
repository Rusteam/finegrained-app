FROM python:3.9.6-slim

COPY frontend/requirements.txt .
RUN pip install --no-cache -r requirements.txt

WORKDIR /app
COPY frontend ./frontend
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]