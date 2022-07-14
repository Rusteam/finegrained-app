FROM python:3.9.6-slim

COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

WORKDIR /app
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]