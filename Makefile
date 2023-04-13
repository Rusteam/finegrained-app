run: up logs

include .env

triton: Triton.dockerfile
	docker-compose build triton

backend:
	poetry export -o backend/requirements.txt --without-hashes
	docker-compose build backend

frontend:
	poetry export -o frontend/requirements.txt --only=frontend --without-hashes
	docker-compose build frontend

build_all: triton backend frontend

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f --tail=20

backend_run:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
        MLFLOW_TRACKING_INSECURE_TLS=$(MLFLOW_TRACKING_INSECURE_TLS) \
        MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL) \
        AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
        AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
        AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
        uvicorn backend.main:app --reload --port 8100

frontend_run:
	streamlit run frontend/home.py