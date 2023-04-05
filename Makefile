run: up logs

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