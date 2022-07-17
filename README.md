# Finegrained app

The goal is to build REST api and UI
to interact with deployed models.

Main components are:
1. Triton-inference-serve to serve models
2. Backend api for simple http rest api and
pre/post-processing steps
3. Frontend api to interact with models

### Deployment

The deployment happens via `docker-compose.yml`

Build apps:
```shell
docker-compose build triton
docker-compose build backend
docker-compose build frontend
```

Run: `docker-compose -d up`

Access:
- triton at localhost:8001 for grpc
- triton at localhost:8000 for http
- http://localhost:8100 for rest api
- http://localhost:8100/redoc for rest api docs
- http://localhost:8501 for the frontend app