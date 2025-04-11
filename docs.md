uvicorn user_service.main:app --host 0.0.0.0 --port 8001 --reload
alembic revision --autogenerate -m "Initial commit"
alembic upgrade head