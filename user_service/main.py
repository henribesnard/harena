# user_service/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from user_service.api.endpoints import users
from user_service.core.config import settings

# Création de l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À remplacer par les domaines autorisés en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)