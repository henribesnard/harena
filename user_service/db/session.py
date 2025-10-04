# user_service/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config_service.config import settings

# Construire l'URL de base de données depuis la configuration
database_url = settings.DATABASE_URL or "sqlite://"
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=20,  # Augmenté de 5 à 20
    max_overflow=30,  # Augmenté de 10 à 30
    pool_recycle=3600  # Recycler les connexions après 1h
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dépendance pour obtenir la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()