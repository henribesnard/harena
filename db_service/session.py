from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager

from config_service.config import settings

# Récupérer l'URL de connexion, fallback sur SQLite pour les tests
database_url = settings.DATABASE_URL or "sqlite://"
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

# Création de l'engine central
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=20,  # Augmenté de 5 à 20
    max_overflow=30,  # Augmenté de 10 à 30
    pool_recycle=3600  # Recycler les connexions après 1h
)

# Création d'une factory de session partagée
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

# Pour obtenir une session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper contextmanager pour les opérations qui ne sont pas dans des endpoints FastAPI
@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()