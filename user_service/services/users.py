from __future__ import annotations

from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import Optional, Dict, Any

from db_service.models.user import User, UserPreference
from user_service.schemas.user import UserCreate, UserUpdate
from user_service.core.security import get_password_hash, verify_password
from db_service.config.default_preferences import get_default_budget_settings
from user_service.utils import deep_merge


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[User]:
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user_in: UserCreate) -> User:
    # Vérifier si l'email existe déjà
    existing_user = get_user_by_email(db, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Créer l'utilisateur
    db_user = User(
        email=user_in.email,
        password_hash=get_password_hash(user_in.password),
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        is_active=True,
        is_superuser=getattr(user_in, 'is_superuser', False)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Créer les préférences utilisateur par défaut avec budget_settings
    db_pref = UserPreference(
        user_id=db_user.id,
        notification_settings={},
        display_preferences={},
        budget_settings=get_default_budget_settings()
    )
    db.add(db_pref)
    db.commit()
    
    return db_user


def update_user(db: Session, user_id: int, user_in: UserUpdate) -> User:
    db_user = get_user_by_id(db, user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    update_data = user_in.dict(exclude_unset=True)

    # Extraire les préférences si présentes
    preferences_update = update_data.pop("preferences", None)

    # Hacher le mot de passe si fourni
    if "password" in update_data:
        password = update_data.pop("password")
        update_data["password_hash"] = get_password_hash(password)

    # Mettre à jour les champs de l'utilisateur (sauf preferences)
    for field, value in update_data.items():
        setattr(db_user, field, value)

    # Gérer la mise à jour des préférences séparément
    if preferences_update is not None:
        # Récupérer ou créer les préférences existantes
        if db_user.preferences is None:
            # Créer de nouvelles préférences avec les valeurs par défaut
            db_pref = UserPreference(
                user_id=db_user.id,
                notification_settings={},
                display_preferences={},
                budget_settings=get_default_budget_settings()
            )
            db.add(db_pref)
            db.flush()  # Pour obtenir l'ID
            db.refresh(db_user)

        # Effectuer un merge profond pour chaque section de préférences
        current_prefs = {
            "notification_settings": db_user.preferences.notification_settings or {},
            "display_preferences": db_user.preferences.display_preferences or {},
            "budget_settings": db_user.preferences.budget_settings or {}
        }

        # Merge des préférences
        merged_prefs = deep_merge(current_prefs, preferences_update)

        # Appliquer les préférences mergées
        if "notification_settings" in merged_prefs:
            db_user.preferences.notification_settings = merged_prefs["notification_settings"]
        if "display_preferences" in merged_prefs:
            db_user.preferences.display_preferences = merged_prefs["display_preferences"]
        if "budget_settings" in merged_prefs:
            db_user.preferences.budget_settings = merged_prefs["budget_settings"]

        db.add(db_user.preferences)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def is_active_user(user: User) -> bool:
    return user.is_active


def deactivate_user(db: Session, user_id: int) -> User:
    db_user = get_user_by_id(db, user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    db_user.is_active = False
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user