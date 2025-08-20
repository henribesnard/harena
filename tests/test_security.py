import sys
import importlib
import types


def test_bcrypt_version_without_about(monkeypatch):
    """Assure que l'import de security fonctionne même sans __about__."""
    import bcrypt

    # Supprimer l'attribut __about__ pour simuler les anciennes versions
    monkeypatch.delattr(bcrypt, "__about__", raising=False)

    # Injecter un module de configuration minimal pour éviter les dépendances
    dummy_cfg = types.ModuleType("config_service.config")
    dummy_cfg.settings = types.SimpleNamespace(
        ACCESS_TOKEN_EXPIRE_MINUTES=15,
        SECRET_KEY="test",
    )
    pkg = types.ModuleType("config_service")
    pkg.config = dummy_cfg
    sys.modules["config_service"] = pkg
    sys.modules["config_service.config"] = dummy_cfg

    # Réimporter le module de sécurité avec le module bcrypt modifié
    sys.modules.pop("user_service.core.security", None)
    security = importlib.import_module("user_service.core.security")

    assert security.bcrypt_version == bcrypt.__version__

