"""
LLM Factory - Crée des instances LLM avec support de fallback automatique

Supporte:
- OpenAI (gpt-4o, gpt-4o-mini, etc.)
- DeepSeek (deepseek-chat, deepseek-reasoner)
- Fallback automatique si le provider principal échoue

Author: Claude Code
Date: 2025-10-26
Updated: 2025-10-26 (added fallback support)
"""

import logging
from typing import Optional, Literal, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

LLMProvider = Literal["openai", "deepseek"]


class LLMFactory:
    """
    Factory pour créer des instances LLM selon le provider

    Usage:
        factory = LLMFactory(
            provider="deepseek",
            api_key="sk-...",
            base_url="https://api.deepseek.com"
        )
        llm = factory.create_llm(model="deepseek-chat", temperature=0.1)
    """

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Initialise la factory

        Args:
            provider: Provider LLM ("openai" ou "deepseek")
            api_key: Clé API du provider
            base_url: URL de base (requis pour DeepSeek, optionnel pour OpenAI)
            default_model: Modèle par défaut si non spécifié
            timeout: Timeout en secondes
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout

        logger.info(f"LLMFactory initialized: provider={provider}, base_url={base_url}, timeout={timeout}s")

    def create_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs
    ) -> ChatOpenAI:
        """
        Crée une instance LLM

        Args:
            model: Nom du modèle (si None, utilise default_model)
            temperature: Température (0.0 = déterministe, 1.0 = créatif)
            **kwargs: Paramètres additionnels pour ChatOpenAI

        Returns:
            Instance ChatOpenAI configurée
        """
        model_name = model or self.default_model

        if not model_name:
            raise ValueError("Model name must be specified or default_model must be set")

        # Configuration commune
        config = {
            "model": model_name,
            "temperature": temperature,
            "api_key": self.api_key,
            "timeout": kwargs.pop("timeout", self.timeout),
            **kwargs
        }

        # Configuration spécifique au provider
        if self.provider == "deepseek":
            if not self.base_url:
                raise ValueError("base_url is required for DeepSeek provider")

            config["base_url"] = self.base_url

            logger.info(f"Creating DeepSeek LLM: model={model_name}, temperature={temperature}, timeout={config['timeout']}s")

        elif self.provider == "openai":
            # OpenAI utilise l'API officielle (pas besoin de base_url custom)
            # Mais on peut l'override si fourni (pour Azure OpenAI par exemple)
            if self.base_url:
                config["base_url"] = self.base_url

            logger.info(f"Creating OpenAI LLM: model={model_name}, temperature={temperature}, timeout={config['timeout']}s")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return ChatOpenAI(**config)

    @classmethod
    def from_settings(cls, settings, provider: Optional[str] = None):
        """
        Crée une factory depuis les settings de l'application

        Args:
            settings: Instance Settings
            provider: Provider à utiliser (si None, utilise settings.LLM_PROVIDER)

        Returns:
            LLMFactory configurée
        """
        provider = (provider or settings.LLM_PROVIDER).lower()

        if provider == "openai":
            return cls(
                provider="openai",
                api_key=settings.OPENAI_API_KEY,
                base_url=None,  # OpenAI officiel
                default_model=settings.LLM_MODEL,
                timeout=settings.LLM_TIMEOUT
            )

        elif provider == "deepseek":
            return cls(
                provider="deepseek",
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                default_model=settings.LLM_MODEL,
                timeout=settings.LLM_TIMEOUT
            )

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {provider}. "
                f"Supported: openai, deepseek"
            )


class LLMWithFallback:
    """
    Wrapper LLM avec fallback automatique

    Si le LLM primaire échoue, bascule automatiquement sur le fallback.
    """

    def __init__(
        self,
        primary_llm: ChatOpenAI,
        fallback_llm: Optional[ChatOpenAI] = None,
        primary_provider: str = "unknown",
        fallback_provider: str = "unknown"
    ):
        """
        Initialise le LLM avec fallback

        Args:
            primary_llm: LLM principal
            fallback_llm: LLM de fallback (optionnel)
            primary_provider: Nom du provider principal
            fallback_provider: Nom du provider de fallback
        """
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.fallback_used = False

        logger.info(f"LLMWithFallback initialized: primary={primary_provider}, fallback={fallback_provider or 'none'}")

    def invoke(self, *args, **kwargs):
        """Invoke synchrone avec fallback"""
        try:
            logger.debug(f"Attempting invoke with primary LLM ({self.primary_provider})")
            result = self.primary_llm.invoke(*args, **kwargs)
            self.fallback_used = False
            return result
        except Exception as e:
            if self.fallback_llm:
                logger.warning(
                    f"Primary LLM ({self.primary_provider}) failed: {str(e)}. "
                    f"Falling back to {self.fallback_provider}"
                )
                try:
                    result = self.fallback_llm.invoke(*args, **kwargs)
                    self.fallback_used = True
                    logger.info(f"Fallback LLM ({self.fallback_provider}) succeeded")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback LLM ({self.fallback_provider}) also failed: {str(fallback_error)}")
                    raise Exception(
                        f"Both primary ({self.primary_provider}) and fallback ({self.fallback_provider}) failed. "
                        f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}"
                    )
            else:
                logger.error(f"Primary LLM ({self.primary_provider}) failed and no fallback configured: {str(e)}")
                raise

    async def ainvoke(self, *args, **kwargs):
        """Invoke asynchrone avec fallback"""
        try:
            logger.debug(f"Attempting ainvoke with primary LLM ({self.primary_provider})")
            result = await self.primary_llm.ainvoke(*args, **kwargs)
            self.fallback_used = False
            return result
        except Exception as e:
            if self.fallback_llm:
                logger.warning(
                    f"Primary LLM ({self.primary_provider}) failed: {str(e)}. "
                    f"Falling back to {self.fallback_provider}"
                )
                try:
                    result = await self.fallback_llm.ainvoke(*args, **kwargs)
                    self.fallback_used = True
                    logger.info(f"Fallback LLM ({self.fallback_provider}) succeeded")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback LLM ({self.fallback_provider}) also failed: {str(fallback_error)}")
                    raise Exception(
                        f"Both primary ({self.primary_provider}) and fallback ({self.fallback_provider}) failed. "
                        f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}"
                    )
            else:
                logger.error(f"Primary LLM ({self.primary_provider}) failed and no fallback configured: {str(e)}")
                raise

    async def astream(self, *args, **kwargs):
        """Stream asynchrone avec fallback"""
        try:
            logger.debug(f"Attempting astream with primary LLM ({self.primary_provider})")
            async for chunk in self.primary_llm.astream(*args, **kwargs):
                yield chunk
            self.fallback_used = False
        except Exception as e:
            if self.fallback_llm:
                logger.warning(
                    f"Primary LLM ({self.primary_provider}) failed during streaming: {str(e)}. "
                    f"Falling back to {self.fallback_provider}"
                )
                try:
                    async for chunk in self.fallback_llm.astream(*args, **kwargs):
                        yield chunk
                    self.fallback_used = True
                    logger.info(f"Fallback LLM ({self.fallback_provider}) streaming succeeded")
                except Exception as fallback_error:
                    logger.error(f"Fallback LLM ({self.fallback_provider}) streaming also failed: {str(fallback_error)}")
                    raise Exception(
                        f"Both primary ({self.primary_provider}) and fallback ({self.fallback_provider}) streaming failed. "
                        f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}"
                    )
            else:
                logger.error(f"Primary LLM ({self.primary_provider}) failed during streaming and no fallback configured: {str(e)}")
                raise

    @property
    def model_name(self):
        """Retourne le model_name du LLM actif"""
        return self.primary_llm.model_name


def create_llm_from_settings(
    settings,
    model: Optional[str] = None,
    temperature: float = 0.1,
    use_fallback: Optional[bool] = None,
    **kwargs
):
    """
    Fonction utilitaire pour créer un LLM directement depuis les settings
    avec support de fallback automatique

    Args:
        settings: Instance Settings
        model: Nom du modèle (optionnel, utilise settings.LLM_MODEL par défaut)
        temperature: Température
        use_fallback: Active le fallback (par défaut settings.LLM_FALLBACK_ENABLED)
        **kwargs: Paramètres additionnels

    Returns:
        LLMWithFallback si fallback activé, ChatOpenAI sinon

    Usage:
        from app.config.settings import settings
        from app.core.llm_factory import create_llm_from_settings

        # Avec fallback automatique (DeepSeek → OpenAI)
        llm = create_llm_from_settings(settings, temperature=0.1)

        # Sans fallback
        llm = create_llm_from_settings(settings, use_fallback=False)
    """
    # Déterminer si on active le fallback
    fallback_enabled = use_fallback if use_fallback is not None else settings.LLM_FALLBACK_ENABLED

    # Créer le LLM primaire
    primary_provider = settings.LLM_PRIMARY_PROVIDER
    primary_factory = LLMFactory.from_settings(settings, provider=primary_provider)
    primary_llm = primary_factory.create_llm(model=model, temperature=temperature, **kwargs)

    # Si fallback désactivé, retourner le LLM primaire directement
    if not fallback_enabled:
        logger.info(f"LLM created without fallback: {primary_provider}")
        return primary_llm

    # Créer le LLM de fallback
    fallback_provider = settings.LLM_FALLBACK_PROVIDER
    if fallback_provider and fallback_provider != primary_provider:
        try:
            # Utiliser le modèle de fallback spécifique
            fallback_model = model
            if not fallback_model:
                # Utiliser le modèle de fallback configuré
                if fallback_provider == "openai":
                    fallback_model = settings.LLM_FALLBACK_MODEL
                elif fallback_provider == "deepseek":
                    fallback_model = "deepseek-chat"

            fallback_factory = LLMFactory.from_settings(settings, provider=fallback_provider)
            fallback_llm = fallback_factory.create_llm(model=fallback_model, temperature=temperature, **kwargs)

            logger.info(f"LLM created with fallback: {primary_provider} → {fallback_provider}")
            return LLMWithFallback(
                primary_llm=primary_llm,
                fallback_llm=fallback_llm,
                primary_provider=primary_provider,
                fallback_provider=fallback_provider
            )
        except Exception as e:
            logger.warning(f"Failed to create fallback LLM ({fallback_provider}): {str(e)}. Using primary only.")
            return primary_llm
    else:
        logger.info(f"LLM created without fallback (no valid fallback provider): {primary_provider}")
        return primary_llm
