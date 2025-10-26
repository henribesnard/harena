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
from typing import Optional, Literal, Any, Iterator, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

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


class LLMWithFallback(BaseChatModel):
    """
    Wrapper LLM avec fallback automatique

    Si le LLM primaire échoue, bascule automatiquement sur le fallback.
    """

    primary_llm: ChatOpenAI
    fallback_llm: Optional[ChatOpenAI] = None
    primary_provider: str = "unknown"
    fallback_provider: str = "unknown"
    fallback_used: bool = False

    def __init__(
        self,
        primary_llm: ChatOpenAI,
        fallback_llm: Optional[ChatOpenAI] = None,
        primary_provider: str = "unknown",
        fallback_provider: str = "unknown",
        **kwargs
    ):
        """
        Initialise le LLM avec fallback

        Args:
            primary_llm: LLM principal
            fallback_llm: LLM de fallback (optionnel)
            primary_provider: Nom du provider principal
            fallback_provider: Nom du provider de fallback
        """
        # IMPORTANT: Passer les valeurs à super().__init__() pour Pydantic v2
        super().__init__(
            primary_llm=primary_llm,
            fallback_llm=fallback_llm,
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            fallback_used=False,
            **kwargs
        )

        logger.info(f"LLMWithFallback initialized: primary={primary_provider}, fallback={fallback_provider or 'none'}")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llm_with_fallback"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override base implementation to use invoke."""
        try:
            logger.debug(f"Attempting _generate with primary LLM ({self.primary_provider})")
            result = self.primary_llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self.fallback_used = False
            return result
        except Exception as e:
            if self.fallback_llm:
                logger.warning(
                    f"Primary LLM ({self.primary_provider}) failed: {str(e)}. "
                    f"Falling back to {self.fallback_provider}"
                )
                try:
                    result = self.fallback_llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
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

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override base async implementation."""
        try:
            logger.debug(f"Attempting _agenerate with primary LLM ({self.primary_provider})")
            result = await self.primary_llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self.fallback_used = False
            return result
        except Exception as e:
            if self.fallback_llm:
                logger.warning(
                    f"Primary LLM ({self.primary_provider}) failed: {str(e)}. "
                    f"Falling back to {self.fallback_provider}"
                )
                try:
                    result = await self.fallback_llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
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

    async def apredict_messages(self, *args, **kwargs):
        """apredict_messages avec fallback (utilisé pour function calling)"""
        try:
            logger.info(f"🔵 [LLM] Attempting apredict_messages with PRIMARY LLM: {self.primary_provider} (model: {self.primary_llm.model_name})")
            result = await self.primary_llm.apredict_messages(*args, **kwargs)

            # Vérifier si le function call est présent (si demandé)
            if kwargs.get("function_call") or kwargs.get("functions"):
                function_call = result.additional_kwargs.get("function_call")
                if not function_call:
                    # DeepSeek n'a pas retourné de function call, tenter le fallback
                    logger.warning(f"⚠️  [LLM] Primary LLM ({self.primary_provider}) returned no function_call - trying fallback")
                    raise ValueError("No function call in LLM response - trying fallback")

            self.fallback_used = False
            logger.info(f"✅ [LLM] SUCCESS with PRIMARY LLM: {self.primary_provider}")
            return result
        except Exception as e:
            if self.fallback_llm:
                logger.warning(
                    f"❌ [LLM] Primary LLM ({self.primary_provider}) failed: {str(e)}. "
                    f"🔄 FALLING BACK to {self.fallback_provider}"
                )
                try:
                    logger.info(f"🟠 [LLM] Attempting apredict_messages with FALLBACK LLM: {self.fallback_provider} (model: {self.fallback_llm.model_name})")
                    result = await self.fallback_llm.apredict_messages(*args, **kwargs)
                    self.fallback_used = True
                    logger.info(f"✅ [LLM] SUCCESS with FALLBACK LLM: {self.fallback_provider}")
                    return result
                except Exception as fallback_error:
                    logger.error(f"❌ [LLM] Fallback LLM ({self.fallback_provider}) also failed: {str(fallback_error)}")
                    raise Exception(
                        f"Both primary ({self.primary_provider}) and fallback ({self.fallback_provider}) failed. "
                        f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}"
                    )
            else:
                logger.error(f"❌ [LLM] Primary LLM ({self.primary_provider}) failed and no fallback configured: {str(e)}")
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
    provider: Optional[str] = None,  # Override provider (for fallback creation)
    _is_creating_fallback: bool = False,  # Internal flag to prevent recursion
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
        _is_creating_fallback: INTERNAL - indique qu'on crée un fallback LLM (évite récursion)
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
    # Si on crée déjà un fallback LLM, ne PAS créer de fallback récursif
    if _is_creating_fallback:
        fallback_enabled = False
        logger.debug("Creating fallback LLM - disabling recursive fallback")
    else:
        # Déterminer si on active le fallback
        fallback_enabled = use_fallback if use_fallback is not None else settings.LLM_FALLBACK_ENABLED

    # Créer le LLM primaire (utiliser provider override si fourni, sinon LLM_PRIMARY_PROVIDER)
    primary_provider = provider if provider else settings.LLM_PRIMARY_PROVIDER
    primary_factory = LLMFactory.from_settings(settings, provider=primary_provider)
    primary_llm = primary_factory.create_llm(model=model, temperature=temperature, **kwargs)

    logger.info(f"🏗️  [LLM FACTORY] Creating LLM with:")
    logger.info(f"   - Primary Provider: {primary_provider}")
    logger.info(f"   - Primary Model: {primary_llm.model_name}")
    logger.info(f"   - Fallback Enabled: {fallback_enabled}")
    logger.info(f"   - Fallback Provider: {settings.LLM_FALLBACK_PROVIDER if fallback_enabled else 'None'}")

    # Si fallback désactivé, retourner le LLM primaire directement
    if not fallback_enabled:
        logger.info(f"✅ [LLM FACTORY] LLM created without fallback: {primary_provider}")
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

            # IMPORTANT: Créer le fallback LLM SANS fallback récursif
            # On passe _is_creating_fallback=True et provider=fallback_provider
            fallback_llm = create_llm_from_settings(
                settings,
                model=fallback_model,
                temperature=temperature,
                provider=fallback_provider,  # Use fallback provider
                _is_creating_fallback=True,  # Prevent recursive fallback
                **kwargs
            )

            logger.info(f"   - Fallback Model: {fallback_llm.model_name}")
            logger.info(f"✅ [LLM FACTORY] LLM created with fallback: {primary_provider} → {fallback_provider}")
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
