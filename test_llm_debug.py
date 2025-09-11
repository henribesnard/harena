#!/usr/bin/env python3

"""
Test simple pour diagnostiquer le problème LLM response_generator
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# Ajouter le chemin racine
sys.path.insert(0, str(Path(__file__).parent))

from conversation_service.config.settings import ConfigManager
from conversation_service.agents.llm.llm_provider import LLMProviderManager, LLMRequest, ProviderConfig, ProviderType

async def test_llm_direct():
    """Test direct des providers LLM"""
    
    print("=== TEST LLM PROVIDERS ===\n")
    
    # 1. Vérifier les API keys dans l'environnement
    print("1. API Keys dans l'environnement:")
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"   DEEPSEEK_API_KEY: {'OK' if deepseek_key else 'MANQUANT'}")
    print(f"   OPENAI_API_KEY: {'OK' if openai_key else 'MANQUANT'}")
    
    if deepseek_key:
        print(f"   DeepSeek key preview: {deepseek_key[:10]}...")
    if openai_key:
        print(f"   OpenAI key preview: {openai_key[:10]}...")
    
    # 2. Test ConfigManager
    print("\n2. Test ConfigManager:")
    config_manager = ConfigManager()
    await config_manager.load_configurations()
    
    llm_config = config_manager.get_llm_providers_config()
    print(f"   Configuration chargée: {len(llm_config.get('providers', {}))} providers")
    
    for provider_name, provider_config in llm_config.get('providers', {}).items():
        enabled = provider_config.get('enabled', False)
        api_key = provider_config.get('api_key', '')
        has_key = bool(api_key)
        base_url = provider_config.get('base_url', 'N/A')
        key_preview = (api_key[:10] + '...' if api_key else 'VIDE')
        print(f"   {provider_name}: enabled={enabled}, has_key={has_key}, base_url={base_url}, key={key_preview}")
    
    # 3. Test LLMProviderManager
    print("\n3. Test LLMProviderManager:")
    
    provider_configs = {}
    providers = llm_config.get("providers", {})
    
    # DeepSeek
    if providers.get("deepseek", {}).get("enabled", False):
        deepseek_config = providers["deepseek"]
        provider_configs[ProviderType.DEEPSEEK] = ProviderConfig(
            api_key=deepseek_config.get("api_key", ""),
            base_url=deepseek_config.get("base_url", "https://api.deepseek.com"),
            models=[deepseek_config.get("model", "deepseek-chat")],
            capabilities=[],
            rate_limit_rpm=deepseek_config.get("rate_limit", 60),
            priority=deepseek_config.get("priority", 1)
        )
        print(f"   DeepSeek configure: {deepseek_config.get('base_url')}")
    else:
        print(f"   DeepSeek non configure")
    
    # OpenAI
    if providers.get("openai", {}).get("enabled", False):
        openai_config = providers["openai"]
        provider_configs[ProviderType.OPENAI] = ProviderConfig(
            api_key=openai_config.get("api_key", ""),
            base_url=openai_config.get("base_url", "https://api.openai.com/v1"),
            models=[openai_config.get("model", "gpt-3.5-turbo")],
            capabilities=[],
            rate_limit_rpm=openai_config.get("rate_limit", 60),
            priority=openai_config.get("priority", 2)
        )
        print(f"   OpenAI configure: {openai_config.get('base_url')}")
    else:
        print(f"   OpenAI non configure")
    
    if not provider_configs:
        print("   ERREUR: Aucun provider configure!")
        return
    
    # 4. Initialiser et tester le manager
    print("\n4. Test génération LLM:")
    llm_manager = LLMProviderManager(provider_configs)
    init_success = await llm_manager.initialize()
    print(f"   Initialisation: {'OK' if init_success else 'ECHEC'}")
    
    # 5. Test simple d'appel
    if init_success:
        test_request = LLMRequest(
            messages=[{
                "role": "user",
                "content": "Réponds simplement 'OK' pour tester la connexion"
            }],
            temperature=0.1,
            max_tokens=10
        )
        
        print("   Test appel LLM...")
        response = await llm_manager.generate(test_request)
        
        if response.error:
            print(f"   ERREUR: {response.error}")
        else:
            print(f"   SUCCES: {response.content}")
            print(f"   Provider utilise: {response.provider_used.value}")
            print(f"   Modele: {response.model_used}")
    
    # 6. Health check
    print("\n5. Health check:")
    health = llm_manager.get_health_status()
    print(f"   Status global: {health.get('status')}")
    
    for provider_name, provider_status in health.get('providers', {}).items():
        available = provider_status.get('available', False)
        print(f"   {provider_name}: {'disponible' if available else 'indisponible'}")
    
    await llm_manager.close()

if __name__ == "__main__":
    asyncio.run(test_llm_direct())