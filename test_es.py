#!/usr/bin/env python3
"""
Test rapide du fix pour le client Bonsai
"""
import asyncio
import aiohttp
import os
from urllib.parse import urlparse

async def test_simple_bonsai_fix():
    """Test du fix le plus simple possible."""
    
    # Charger BONSAI_URL
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    bonsai_url = os.environ.get("BONSAI_URL")
    if not bonsai_url:
        print("❌ BONSAI_URL non configurée")
        return False
    
    print("🧪 Test du client Bonsai corrigé...")
    
    try:
        # Parser l'URL
        parsed = urlparse(bonsai_url)
        auth = aiohttp.BasicAuth(parsed.username, parsed.password)
        base_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        
        print(f"🔗 Connexion à: {parsed.hostname}:{parsed.port}")
        
        # Session simple SANS les paramètres problématiques
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30
            # PAS de keepalive_timeout ni force_close
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        ) as session:
            
            # Test connexion de base
            print("🔍 Test connexion...")
            async with session.get(base_url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Connexion OK: {data.get('cluster_name')} v{data.get('version', {}).get('number')}")
                else:
                    print(f"❌ Erreur HTTP: {response.status}")
                    return False
            
            # Test santé cluster
            print("🩺 Test santé...")
            async with session.get(f"{base_url}/_cluster/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"💚 Santé cluster: {health.get('status')}")
                else:
                    print(f"⚠️ Pas de données de santé: {response.status}")
            
            # Test recherche basique
            print("🔍 Test recherche...")
            search_body = {
                "query": {"match_all": {}},
                "size": 1
            }
            
            search_url = f"{base_url}/harena_transactions/_search"
            async with session.post(search_url, json=search_body) as response:
                if response.status == 200:
                    result = await response.json()
                    total = result.get('hits', {}).get('total', 0)
                    if isinstance(total, dict):
                        total = total.get('value', 0)
                    print(f"🔍 Index accessible: {total} documents")
                elif response.status == 404:
                    print("📚 Index harena_transactions n'existe pas encore (normal)")
                else:
                    print(f"⚠️ Erreur recherche: {response.status}")
            
            print("🎉 Tous les tests passés!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_simple_bonsai_fix())
    if result:
        print("\n✅ Le fix fonctionne! Vous pouvez l'appliquer à votre code.")
    else:
        print("\n❌ Le fix ne résout pas le problème. Investigation supplémentaire nécessaire.")