"""
Service d'extraction de noms de marchands avec Deepseek LLM.

Ce service utilise l'API Deepseek pour analyser les descriptions de transactions
et extraire intelligemment les noms de marchands avec un score de confiance.
"""

import os
import logging
import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MerchantExtractionResult:
    """Résultat d'extraction de nom de marchand."""
    merchant_name: Optional[str]
    confidence: float  # 0.0 à 1.0
    processing_time: float
    raw_response: Optional[str] = None
    error_message: Optional[str] = None

class DeepseekMerchantService:
    """Service d'extraction de marchands avec Deepseek LLM."""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ DEEPSEEK_API_KEY non configurée - service LLM désactivé")
        
        self.api_base = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.95"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "150"))
        
        # Configuration du timeout et retry
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.max_retries = 3
        
    async def extract_merchant_name(
        self, 
        description: str, 
        amount: float,
        transaction_type: str = None
    ) -> MerchantExtractionResult:
        """
        Extrait le nom du marchand à partir de la description de transaction.
        
        Args:
            description: Description de la transaction
            amount: Montant de la transaction
            transaction_type: Type de transaction ("debit" ou "credit")
            
        Returns:
            MerchantExtractionResult: Résultat avec nom et confiance
        """
        if not self.api_key:
            return MerchantExtractionResult(
                merchant_name=None,
                confidence=0.0,
                processing_time=0.0,
                error_message="DEEPSEEK_API_KEY not configured"
            )
        
        if not description or not description.strip():
            return MerchantExtractionResult(
                merchant_name=None,
                confidence=0.0,
                processing_time=0.0,
                error_message="Empty description"
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Déterminer automatiquement le type de transaction si non fourni
            if transaction_type is None:
                transaction_type = "debit" if amount < 0 else "credit"
            
            # Construire le prompt selon le type de transaction
            prompt = self._build_extraction_prompt(description, amount, transaction_type)
            
            # Appel API avec retry
            result = await self._call_deepseek_api(prompt)
            processing_time = asyncio.get_event_loop().time() - start_time
            
            if result:
                return MerchantExtractionResult(
                    merchant_name=result.get("merchant_name"),
                    confidence=float(result.get("confidence", 0.0)),
                    processing_time=processing_time,
                    raw_response=json.dumps(result)
                )
            else:
                return MerchantExtractionResult(
                    merchant_name=None,
                    confidence=0.0,
                    processing_time=processing_time,
                    error_message="No valid response from LLM"
                )
                
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Erreur extraction marchand: {e}")
            return MerchantExtractionResult(
                merchant_name=None,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _build_extraction_prompt(self, description: str, amount: float, transaction_type: str) -> str:
        """Construit le prompt d'extraction selon le type de transaction."""
        
        if transaction_type == "debit":
            # Pour les dépenses - extraire le marchand/bénéficiaire
            prompt = f"""Analyse cette transaction bancaire et extrait le nom du marchand ou bénéficiaire.

Transaction: "{description}"
Montant: {abs(amount):.2f} EUR (dépense)

Tâche: Identifier le marchand, magasin, service ou personne qui a reçu le paiement.

Exemples:
- "CARREFOUR MARKET 123 PARIS" → "Carrefour Market"
- "VIR LOYER OCTOBRE MARTIN" → "Martin"
- "CB AMAZON FR 12345" → "Amazon"
- "PRELEVEMENT EDF GAZ" → "EDF"

Réponds uniquement en JSON:
{{"merchant_name": "nom du marchand ou null", "confidence": 0.85}}

Confiance:
- 0.9-1.0: Nom de marque/entreprise claire
- 0.7-0.8: Nom probable avec contexte
- 0.4-0.6: Extraction incertaine
- 0.0-0.3: Impossible à identifier

JSON:"""

        else:
            # Pour les revenus - extraire la source du paiement
            prompt = f"""Analyse cette transaction bancaire et extrait la source du revenu.

Transaction: "{description}"
Montant: {abs(amount):.2f} EUR (revenu)

Tâche: Identifier l'entreprise, institution ou personne qui a effectué le versement.

Exemples:
- "VIR SALAIRE OCTOBRE ENTREPRISE ABC" → "Entreprise ABC"
- "VIR MARTIN REMBOURSEMENT" → "Martin"
- "VIREMENT CAF ALLOCATION" → "CAF"
- "VIR POLE EMPLOI" → "Pôle Emploi"

Réponds uniquement en JSON:
{{"merchant_name": "source du paiement ou null", "confidence": 0.85}}

Confiance:
- 0.9-1.0: Source institutionnelle claire
- 0.7-0.8: Source probable avec contexte
- 0.4-0.6: Extraction incertaine
- 0.0-0.3: Impossible à identifier

JSON:"""

        return prompt.strip()
    
    async def _call_deepseek_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Effectue l'appel API vers Deepseek avec retry."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            return self._parse_json_response(content)
                        
                        elif response.status == 429:
                            # Rate limiting - attendre avant retry
                            wait_time = 2 ** attempt
                            logger.warning(f"⏰ Rate limit - attente {wait_time}s (tentative {attempt + 1}/{self.max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ API Error {response.status}: {error_text}")
                            if attempt == self.max_retries - 1:
                                return None
                            
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout tentative {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Erreur réseau tentative {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse la réponse JSON du LLM."""
        if not content:
            return None
            
        try:
            # Nettoyer la réponse (enlever markdown, espaces, etc.)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parser JSON
            result = json.loads(content)
            
            # Validation basique
            if isinstance(result, dict) and "merchant_name" in result and "confidence" in result:
                # Normaliser les valeurs
                merchant_name = result.get("merchant_name")
                if merchant_name == "null" or merchant_name == "":
                    merchant_name = None
                
                confidence = float(result.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))  # Clamp entre 0 et 1
                
                return {
                    "merchant_name": merchant_name,
                    "confidence": confidence
                }
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"⚠️ Erreur parsing JSON: {e}, content: {content[:100]}")
        
        return None

# Instance singleton
_merchant_service = None

def get_merchant_service() -> DeepseekMerchantService:
    """Récupère l'instance du service d'extraction de marchands."""
    global _merchant_service
    if _merchant_service is None:
        _merchant_service = DeepseekMerchantService()
    return _merchant_service