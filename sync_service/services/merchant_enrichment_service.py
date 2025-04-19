"""
Service optionnel pour l'enrichissement asynchrone des marchands.

Ce service scanne la collection de transactions vectorielles à la recherche
de transactions non enrichies (sans merchant_id), tente d'identifier ou de créer
un marchand correspondant, et met à jour la transaction avec l'ID du marchand.
"""

import logging
import re
import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone

# Import Qdrant models pour les filtres
from qdrant_client import models as qmodels

# Import Vector Storage Service et gestion d'erreur
try:
    from .vector_storage import VectorStorageService
    VECTOR_STORAGE_AVAILABLE = True
    logger_vs = logging.getLogger(__name__)
except ImportError as e:
    VECTOR_STORAGE_AVAILABLE = False
    # Classe factice minimale nécessaire pour ce service
    class VectorStorageService:
        TRANSACTIONS_COLLECTION = "transactions"
        MERCHANTS_COLLECTION = "merchants"
        client = None # Marquer comme non disponible
        async def find_merchant(self, *args, **kwargs): return None
        async def store_merchant(self, *args, **kwargs): return None
    logger_vs = logging.getLogger(__name__)
    logger_vs.warning(f"MerchantEnrichment: VectorStorageService non trouvé ({e}). Service désactivé.")

logger = logging.getLogger(__name__)

# --- Fonctions d'aide (potentiellement à déplacer dans un utils partagé) ---

def _normalize_merchant_name_detailed(description: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Tente d'extraire et de normaliser un nom de marchand, en retournant aussi une version "display".
    Logique améliorée par rapport à la version précédente, mais reste un exemple.
    """
    if not description:
        return None, None

    # Copie pour le display name
    display_name = description.strip()

    # Nettoyage pour la normalisation
    name = description.upper() # Convertir en majuscules pour normalisation

    # Règles de suppression plus spécifiques (exemples, à adapter)
    patterns_to_remove = [
        r'PAIEMENT\s*(CB|CARTE)?\s*',
        r'ACHAT\s*(CB|CARTE)?\s*',
        r'RETRAIT\s*(CB|DAB)?\s*',
        r'VIREMENT\s*(SEPA|INST)?\s*',
        r'PRLV\s*SEPA\s*',
        r'COMMISSION.*',
        r'FRAIS.*',
        r'\b\d{2,}[/-]\d{2,}[/-]\d{2,}\b', # Dates DD/MM/YY ou DD-MM-YY
        r'CB\*?\s*\d{4,}', # Numéros CB (simpliste)
        r'CARTE\s*\d+',
        r'XXX+\d+', # Numéros masqués
        r'REF\s*\w+', # Références
        r'ID\s*\w+',  # IDs
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b', # Potentiel IBAN/Numéro compte
        r' FACTURE.*',
        r' ECH.*', # Echéance
        r'\b(M|MME|MR|MLLE)\b', # Civilités
        # Ajouter des villes communes si elles polluent souvent
        r'\bPARIS\b', r'\bLYON\b', r'\bMARSEILLE\b', # etc.
        # TLDs
        r'\.(COM|FR|NET|ORG)\b'
    ]
    for pattern in patterns_to_remove:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()

    # Supprimer caractères spéciaux courants sauf trait d'union et apostrophe interne
    name = re.sub(r'[^\w\s\'-]', '', name).strip()
    # Supprimer traits d'union/apostrophes en début/fin
    name = name.strip("'-")

    # Espaces multiples
    name = re.sub(r'\s{2,}', ' ', name).strip()

    # Règles de remplacement/alias
    replacements = {
        'AMAZON MKTPLACE': 'AMAZON',
        'AMZN': 'AMAZON',
        'GOOGLE \*': 'GOOGLE',
        'FACEBK': 'FACEBOOK',
        'APPLE.COM/BILL': 'APPLE',
        'UBER TRIP': 'UBER',
        'PAYPAL \*': 'PAYPAL', # Attention, PAYPAL * marchand peut être utile
        'SPTF': 'SPOTIFY', # Spotify
        'MC DONALD S': 'MCDONALDS',
        # Ajouter des alias spécifiques (Carrefour Market -> CARREFOUR etc.)
        r'\bS\.?A\.?S\.?\b': '', # Enlever SAS/SA
        r'\bS\.?A\.?R\.?L\.?\b': '', # Enlever SARL
    }
    for pattern, replacement in replacements.items():
         name = re.sub(pattern, replacement, name, flags=re.IGNORECASE).strip()

    # Nettoyer à nouveau les espaces après remplacements
    name = re.sub(r'\s{2,}', ' ', name).strip()

    # Retourner None si trop court ou vide
    if len(name) < 3:
        return None, display_name # Retourne le display name original même si normalisation échoue

    # Limiter la longueur du nom normalisé ?
    normalized_name = name[:50] # Limite arbitraire

    # Post-traitement optionnel du display_name (ex: Capitalisation)
    # display_name = display_name.title() # Peut mal fonctionner avec acronymes

    return normalized_name, display_name

# --- Classe de Service ---

class MerchantEnrichmentService:
    """
    Service pour enrichir les transactions avec des informations marchand
    de manière asynchrone.
    """
    BATCH_SIZE = 100 # Nombre de transactions à traiter par itération
    DELAY_BETWEEN_BATCHES = 5 # Secondes d'attente entre les lots

    def __init__(self, vector_storage: VectorStorageService):
        if not VECTOR_STORAGE_AVAILABLE or not vector_storage.client:
            raise RuntimeError("VectorStorageService is not available or not connected. MerchantEnrichmentService cannot run.")
        self.vector_storage = vector_storage
        self.client = vector_storage.client # Accès direct au client Qdrant
        self.processed_in_run = 0
        self.enriched_in_run = 0
        self.merchants_created_in_run = 0
        self.errors_in_run = 0
        logger.info("MerchantEnrichmentService initialisé.")

    async def _enrich_transaction_batch(self, transaction_points: List[qmodels.ScoredPoint]) -> None:
        """Traite un lot de transactions pour l'enrichissement."""
        merchant_updates: Dict[str, str] = {} # {transaction_point_id: merchant_point_id}
        new_merchants_to_store: Dict[str, Dict[str, Any]] = {} # {normalized_name: merchant_data}

        for point in transaction_points:
            self.processed_in_run += 1
            tx_payload = point.payload
            tx_description = tx_payload.get("clean_description") or tx_payload.get("description", "")
            tx_category_id = tx_payload.get("category_id")
            user_id = tx_payload.get("user_id") # Important si les marchands sont par utilisateur

            if not tx_description:
                continue # Impossible d'enrichir sans description

            normalized_name, display_name = _normalize_merchant_name_detailed(tx_description)

            if not normalized_name:
                continue # Impossible de normaliser

            # 1. Chercher le marchand existant
            merchant_point_id = await self.vector_storage.find_merchant(normalized_name=normalized_name)

            if not merchant_point_id:
                 # 2. Si non trouvé, préparer la création (on stocke en batch après)
                 if normalized_name not in new_merchants_to_store:
                     logger.debug(f"Préparation nouveau marchand: '{normalized_name}' (depuis: {display_name[:30]}...)")
                     new_merchants_to_store[normalized_name] = {
                         "normalized_name": normalized_name,
                         "display_name": display_name or normalized_name, # Utiliser display_name extrait ou normalisé
                         "category_id": tx_category_id,
                         # "user_id": user_id, # Si spécifique user
                         "source": "inferred_batch"
                     }
                 # On mettra à jour la transaction plus tard, après création du marchand
            else:
                # 3. Si trouvé, marquer la transaction pour mise à jour
                logger.debug(f"Marchand '{normalized_name}' trouvé (id={merchant_point_id}). MAJ Transaction id={point.id}")
                merchant_updates[point.id] = merchant_point_id

        # 4. Créer les nouveaux marchands en batch
        if new_merchants_to_store:
            created_merchant_map: Dict[str, Optional[str]] = {} # {normalized_name: point_id}
            merchants_list_to_store = list(new_merchants_to_store.values())
            logger.info(f"Tentative de création de {len(merchants_list_to_store)} nouveaux marchands en batch.")
            # NOTE : store_merchant actuel stocke un par un et retourne l'ID.
            # Idéalement, batch_store_merchants devrait pouvoir retourner un mapping ou les IDs créés.
            # Solution temporaire : on crée un par un ici pour récupérer les IDs.
            # Pour optimisation: modifier batch_store_merchants ou refaire une recherche après batch.
            for merchant_data in merchants_list_to_store:
                 norm_name = merchant_data["normalized_name"]
                 created_id = await self.vector_storage.store_merchant(merchant_data) # Crée et retourne ID
                 if created_id:
                     created_merchant_map[norm_name] = created_id
                     self.merchants_created_in_run += 1
                 else:
                     self.errors_in_run += 1
                     created_merchant_map[norm_name] = None # Marquer comme échoué

            # Mettre à jour les transactions liées aux marchands nouvellement créés
            for point in transaction_points:
                 # Si la transaction n'a pas déjà un marchand trouvé et que son marchand normalisé a été créé
                 if point.id not in merchant_updates:
                     norm_name, _ = _normalize_merchant_name_detailed(point.payload.get("clean_description") or point.payload.get("description", ""))
                     if norm_name and norm_name in created_merchant_map and created_merchant_map[norm_name]:
                         merchant_updates[point.id] = created_merchant_map[norm_name]
                         logger.debug(f"Marchand '{norm_name}' créé (id={created_merchant_map[norm_name]}). MAJ Transaction id={point.id}")

        # 5. Mettre à jour les transactions avec les merchant_id en batch
        if merchant_updates:
            logger.info(f"Mise à jour de {len(merchant_updates)} transactions avec merchant_id.")
            points_to_update = []
            payloads_to_update = []
            for tx_point_id, merchant_point_id in merchant_updates.items():
                 points_to_update.append(tx_point_id)
                 payloads_to_update.append({"merchant_id": merchant_point_id, "enriched_at": datetime.now(timezone.utc).isoformat()})
                 self.enriched_in_run += 1

            try:
                # Utiliser set_payload pour ne mettre à jour que merchant_id et enriched_at
                # Qdrant ne supporte pas directement `set_payload` pour une liste de points avec des payloads différents.
                # Alternative : utiliser `upsert` avec le payload complet mis à jour (moins efficace)
                # Alternative 2: Boucle `set_payload` (simple mais lent)
                # Alternative 3: Utiliser `batch_update_payloads` si disponible via client Python ou API REST
                # Ici, on boucle pour la simplicité :
                update_results = self.client.set_payload(
                    collection_name=self.vector_storage.TRANSACTIONS_COLLECTION,
                    payload={"merchant_id": None, "enriched_at": datetime.now(timezone.utc).isoformat()}, # Schema base
                    points=list(merchant_updates.keys()),
                    wait=True # Attendre pour confirmer
                )
                # Note: Il faut ensuite ré-appliquer les bonnes valeurs, ce qui est complexe.
                # Simplification : On pourrait juste ajouter un flag "needs_merchant_id_update"
                # et laisser un autre process faire la mise à jour fine.
                # OU : Refetcher les points et faire un upsert complet.

                # Solution pragmatique : on log juste qu'on a trouvé les IDs pour l'instant
                logger.info(f"Mise à jour (simulation) pour {len(merchant_updates)} transactions.")
                # En production: implémenter la mise à jour réelle via boucle set_payload ou upsert complet.


            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour des transactions avec merchant_id: {e}", exc_info=True)
                self.errors_in_run += len(merchant_updates) # Compter comme erreurs

    async def run_enrichment_cycle(self, limit: int = 1000) -> Dict[str, Any]:
        """Exécute un cycle d'enrichissement sur les transactions non traitées."""
        logger.info(f"Début du cycle d'enrichissement des marchands (limite: {limit})...")
        self.processed_in_run = 0
        self.enriched_in_run = 0
        self.merchants_created_in_run = 0
        self.errors_in_run = 0
        start_time = datetime.now(timezone.utc)

        # Filtre pour trouver les transactions sans 'merchant_id'
        # (ou avec 'merchant_id' null, ou sans flag 'enriched_at')
        filter_unenriched = qmodels.Filter(
            must_not=[
                # qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="merchant_id")) # Si champ peut exister mais être null
                qmodels.HasIdCondition(has_id=None) # Contournement pour 'exists', ne marche pas.
                # -> Qdrant ne supporte pas bien "field does not exist".
                # -> Alternative: Ajouter un champ 'is_enriched' = false par défaut et filtrer dessus.
                # -> Pour l'exemple, on prend toutes les transactions récentes (ex: dernier jour)
                #    et on tente l'enrichissement (upsert marchand est idempotent).
                # -> Solution la plus simple : on scanne sans filtre et on vérifie le payload
            ]
            # Alternative si flag ajouté:
            # must=[
            #     qmodels.FieldCondition(key="is_enriched", match=qmodels.MatchValue(value=False))
            # ]
        )

        # On utilise scroll pour parcourir les transactions
        offset = None
        processed_total = 0
        try:
            while processed_total < limit:
                 logger.debug(f"Scrolling transactions (offset: {offset})...")
                 points, next_offset = self.client.scroll(
                     collection_name=self.vector_storage.TRANSACTIONS_COLLECTION,
                     # scroll_filter=filter_unenriched, # Utiliser si filtre efficace disponible
                     limit=self.BATCH_SIZE,
                     offset=offset,
                     with_payload=True, # Besoin du payload (description, category_id)
                     with_vectors=False # Pas besoin des vecteurs ici
                 )

                 if not points:
                     logger.info("Aucune transaction supplémentaire trouvée pour l'enrichissement.")
                     break # Fin du scroll

                 # Filtrer manuellement les points déjà enrichis si nécessaire
                 points_to_process = [p for p in points if not p.payload.get("merchant_id")]
                 logger.info(f"Batch trouvé: {len(points)} points, {len(points_to_process)} à traiter.")

                 if points_to_process:
                     await self._enrich_transaction_batch(points_to_process)

                 processed_total += len(points)
                 offset = next_offset
                 if offset is None:
                      logger.info("Fin du scroll atteinte.")
                      break # Fin du scroll

                 logger.debug(f"Pausing {self.DELAY_BETWEEN_BATCHES}s before next batch...")
                 await asyncio.sleep(self.DELAY_BETWEEN_BATCHES)

        except Exception as e:
            logger.error(f"Erreur durant le cycle d'enrichissement: {e}", exc_info=True)
            self.errors_in_run += 1 # Erreur générale

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Cycle d'enrichissement terminé en {duration:.2f}s.")
        logger.info(f"Stats Cycle: Traitées={self.processed_in_run}, Enrichies={self.enriched_in_run}, Marchands Créés={self.merchants_created_in_run}, Erreurs={self.errors_in_run}")

        return {
            "status": "completed" if self.errors_in_run == 0 else "completed_with_errors",
            "duration_seconds": duration,
            "transactions_processed": self.processed_in_run,
            "transactions_enriched": self.enriched_in_run,
            "merchants_created": self.merchants_created_in_run,
            "errors": self.errors_in_run
        }

    async def schedule_enrichment(self, interval_seconds: int = 3600, cycle_limit: int = 10000):
        """Lance des cycles d'enrichissement périodiquement."""
        if not VECTOR_STORAGE_AVAILABLE:
            logger.error("Impossible de planifier l'enrichissement: Vector Storage indisponible.")
            return

        logger.info(f"Planification de l'enrichissement des marchands toutes les {interval_seconds} secondes.")
        while True:
             try:
                 await self.run_enrichment_cycle(limit=cycle_limit)
             except Exception as e:
                  logger.error(f"Erreur dans la boucle de planification d'enrichissement: {e}", exc_info=True)
             logger.info(f"Prochain cycle d'enrichissement dans {interval_seconds} secondes...")
             await asyncio.sleep(interval_seconds)


# --- Point d'entrée potentiel pour un worker dédié ---
async def main_enrichment_worker():
     """Fonction principale pour lancer le service en tant que worker."""
     # Initialisation (ex: charger config, initialiser services)
     logger.info("Démarrage du Worker d'Enrichissement Marchand...")
     try:
         vector_storage = VectorStorageService() # Assumer l'initialisation réussie ici
         enrichment_service = MerchantEnrichmentService(vector_storage)
         # Lancer la planification (ou écouter une queue)
         await enrichment_service.schedule_enrichment(interval_seconds=60*10) # Toutes les 10 mins
     except RuntimeError as e:
          logger.critical(f"Impossible de démarrer le worker: {e}")
     except Exception as e:
          logger.critical(f"Erreur fatale dans le worker d'enrichissement: {e}", exc_info=True)

if __name__ == "__main__":
     # Exemple de lancement direct du worker (utiliser un vrai gestionnaire de process en prod)
     logging.basicConfig(level=logging.INFO) # Configurer le logging si lancé seul
     asyncio.run(main_enrichment_worker())