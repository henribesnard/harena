"""
Moteur de recherche Whoosh pour le service de recherche Harena.

Ce module implémente un moteur de recherche lexical basé sur la bibliothèque Whoosh,
offrant des fonctionnalités avancées d'indexation et de recherche pour les données financières.
"""
import logging
import os
import re
import uuid
from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path
import shutil
from datetime import datetime
import json

# Import Whoosh
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, NUMERIC, DATETIME
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.scoring import BM25F
from whoosh.query import Term, And, Or, Range as WhooshRange
from whoosh import scoring

logger = logging.getLogger(__name__)

# Configuration des poids par défaut pour les différents champs
DEFAULT_FIELD_WEIGHTS = {
    "description": 3.0,
    "clean_description": 3.5,
    "merchant_name": 4.0,
    "category": 2.0
}

# Définition du schéma Whoosh pour les transactions
TRANSACTION_SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    user_id=NUMERIC(stored=True),
    account_id=NUMERIC(stored=True),
    bridge_transaction_id=ID(stored=True),
    amount=NUMERIC(stored=True),
    currency_code=KEYWORD(stored=True),
    description=TEXT(stored=True),
    clean_description=TEXT(stored=True),
    merchant_name=TEXT(stored=True),
    category=TEXT(stored=True),
    transaction_date=STORED,  # Stocké mais pas indexé pour la recherche textuelle
    operation_type=KEYWORD(stored=True),
    is_recurring=STORED,
    content=STORED  # Pour stocker le document complet
)

class WhooshSearchEngine:
    """
    Moteur de recherche basé sur Whoosh avec support de BM25F.
    """
    
    def __init__(self, storage_dir: str = "./data/whoosh_indexes"):
        """
        Initialise le moteur de recherche Whoosh.
        
        Args:
            storage_dir: Répertoire pour stocker les index Whoosh
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistiques d'utilisation
        self.total_indexed_docs = 0
        self.total_users = 0
        
        # Charger les métadonnées existantes
        self.metadata = self._load_metadata()
        
        logger.info(f"Moteur Whoosh initialisé avec {len(self.metadata)} index utilisateurs")
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Charge les métadonnées des index existants.
        
        Returns:
            Dictionnaire des métadonnées par utilisateur
        """
        metadata = {}
        metadata_file = self.storage_dir / "metadata.json"
        
        try:
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Calculer les statistiques globales
                self.total_users = len(metadata)
                self.total_indexed_docs = sum(meta.get("document_count", 0) for meta in metadata.values())
            
            logger.debug(f"Métadonnées chargées: {self.total_users} utilisateurs, {self.total_indexed_docs} documents")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des métadonnées: {e}", exc_info=True)
        
        return metadata
    
    def _save_metadata(self):
        """Sauvegarde les métadonnées des index."""
        try:
            metadata_file = self.storage_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(self.metadata, f)
            
            logger.debug("Métadonnées sauvegardées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées: {e}", exc_info=True)
    
    def _get_user_index_path(self, user_id: int) -> Path:
        """
        Obtient le chemin du répertoire d'index pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Chemin du répertoire d'index
        """
        return self.storage_dir / f"user_{user_id}"
    
    def _get_or_create_index(self, user_id: int) -> Any:
        """
        Obtient ou crée l'index Whoosh pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Index Whoosh
        """
        str_user_id = str(user_id)
        index_dir = self._get_user_index_path(user_id)
        
        try:
            # Créer le répertoire si nécessaire
            if not index_dir.exists():
                index_dir.mkdir(parents=True, exist_ok=True)
                
                # Créer un nouvel index
                index = create_in(str(index_dir), TRANSACTION_SCHEMA)
                
                # Initialiser les métadonnées
                self.metadata[str_user_id] = {
                    "created_at": datetime.now().isoformat(),
                    "document_count": 0,
                    "updated_at": datetime.now().isoformat()
                }
                self._save_metadata()
                
                logger.info(f"Nouvel index créé pour l'utilisateur {user_id}")
                self.total_users += 1
                
                return index
            
            # Ouvrir l'index existant
            if exists_in(str(index_dir)):
                return open_dir(str(index_dir))
            
            # Si le répertoire existe mais pas l'index, le créer
            return create_in(str(index_dir), TRANSACTION_SCHEMA)
            
        except Exception as e:
            logger.error(f"Erreur lors de la création/ouverture de l'index pour l'utilisateur {user_id}: {e}", exc_info=True)
            raise
    
    def index_documents(self, user_id: int, documents: List[Dict[str, Any]]) -> bool:
        """
        Indexe un lot de documents pour un utilisateur spécifique.
        
        Args:
            user_id: ID de l'utilisateur
            documents: Liste de documents à indexer
            
        Returns:
            True si l'indexation a réussi, False sinon
        """
        str_user_id = str(user_id)
        logger.info(f"Indexation de {len(documents)} documents pour l'utilisateur {user_id}")
        
        try:
            # Obtenir ou créer l'index
            index = self._get_or_create_index(user_id)
            
            # Ouvrir un writer pour ajouter des documents
            writer = index.writer()
            
            # Ajouter chaque document
            for doc in documents:
                # Préparation des données
                doc_id = doc.get("id", str(uuid.uuid4()))
                
                # Nettoyage et conversion des champs spéciaux
                clean_doc = {
                    "id": doc_id,
                    "user_id": user_id,
                    "account_id": doc.get("account_id"),
                    "bridge_transaction_id": str(doc.get("bridge_transaction_id", "")),
                    "amount": float(doc.get("amount", 0.0)),
                    "currency_code": doc.get("currency_code", ""),
                    "description": doc.get("description", ""),
                    "clean_description": doc.get("clean_description", ""),
                    "merchant_name": doc.get("merchant_name", ""),
                    "category": doc.get("category", ""),
                    "transaction_date": doc.get("transaction_date", ""),
                    "operation_type": doc.get("operation_type", ""),
                    "is_recurring": doc.get("is_recurring", False),
                    "content": json.dumps(doc)  # Stocker le document complet pour extraction
                }
                
                # Ajouter à l'index
                writer.add_document(**clean_doc)
            
            # Valider les changements
            writer.commit()
            
            # Mettre à jour les métadonnées
            if str_user_id not in self.metadata:
                self.metadata[str_user_id] = {
                    "created_at": datetime.now().isoformat(),
                    "document_count": len(documents)
                }
                self.total_users += 1
            else:
                current_count = self.metadata[str_user_id].get("document_count", 0)
                self.metadata[str_user_id]["document_count"] = current_count + len(documents)
            
            self.metadata[str_user_id]["updated_at"] = datetime.now().isoformat()
            self._save_metadata()
            
            # Mettre à jour les statistiques
            self.total_indexed_docs += len(documents)
            
            logger.info(f"Indexation réussie pour l'utilisateur {user_id}: {len(documents)} documents ajoutés")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation pour l'utilisateur {user_id}: {e}", exc_info=True)
            return False
    
    def search(self, 
               user_id: int, 
               query_text: str, 
               field_weights: Optional[Dict[str, float]] = None,
               top_k: int = 50,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Effectue une recherche dans les documents d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query_text: Texte de la requête
            field_weights: Poids des champs (optionnel, utilise les poids par défaut sinon)
            top_k: Nombre de résultats à retourner
            filters: Filtres additionnels (optionnel)
            
        Returns:
            Liste des résultats de recherche
        """
        str_user_id = str(user_id)
        
        try:
            # Vérifier si l'index existe
            index_dir = self._get_user_index_path(user_id)
            if not index_dir.exists() or not exists_in(str(index_dir)):
                logger.warning(f"Aucun index trouvé pour l'utilisateur {user_id}")
                return []
            
            # Ouvrir l'index
            index = open_dir(str(index_dir))
            
            # Utiliser les poids par défaut si non spécifiés
            if field_weights is None:
                field_weights = DEFAULT_FIELD_WEIGHTS
            
            # Création du scoreur BM25F avec les poids personnalisés
            weighting = BM25F(field_weights)
            
            # Créer un parser multi-champs
            search_fields = list(field_weights.keys())
            parser = MultifieldParser(search_fields, schema=index.schema)
            
            # Parser la requête
            query = parser.parse(query_text)
            
            # Appliquer les filtres si présents
            if filters:
                filter_queries = []
                
                if "amount_range" in filters:
                    amount_range = filters["amount_range"]
                    if "min" in amount_range and amount_range["min"] is not None:
                        filter_queries.append(WhooshRange("amount", amount_range["min"], None))
                    if "max" in amount_range and amount_range["max"] is not None:
                        filter_queries.append(WhooshRange("amount", None, amount_range["max"]))
                
                if "date_range" in filters:
                    date_range = filters["date_range"]
                    if "start" in date_range and date_range["start"] is not None:
                        filter_queries.append(WhooshRange("transaction_date", date_range["start"].isoformat(), None))
                    if "end" in date_range and date_range["end"] is not None:
                        filter_queries.append(WhooshRange("transaction_date", None, date_range["end"].isoformat()))
                
                if "categories" in filters and filters["categories"]:
                    category_terms = [Term("category", cat) for cat in filters["categories"]]
                    filter_queries.append(Or(category_terms))
                
                if "operation_types" in filters and filters["operation_types"]:
                    op_terms = [Term("operation_type", op) for op in filters["operation_types"]]
                    filter_queries.append(Or(op_terms))
                
                # Ajouter un filtre pour l'ID utilisateur
                filter_queries.append(Term("user_id", str(user_id)))
                
                # Combiner tous les filtres avec la requête principale
                if filter_queries:
                    query = And([query] + filter_queries)
            
            # Effectuer la recherche
            with index.searcher(weighting=weighting) as searcher:
                results_raw = searcher.search(query, limit=top_k)
                
                # Construire les résultats formatés
                results = []
                for hit in results_raw:
                    # Reconstituer le document complet à partir du JSON stocké
                    try:
                        content = json.loads(hit["content"])
                    except (json.JSONDecodeError, KeyError):
                        # Fallback si le content n'est pas du JSON valide
                        content = {field: hit[field] for field in hit.keys() if field != "content"}
                    
                    # Créer l'objet résultat
                    result = {
                        "id": hit["id"],
                        "type": "transaction",  # Type par défaut
                        "content": content,
                        "score": hit.score,
                        "match_details": {
                            "lexical_score": hit.score
                        },
                        "highlight": self._extract_highlights(hit, query_text, search_fields)
                    }
                    results.append(result)
                
                logger.info(f"Recherche pour '{query_text}': {len(results)} résultats trouvés")
                return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche pour l'utilisateur {user_id}: {e}", exc_info=True)
            return []
    
    def _extract_highlights(self, hit, query_text, fields):
        """
        Extrait les passages pertinents pour le highlighting.
        Simple implémentation qui pourrait être améliorée en production.
        """
        highlights = {}
        query_terms = set(query_text.lower().split())
        
        for field in fields:
            if field in hit:
                text = hit[field]
                if not text or not isinstance(text, str):
                    continue
                
                # Implémentation naïve: extraire la phrase contenant un terme de la requête
                sentences = text.split('. ')
                matching_sentences = []
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    for term in query_terms:
                        if term in sentence_lower and sentence not in matching_sentences:
                            # Entourer le terme avec <em></em> pour le highlighting
                            highlighted = sentence
                            for term in query_terms:
                                term_pattern = re.compile(re.escape(term), re.IGNORECASE)
                                highlighted = term_pattern.sub(f"<em>{term}</em>", highlighted)
                            
                            matching_sentences.append(highlighted)
                
                if matching_sentences:
                    highlights[field] = matching_sentences
        
        return highlights if highlights else None
    
    def delete_user_index(self, user_id: int) -> bool:
        """
        Supprime l'index d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            True si la suppression a réussi, False sinon
        """
        str_user_id = str(user_id)
        
        try:
            # Supprimer le répertoire d'index
            index_dir = self._get_user_index_path(user_id)
            if index_dir.exists():
                shutil.rmtree(index_dir)
            
            # Mettre à jour les métadonnées
            if str_user_id in self.metadata:
                doc_count = self.metadata[str_user_id].get("document_count", 0)
                self.total_indexed_docs -= doc_count
                self.total_users -= 1
                
                del self.metadata[str_user_id]
                self._save_metadata()
            
            logger.info(f"Index supprimé pour l'utilisateur {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'index pour l'utilisateur {user_id}: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtient des statistiques sur les index.
        
        Returns:
            Dictionnaire contenant des statistiques
        """
        stats = {
            "total_users": self.total_users,
            "total_documents": self.total_indexed_docs,
            "users": self.metadata
        }
        
        return stats


# Instance globale du moteur Whoosh
_whoosh_engine = None

def get_whoosh_engine() -> WhooshSearchEngine:
    """
    Obtient l'instance singleton du moteur Whoosh.
    
    Returns:
        Instance du moteur Whoosh
    """
    global _whoosh_engine
    
    if _whoosh_engine is None:
        # Créer le répertoire de données si nécessaire
        data_dir = os.environ.get("WHOOSH_DATA_DIR", "./data/whoosh_indexes")
        _whoosh_engine = WhooshSearchEngine(storage_dir=data_dir)
    
    return _whoosh_engine