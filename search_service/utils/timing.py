"""
Utilitaires pour la mesure des performances.

Ce module fournit des outils pour mesurer et enregistrer les temps d'exécution
des différentes étapes du processus de recherche.
"""
import logging
import time
import contextlib
from typing import Dict, Optional
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Variables de contexte pour suivre les temps par requête
_timing_stack = ContextVar("timing_stack", default=[])
_request_timings = ContextVar("request_timings", default={})

@contextlib.contextmanager
def timer(label: str):
    """
    Context manager pour mesurer le temps d'exécution d'un bloc de code.
    
    Args:
        label: Étiquette pour identifier cette mesure
    """
    # Récupérer l'état actuel
    stack = _timing_stack.get()
    timings = _request_timings.get()
    
    # Commencer à mesurer
    start_time = time.time()
    stack.append(label)
    
    _timing_stack.set(stack)
    
    try:
        yield
    finally:
        # Calculer le temps écoulé
        elapsed_time = (time.time() - start_time) * 1000  # en millisecondes
        
        # Mettre à jour les timings avec une hiérarchie de clés imbriquées
        current_level = timings
        for idx, key in enumerate(stack):
            if idx == len(stack) - 1:  # dernier élément
                current_level[key] = elapsed_time
            else:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
        
        # Restaurer la pile
        stack.pop()
        _timing_stack.set(stack)
        _request_timings.set(timings)
        
        # Logger le timing
        logger.debug(f"Timing: {label} took {elapsed_time:.2f}ms")

def get_current_request_timings() -> Dict[str, float]:
    """
    Récupère les temps mesurés pour la requête en cours.
    
    Returns:
        Dictionnaire des temps mesurés
    """
    return _request_timings.get()

def reset_request_timings():
    """Réinitialise les temps mesurés pour la requête en cours."""
    _request_timings.set({})
    _timing_stack.set([])

def calculate_total_request_time() -> float:
    """
    Calcule le temps total d'exécution de la requête.
    
    Returns:
        Temps total en millisecondes
    """
    timings = _request_timings.get()
    total = 0.0
    
    # Sommer les temps de premier niveau
    for _, time_value in timings.items():
        if isinstance(time_value, (int, float)):
            total += time_value
    
    return total