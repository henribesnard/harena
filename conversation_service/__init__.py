"""
Conversation Service MVP Package

Service de conversation intelligent basé sur AutoGen v0.4 et DeepSeek
pour le domaine financier avec détection d'intentions entièrement LLM.
"""

__version__ = "1.0.0"
__author__ = "Conversation Service Team"
from . import agents

# Expose commonly used subpackages when available
__all__ = ["__version__", "__author__", "agents"]
for _mod in ["api", "core", "models", "services", "utils"]:
    try:
        globals()[_mod] = __import__(f"{__name__}.{_mod}", fromlist=["*"])
    except Exception:
        continue
    else:
        __all__.append(_mod)
