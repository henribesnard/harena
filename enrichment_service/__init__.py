"""Enrichment and indexing service for Harena (Elasticsearch only)."""

__version__ = "2.0.0-elasticsearch"

# Re-export frequently used utilities so they can be imported directly from
# ``enrichment_service``.  This mirrors the old API where caches lived in a
# top-level ``account_enrichment_service`` package.
from .cache import AccountLRUCache, MerchantCategoryCache

__all__ = ["__version__", "AccountLRUCache", "MerchantCategoryCache"]
