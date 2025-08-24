"""Index template and ILM policy management for Elasticsearch indices."""
import copy
import logging
from typing import Any, Dict

logger = logging.getLogger("enrichment_service.index_management")

ILM_POLICY_NAME = "harena_transactions_policy"
INDEX_TEMPLATE_NAME = "harena_transactions_template"

ILM_POLICY: Dict[str, Any] = {
    "policy": {
        "phases": {
            "hot": {
                "actions": {
                    "rollover": {
                        "max_size": "50gb",
                        "max_age": "30d"
                    }
                }
            },
            "cold": {
                "min_age": "60d",
                "actions": {
                    "set_priority": {"priority": 0}
                }
            },
            "delete": {
                "min_age": "180d",
                "actions": {"delete": {}}
            }
        }
    }
}

INDEX_TEMPLATE: Dict[str, Any] = {
    "index_patterns": ["harena_transactions*"],
    "template": {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "max_result_window": 10000
            },
            "index.lifecycle.name": ILM_POLICY_NAME,
            "index.lifecycle.rollover_alias": "harena_transactions"
        },
        "mappings": {
            "properties": {
                # Identifiants
                "user_id": {"type": "integer"},
                "transaction_id": {"type": "keyword"},
                "account_id": {"type": "integer"},

                # Contenu recherchable
                "searchable_text": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "primary_description": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "merchant_name": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },

                # Données financières
                "amount": {"type": "float"},
                "amount_abs": {"type": "float"},
                "transaction_type": {"type": "keyword"},
                "currency_code": {"type": "keyword"},

                # Dates
                "date": {"type": "date"},
                "transaction_date": {"type": "date"},
                "month_year": {"type": "keyword"},
                "weekday": {"type": "keyword"},

                # Catégorisation
                "category_id": {"type": "integer"},
                "category_name": {"type": "keyword"},
                "operation_type": {"type": "keyword"},

                # Flags
                "is_future": {"type": "boolean"},
                "is_deleted": {"type": "boolean"},

                # Métadonnées
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"}
            }
        }
    },
    "priority": 500
}

async def _create_ilm_policy(session: Any, base_url: str) -> None:
    """Create or update ILM policy."""
    url = f"{base_url}/_ilm/policy/{ILM_POLICY_NAME}"
    async with session.put(url, json=ILM_POLICY) as response:
        if response.status not in (200, 201):
            text = await response.text()
            raise Exception(text)
        logger.info("ILM policy ensured")

async def _create_index_template(
    session: Any, base_url: str, ilm_enabled: bool = True
) -> None:
    """Create or update index template.

    If ``ilm_enabled`` is ``False`` the ILM settings are stripped before
    sending the template to Elasticsearch so that index management works on
    clusters without ILM support.
    """
    template = copy.deepcopy(INDEX_TEMPLATE)
    if not ilm_enabled:
        settings = template["template"]["settings"]
        settings.pop("index.lifecycle.name", None)
        settings.pop("index.lifecycle.rollover_alias", None)

    url = f"{base_url}/_index_template/{INDEX_TEMPLATE_NAME}"
    async with session.put(url, json=template) as response:
        if response.status not in (200, 201):
            text = await response.text()
            logger.error(
                f"Failed to create index template: {response.status} - {text}"
            )
            raise Exception(f"Failed to create index template: {text}")
        logger.info("Index template ensured")

async def ensure_template_and_policy(session: Any, base_url: str) -> None:
    """Ensure both ILM policy and index template exist.

    If ILM policy creation fails (for instance because the cluster does not
    support ILM or because auto creation of the ``_ilm`` index is forbidden),
    the service logs a warning and continues by creating the index template
    without ILM settings.
    """
    ilm_enabled = True
    try:
        await _create_ilm_policy(session, base_url)
    except Exception as exc:  # pragma: no cover - small utility method
        logger.warning("ILM disabled: %s", exc)
        ilm_enabled = False

    await _create_index_template(session, base_url, ilm_enabled=ilm_enabled)
