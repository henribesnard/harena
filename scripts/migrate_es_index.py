#!/usr/bin/env python
"""Reindex documents into the new Elasticsearch mapping.

This script creates a new index with the updated mapping and custom analyzers,
then reindexes all documents from the existing index into the new one.
"""
import asyncio
from typing import Any

from config_service.config import settings
from enrichment_service.storage.elasticsearch_client import ElasticsearchClient

NEW_INDEX_SUFFIX = "_v2"

async def migrate() -> None:
    old_index = settings.ELASTICSEARCH_INDEX
    new_index = f"{old_index}{NEW_INDEX_SUFFIX}"

    client = ElasticsearchClient()
    client.index_name = new_index
    await client.initialize()

    async with client.session.post(
        f"{client.base_url}/_reindex",
        json={"source": {"index": old_index}, "dest": {"index": new_index}},
    ) as resp:
        data: Any = await resp.json()
        print(f"Reindex response: {data}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(migrate())
