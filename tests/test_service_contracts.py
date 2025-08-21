"""Tests for the SearchServiceQuery contract."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class QueryMetadata:
    conversation_id: str
    user_id: int
    intent_type: str
    source_agent: str


@dataclass
class SearchParameters:
    query: str = ""
    limit: int = 50
    offset: int = 0


@dataclass
class SearchFilters:
    values: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchServiceQuery:
    query_metadata: QueryMetadata
    search_parameters: SearchParameters
    filters: SearchFilters

    def to_search_request(self) -> Dict[str, object]:
        return {
            "user_id": self.query_metadata.user_id,
            "query": self.search_parameters.query,
            "filters": self.filters.values,
            "limit": self.search_parameters.limit,
            "offset": self.search_parameters.offset,
            "metadata": {
                "conversation_id": self.query_metadata.conversation_id,
                "intent_type": self.query_metadata.intent_type,
                "source_agent": self.query_metadata.source_agent,
            },
        }


def test_to_search_request_minimal():
    query = SearchServiceQuery(
        query_metadata=QueryMetadata(
            conversation_id="conv1",
            user_id=123,
            intent_type="TEST_INTENT",
            source_agent="unit_test",
        ),
        search_parameters=SearchParameters(),
        filters=SearchFilters(),
    )

    expected = {
        "user_id": 123,
        "query": "",
        "filters": {},
        "limit": 50,
        "offset": 0,
        "metadata": {
            "conversation_id": "conv1",
            "intent_type": "TEST_INTENT",
            "source_agent": "unit_test",
        },
    }

    assert query.to_search_request() == expected
