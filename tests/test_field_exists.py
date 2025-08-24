import pytest

from test_search_service import SearchServiceTester


def _mock_data():
    return {"results": [{"_score": 1.0, "highlights": {}}]}


def test_field_exists_returns_true_for_existing_fields():
    tester = SearchServiceTester()
    data = _mock_data()
    assert tester._field_exists(data, "results.0._score")
    assert tester._field_exists(data, "results.0.highlights")


def test_field_exists_returns_false_for_out_of_range_index():
    tester = SearchServiceTester()
    data = _mock_data()
    assert not tester._field_exists(data, "results.1._score")
