from typing import Any, Dict, Optional

REQUEST_TIMEOUT = 5

class HarenaTestClient:
    """Minimal HTTP client used for testing timeout behaviour."""

    def __init__(self, base_url: str, session: Any):
        self.base_url = base_url.rstrip('/')
        self.session = session

    def _make_request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{endpoint}"
        return self.session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)

    def get(self, endpoint: str, **kwargs: Any) -> Any:
        return self._make_request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return self._make_request("POST", endpoint, json=json, **kwargs)
