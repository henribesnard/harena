import argparse
import sys
from typing import Optional

import requests


class HarenaTestClient:
    """Simple client to run sequential API tests against a Harena deployment."""

    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.user_id: Optional[int] = None
        self.conversation_id: Optional[str] = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def login(self) -> None:
        resp = self.session.post(
            self._url("/users/auth/login"),
            data={"username": self.username, "password": self.password},
        )
        if resp.status_code != 200:
            raise AssertionError(f"Login failed: {resp.status_code} {resp.text}")
        token = resp.json().get("access_token")
        if not token:
            raise AssertionError("Missing access token in login response")
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def get_me(self) -> None:
        resp = self.session.get(self._url("/users/me"))
        if resp.status_code != 200:
            raise AssertionError(f"/users/me failed: {resp.status_code} {resp.text}")
        data = resp.json()
        self.user_id = data.get("id") or data.get("user_id") or data.get("userId")
        if not self.user_id:
            raise AssertionError("User id not found in /users/me response")

    def sync_user(self) -> None:
        resp = self.session.post(
            self._url(f"/enrichment/elasticsearch/sync-user/{self.user_id}")
        )
        if resp.status_code != 200:
            raise AssertionError(
                f"sync-user failed: {resp.status_code} {resp.text}"
            )

    def enrichment_health(self) -> None:
        resp = self.session.get(self._url("/enrichment/elasticsearch/health"))
        if resp.status_code != 200:
            raise AssertionError(
                f"enrichment health failed: {resp.status_code} {resp.text}"
            )
        if resp.json().get("status") == "error":
            raise AssertionError("Enrichment health returned error status")

    def search(self) -> None:
        payload = {"user_id": self.user_id, "query": ""}
        resp = self.session.post(self._url("/search/search"), json=payload)
        if resp.status_code != 200:
            raise AssertionError(f"search failed: {resp.status_code} {resp.text}")

    def conversation_health(self) -> None:
        resp = self.session.get(self._url("/conversation/health"))
        if resp.status_code != 200:
            raise AssertionError(
                f"conversation health failed: {resp.status_code} {resp.text}"
            )

    def conversation_status(self) -> None:
        resp = self.session.get(self._url("/conversation/status"))
        if resp.status_code != 200:
            raise AssertionError(
                f"conversation status failed: {resp.status_code} {resp.text}"
            )

    def conversation_chat(self) -> None:
        payload = {"message": "Hello from test"}
        resp = self.session.post(self._url("/conversation/chat"), json=payload)
        if resp.status_code != 200:
            raise AssertionError(
                f"conversation chat failed: {resp.status_code} {resp.text}"
            )
        self.conversation_id = resp.json().get("conversation_id")
        if not self.conversation_id:
            raise AssertionError("conversation_id missing in chat response")

    def conversation_metrics(self) -> None:
        resp = self.session.get(self._url("/conversation/metrics"))
        if resp.status_code != 200:
            raise AssertionError(
                f"conversation metrics failed: {resp.status_code} {resp.text}"
            )

    def conversation_turns(self) -> None:
        resp = self.session.get(
            self._url(
                f"/conversation/conversations/{self.conversation_id}/turns"
            )
        )
        if resp.status_code != 200:
            raise AssertionError(
                f"conversation turns failed: {resp.status_code} {resp.text}"
            )

    def run(self) -> None:
        self.login()
        self.get_me()
        self.sync_user()
        self.enrichment_health()
        self.search()
        self.conversation_health()
        self.conversation_status()
        self.conversation_chat()
        self.conversation_metrics()
        self.conversation_turns()


def main() -> None:
    parser = argparse.ArgumentParser(description="Harena nominal tests")
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    client = HarenaTestClient(args.base_url, args.username, args.password)
    try:
        client.run()
    except Exception as exc:  # noqa: BLE001
        print(f"Test sequence failed: {exc}")
        sys.exit(1)
    print("All tests passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
