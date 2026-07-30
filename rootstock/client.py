"""
HTTP client for Rootstock backend API.

Handles pushing manifests and usage rollups to the central registry.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import UserConfig
from .manifest import Manifest


class RootstockClient:
    """Client for Rootstock backend API."""

    def __init__(self, config: UserConfig):
        """
        Initialize the client.

        Args:
            config: User config containing api_key and api_url
        """
        self.config = config

    def push_manifest(self, manifest: Manifest) -> tuple[bool, str]:
        """
        Push manifest to backend API.

        Args:
            manifest: Manifest to push

        Returns:
            (success, message) tuple
        """
        if not self.config.api_key:
            return False, "API key not configured"
        if not self.config.api_secret:
            return False, "API secret not configured"
        if not self.config.api_url:
            return False, "API URL not configured"

        return self._post(self.config.api_url, manifest.to_dict(), "Manifest pushed successfully")

    def push_usage(self, cluster: str, rows: list[dict]) -> tuple[bool, str]:
        """
        Push aggregated usage rollup rows to the backend usage endpoint.

        Args:
            cluster: Cluster name the rollups are filed under
            rows: Rollup rows as aggregated by ``summarize_spool`` — counts
                only, never the user hashes

        Returns:
            (success, message) tuple
        """
        if not self.config.api_key:
            return False, "API key not configured"
        if not self.config.api_secret:
            return False, "API secret not configured"
        url = self.config.resolve_usage_api_url()
        if not url:
            return False, "Usage API URL not configured (set usage_api_url)"

        return self._post(
            url,
            {"cluster": cluster, "rows": rows},
            f"Pushed {len(rows)} rollup row(s) to {url}",
        )

    def _post(self, url: str, payload: dict, success_message: str) -> tuple[bool, str]:
        api_key = self.config.api_key
        api_secret = self.config.api_secret
        if api_key is None or api_secret is None:  # callers check; keep _post self-contained
            return False, "API credentials not configured"

        data = json.dumps(payload).encode("utf-8")

        request = Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Modal-Key": api_key,
                "Modal-Secret": api_secret,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                if response.status in (200, 201, 204):
                    return True, success_message
                return False, f"Unexpected status: {response.status}"
        except HTTPError as e:
            return False, f"HTTP error {e.code}: {e.reason}"
        except URLError as e:
            return False, f"Connection error: {e.reason}"
        except TimeoutError:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Error: {e}"
