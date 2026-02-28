"""
URL fetch handler:
  read_url
"""

from __future__ import annotations

import asyncio
import re

from nvagent.tools.handlers import BaseHandler

# httpx is preferred for async HTTP; fall back to urllib if unavailable
try:
    import httpx as _httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

# trafilatura for better HTML content extraction
try:
    import trafilatura as _trafilatura

    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False


class UrlHandler(BaseHandler):
    """Handles read_url."""

    async def read_url(self, url: str, max_chars: int = 8000) -> str:
        try:
            if _HTTPX_AVAILABLE:
                async with _httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=20.0,
                    headers={"User-Agent": "nvagent/1.0 (coding assistant)"},
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    raw_html = resp.text
            else:
                loop = asyncio.get_event_loop()
                raw_html = await loop.run_in_executor(None, self._fetch_url_urllib, url)

            if _TRAFILATURA_AVAILABLE:
                extracted = _trafilatura.extract(
                    raw_html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
                content = extracted or raw_html
            else:
                content = re.sub(r"<style[^>]*>.*?</style>", "", raw_html, flags=re.DOTALL)
                content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"&nbsp;", " ", content)
                content = re.sub(r"&amp;", "&", content)
                content = re.sub(r"&lt;", "<", content)
                content = re.sub(r"&gt;", ">", content)
                content = re.sub(r"\s+", " ", content).strip()

            if len(content) > max_chars:
                content = content[:max_chars] + f"\n... [{len(content) - max_chars} more chars]"

            return f"[URL: {url}]\n{content}"
        except Exception as e:
            return f"Error fetching {url}: {type(e).__name__}: {e}"

    def _fetch_url_urllib(self, url: str) -> str:
        """Fallback synchronous URL fetch using urllib."""
        from urllib.request import urlopen  # type: ignore

        with urlopen(url, timeout=15) as resp:
            charset = "utf-8"
            ct = resp.headers.get_content_charset()
            if ct:
                charset = ct
            return resp.read().decode(charset, errors="replace")
