"""Web content sanitization for HydRAG web fallback."""

import html
import re

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMG_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}")


def _sanitize_web_content(
    raw: str,
    max_chars: int = 3000,
    allow_markdown: bool = False,
) -> str:
    """Sanitize web fallback content for safe consumption.

    Default (``allow_markdown=False``): strips all HTML, scripts, and
    markdown formatting, returning plain text only.

    When ``allow_markdown=True``: preserves markdown formatting but
    still strips HTML tags and script/style blocks.
    """
    text = _SCRIPT_STYLE_RE.sub("", raw)
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)

    if not allow_markdown:
        text = _MARKDOWN_IMG_RE.sub(r"\1", text)
        text = _MARKDOWN_LINK_RE.sub(r"\1", text)
        text = _MARKDOWN_HEADING_RE.sub("", text)
        text = _MARKDOWN_BOLD_ITALIC_RE.sub(r"\1", text)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]
