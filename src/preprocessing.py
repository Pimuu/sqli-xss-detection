"""Input canonicalisation, modelled on the transformation pipeline of WAFs.

Rule-based WAFs such as ModSecurity apply chained "transformation functions"
(`t:urlDecodeUni`, `t:htmlEntityDecode`, `t:jsDecode`, `t:lowercase`,
`t:replaceComments`, `t:compressWhitespace`) before matching signatures.
`canonicalize` reproduces that idea so the ML models can be evaluated with and
without normalisation.

Everything here is a *pure string-to-string* transform: decoded text is
returned as a Python `str` and is never interpreted, rendered or executed.
"""
from __future__ import annotations

import html
import re
from urllib.parse import unquote

_UNICODE_ESC = re.compile(r"\\u([0-9a-fA-F]{4})")
_HEX_ESC = re.compile(r"\\x([0-9a-fA-F]{2})")
_MYSQL_VERSIONED = re.compile(r"/\*!\d*\s*(.*?)\*/", re.S)   # /*!50000SELECT*/ -> SELECT
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_WHITESPACE = re.compile(r"\s+")


def _url_decode_loop(s: str, max_rounds: int = 3) -> str:
    """Percent-decode repeatedly to defeat double/triple encoding."""
    for _ in range(max_rounds):
        decoded = unquote(s)          # NOTE: '+' is kept literal on purpose
        if decoded == s:
            break
        s = decoded
    return s


def _js_decode(s: str) -> str:
    s = _UNICODE_ESC.sub(lambda m: chr(int(m.group(1), 16)), s)
    return _HEX_ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def canonicalize(s: str) -> str:
    """Return a normalised, lower-cased view of an HTTP parameter string."""
    s = _url_decode_loop(s)
    s = html.unescape(s)
    s = _js_decode(s)
    s = _url_decode_loop(s)                       # entities may hide %XX again
    s = _MYSQL_VERSIONED.sub(r" \1 ", s)          # keep code inside /*! ... */
    s = _BLOCK_COMMENT.sub(" ", s)                # /**/ used as whitespace
    s = _WHITESPACE.sub(" ", s)
    return s.strip().lower()


def identity_lower(s: str) -> str:
    """Minimal preprocessing used for the 'raw' ablation (case only)."""
    return s.lower()
