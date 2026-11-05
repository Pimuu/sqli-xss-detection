"""Feature extraction: character n-gram TF-IDF and hand-crafted lexical features.

Two complementary views of a request string are provided:

* `build_tfidf` - a bag of character n-grams (1-4) with sublinear TF-IDF
  weighting. It learns attack *syntax* (e.g. "' or", "union se", "onerr",
  "<scr") without any manual rules.
* `HandcraftedFeatures` - 28 interpretable statistics distilled from WAF
  signatures and prior work (length, special-character ratios, entropy,
  keyword hits, structural regexes). It is compact, fast and explainable.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import canonicalize, identity_lower

# --------------------------------------------------------------------------- #
# Keyword lexicons (matched on word boundaries in the lower-cased string)
# --------------------------------------------------------------------------- #
SQL_KEYWORDS = ["select", "union", "insert", "update", "delete", "drop", "from",
                "where", "or", "and", "order by", "group by", "having", "sleep",
                "benchmark", "waitfor", "pg_sleep", "information_schema",
                "concat", "char", "substr", "substring", "ascii", "exec",
                "xp_cmdshell", "extractvalue", "updatexml", "null", "@@version",
                "table", "limit", "cast", "convert", "database", "user"]
XSS_KEYWORDS = ["script", "alert", "prompt", "confirm", "onerror", "onload",
                "onmouseover", "onfocus", "ontoggle", "javascript", "iframe",
                "svg", "img", "src", "document", "cookie", "eval", "fromcharcode",
                "expression", "constructor", "href", "body", "embed", "object",
                "style", "window", "location"]

_SQL_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in SQL_KEYWORDS) + r")\b")
_XSS_RE = re.compile(r"(?:" + "|".join(re.escape(k) for k in XSS_KEYWORDS) + r")")
_HTML_TAG = re.compile(r"<\s*/?\s*[a-z!][a-z0-9]*")
_EVENT_HANDLER = re.compile(r"\bon[a-z]{3,15}\s*=")
_JS_URI = re.compile(r"(?:java|vb)script\s*:")
_TAUTOLOGY = re.compile(r"(\b\d+\b|'[^']*')\s*(?:=|like)\s*\1|\bor\s+true\b|'\s*or\s*'")
_UNION_SELECT = re.compile(r"union(?:\s+all)?\s+select")
_SQL_COMMENT = re.compile(r"--|#|/\*")
_FUNC_CALL = re.compile(r"[a-z_]+\s*\(")
_PCT = re.compile(r"%[0-9a-fA-F]{2}")
_ENTITY = re.compile(r"&#x?[0-9a-fA-F]+;?|&[a-z]+;")
_ESCAPE = re.compile(r"\\[ux][0-9a-fA-F]{2,4}")
_NONALNUM_RUN = re.compile(r"[^a-z0-9\s]+")


def shannon_entropy(s: str) -> float:
    """Shannon entropy (bits / character) of the string's character histogram."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer mapping strings to a dense matrix of 28 features.

    Encoding indicators (percent-, entity-, escape-ratios, raw length, upper-
    case ratio) are always computed on the *raw* input, because encoding is a
    signal in itself. All other features are computed on the canonicalised
    string when ``canonical=True`` (default) or on the lower-cased raw string
    otherwise.
    """

    feature_names_ = [
        "len_raw", "len_canon", "n_tokens", "special_char_ratio", "digit_ratio",
        "upper_ratio_raw", "whitespace_ratio", "entropy", "pct_encoded_ratio",
        "html_entity_count", "js_escape_count", "n_single_quote",
        "n_double_quote", "n_angle_brackets", "n_parentheses", "n_equals",
        "n_semicolon", "n_sql_comment", "sql_keyword_count",
        "xss_keyword_count", "has_html_tag", "has_event_handler", "has_js_uri",
        "has_tautology", "has_union_select", "n_func_calls",
        "max_nonalnum_run", "quote_imbalance",
    ]

    def __init__(self, canonical: bool = True):
        self.canonical = canonical

    def fit(self, X, y=None):           # stateless
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)

    def _one(self, raw: str) -> list[float]:
        s = canonicalize(raw) if self.canonical else identity_lower(raw)
        n = max(len(s), 1)
        n_raw = max(len(raw), 1)
        runs = _NONALNUM_RUN.findall(s)
        return [
            len(raw),
            len(s),
            len(s.split()),
            sum(not (c.isalnum() or c.isspace()) for c in s) / n,
            sum(c.isdigit() for c in s) / n,
            sum(c.isupper() for c in raw) / n_raw,
            sum(c.isspace() for c in s) / n,
            shannon_entropy(s),
            3 * len(_PCT.findall(raw)) / n_raw,
            len(_ENTITY.findall(raw)),
            len(_ESCAPE.findall(raw)),
            s.count("'"),
            s.count('"'),
            s.count("<") + s.count(">"),
            s.count("(") + s.count(")"),
            s.count("="),
            s.count(";"),
            len(_SQL_COMMENT.findall(s)),
            len(_SQL_RE.findall(s)),
            len(_XSS_RE.findall(s)),
            float(bool(_HTML_TAG.search(s))),
            float(bool(_EVENT_HANDLER.search(s))),
            float(bool(_JS_URI.search(s))),
            float(bool(_TAUTOLOGY.search(s))),
            float(bool(_UNION_SELECT.search(s))),
            len(_FUNC_CALL.findall(s)),
            max((len(r) for r in runs), default=0),
            float(s.count("'") % 2 == 1),
        ]

    def transform(self, X):
        return np.asarray([self._one(str(x)) for x in X], dtype=np.float64)


def build_tfidf(canonical: bool = True, ngram_range=(1, 4),
                max_features: int = 50_000) -> TfidfVectorizer:
    """Character n-gram TF-IDF vectoriser.

    `analyzer="char"` (not "char_wb") is used on purpose: attack syntax spans
    word boundaries (e.g. "1=1--", "'><scr"), so n-grams must cross spaces.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        preprocessor=canonicalize if canonical else identity_lower,
        lowercase=False,          # the preprocessor already lower-cases
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
