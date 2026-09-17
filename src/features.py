"""
features.py
-----------
Extracts a 25-feature lexical feature vector from a raw URL string.
No external DNS / WHOIS lookups are required -- all features are derived
solely from parsing the URL string itself.

Feature groups
--------------
Length      : url_length, domain_length, path_length, query_length
Counts      : n_dots, n_hyphens, n_underscores, n_slashes, n_at,
              n_digits, n_special_chars
Structural  : has_ip, has_port, is_https, subdomain_level, path_depth
Entropy     : url_entropy, domain_entropy
Suspicious  : has_suspicious_word, n_suspicious_words
Shortening  : uses_shortener, has_redirect
TLD         : tld_is_common
Brand       : brand_in_subdomain, brand_in_path
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_WORDS = {
    "login", "verify", "verification", "account", "bank", "banking", "update",
    "secure", "security", "paypal", "ebay", "amazon", "apple", "microsoft",
    "netflix", "google", "signin", "password", "confirm", "confirmation",
    "wallet", "credential", "support", "billing", "invoice", "alert",
    "urgent", "suspended", "unauthorized", "access", "phish", "phishing",
    "portal", "auth", "authorize", "admin", "webscr", "payment", "claim",
    "reward", "gift", "recover", "session", "service", "helpdesk", "validate",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "buff.ly",
    "is.gd", "cli.gs", "pic.gd", "ht.ly", "lnkd.in", "su.pr",
    "twurl.nl", "snipurl.com", "short.to", "budurl.com", "ping.fm",
}

COMMON_TLDS = {"com", "org", "net", "edu", "gov", "uk", "us", "ca", "au"}

BRAND_KEYWORDS = {
    "paypal", "amazon", "google", "apple", "microsoft", "netflix",
    "facebook", "instagram", "twitter", "ebay", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "linkedin",
}

IP_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (bits)."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _count_special(s: str) -> int:
    """Count non-alphanumeric, non-common-punctuation characters."""
    return sum(1 for ch in s if not ch.isalnum() and ch not in ".-/_?=&#:%+@")


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """Return a feature dict for a single URL string.

    Parameters
    ----------
    url : str
        Raw URL (with or without scheme).

    Returns
    -------
    dict
        25 numeric features.
    """
    # Ensure scheme so urlparse works correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    ext = tldextract.extract(url)

    full_url = url
    domain = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # --- Length features ---
    url_length = len(full_url)
    domain_length = len(domain)
    path_length = len(path)
    query_length = len(query)

    # --- Count features ---
    n_dots = full_url.count(".")
    n_hyphens = full_url.count("-")
    n_underscores = full_url.count("_")
    n_slashes = full_url.count("/")
    n_at = full_url.count("@")
    n_digits = sum(ch.isdigit() for ch in full_url)
    n_special_chars = _count_special(full_url)

    # --- Structural features ---
    has_ip = int(bool(IP_RE.match(ext.domain + "." + ext.suffix if ext.suffix else ext.domain)))
    has_port = int(bool(parsed.port))
    is_https = int(parsed.scheme == "https")

    # subdomain_level: number of labels in subdomain
    subdomain_level = len(ext.subdomain.split(".")) if ext.subdomain else 0

    # path_depth: number of path segments
    path_depth = len([p for p in path.split("/") if p])

    # --- Entropy features ---
    url_entropy = _shannon_entropy(full_url)
    domain_entropy = _shannon_entropy(domain)

    # --- Suspicious word features ---
    url_lower = full_url.lower()
    matched_suspicious = [w for w in SUSPICIOUS_WORDS if w in url_lower]
    has_suspicious_word = int(len(matched_suspicious) > 0)
    n_suspicious_words = len(matched_suspicious)

    # --- Shortening / redirect features ---
    base_domain = (ext.domain + "." + ext.suffix).lower() if ext.suffix else ext.domain.lower()
    uses_shortener = int(base_domain in SHORTENER_DOMAINS)
    # Redirect: occurrence of "//" anywhere after the protocol part
    has_redirect = int("//" in full_url[7:])  # skip "http://"

    # --- TLD feature ---
    tld_is_common = int(ext.suffix.lower() in COMMON_TLDS) if ext.suffix else 0

    # --- Brand impersonation features ---
    subdomain_lower = ext.subdomain.lower() if ext.subdomain else ""
    path_lower = path.lower()
    brand_in_subdomain = int(any(b in subdomain_lower for b in BRAND_KEYWORDS))
    brand_in_path = int(any(b in path_lower for b in BRAND_KEYWORDS))

    return {
        # Length
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "query_length": query_length,
        # Counts
        "n_dots": n_dots,
        "n_hyphens": n_hyphens,
        "n_underscores": n_underscores,
        "n_slashes": n_slashes,
        "n_at": n_at,
        "n_digits": n_digits,
        "n_special_chars": n_special_chars,
        # Structural
        "has_ip": has_ip,
        "has_port": has_port,
        "is_https": is_https,
        "subdomain_level": subdomain_level,
        "path_depth": path_depth,
        # Entropy
        "url_entropy": url_entropy,
        "domain_entropy": domain_entropy,
        # Suspicious tokens
        "has_suspicious_word": has_suspicious_word,
        "n_suspicious_words": n_suspicious_words,
        # Shortening / redirect
        "uses_shortener": uses_shortener,
        "has_redirect": has_redirect,
        # TLD
        "tld_is_common": tld_is_common,
        # Brand
        "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path": brand_in_path,
    }


def extract_features_batch(urls: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """Extract features for a pandas Series of URL strings.

    Parameters
    ----------
    urls : pd.Series
        Series of raw URL strings.
    n_jobs : int
        Number of parallel jobs (uses joblib). -1 = all cores.

    Returns
    -------
    pd.DataFrame
        DataFrame with 25 feature columns.
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(extract_features)(url) for url in urls
    )
    return pd.DataFrame(results)


FEATURE_NAMES: list[str] = list(extract_features("http://example.com").keys())
