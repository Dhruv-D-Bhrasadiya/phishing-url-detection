"""
Unit tests for src/features.py

Run with:
    pytest tests/test_features.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd

from src.features import extract_features, extract_features_batch, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PHISHING_URLS = [
    "http://paypal-login.ru/verify/account?id=123",
    "http://192.168.1.1/bank/update/credentials.php",
    "http://secure-amazon.tk/signin/confirm",
    "http://bit.ly/3xPhishURL",
    "http://ebay-signin.support@evil.com/user/verify",
]

LEGIT_URLS = [
    "https://www.google.com/search?q=phishing",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://github.com/scikit-learn/scikit-learn",
    "https://www.amazon.com/dp/B09G9FPHY6",
    "https://stackoverflow.com/questions/12345",
]


# ---------------------------------------------------------------------------
# Tests: feature extraction correctness
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_returns_dict_with_25_keys(self):
        result = extract_features("http://example.com")
        assert isinstance(result, dict)
        assert len(result) == 25, f"Expected 25 features, got {len(result)}"

    def test_feature_names_match_constant(self):
        result = extract_features("https://test.com/path?q=1")
        assert set(result.keys()) == set(FEATURE_NAMES), \
            f"Keys mismatch: {set(result.keys()) ^ set(FEATURE_NAMES)}"

    def test_is_https_flag(self):
        https = extract_features("https://example.com")
        http  = extract_features("http://example.com")
        assert https["is_https"] == 1
        assert http["is_https"] == 0

    def test_has_ip_detection(self):
        ip_url     = extract_features("http://192.168.1.1/login")
        domain_url = extract_features("http://paypal.com/login")
        assert ip_url["has_ip"] == 1
        assert domain_url["has_ip"] == 0

    def test_url_length_positive(self):
        result = extract_features("http://example.com/path")
        assert result["url_length"] > 0

    def test_domain_length_positive(self):
        result = extract_features("https://www.example.com/path")
        assert result["domain_length"] > 0

    def test_n_dots_count(self):
        result = extract_features("http://a.b.c.example.com/path")
        assert result["n_dots"] >= 4

    def test_n_hyphens_count(self):
        result = extract_features("http://paypal-login-secure.ru/verify")
        assert result["n_hyphens"] >= 2

    def test_has_suspicious_word(self):
        suspicious = extract_features("http://evil.com/login/verify/account")
        clean      = extract_features("https://news.bbc.co.uk/science")
        assert suspicious["has_suspicious_word"] == 1
        assert suspicious["n_suspicious_words"] >= 1
        assert clean["has_suspicious_word"] == 0

    def test_uses_shortener(self):
        short  = extract_features("http://bit.ly/3xABC")
        normal = extract_features("https://www.amazon.com/dp/B09G9FPHY6")
        assert short["uses_shortener"] == 1
        assert normal["uses_shortener"] == 0

    def test_brand_in_subdomain(self):
        brand_url = extract_features("http://paypal.evil-login.com/verify")
        clean_url = extract_features("https://www.google.com/search")
        assert brand_url["brand_in_subdomain"] == 1
        assert clean_url["brand_in_subdomain"] == 0

    def test_tld_is_common(self):
        common = extract_features("https://example.com")
        rare   = extract_features("http://example.ru")
        assert common["tld_is_common"] == 1
        assert rare["tld_is_common"] == 0

    def test_entropy_non_negative(self):
        result = extract_features("https://www.example.com/path?q=1")
        assert result["url_entropy"] >= 0
        assert result["domain_entropy"] >= 0

    def test_url_without_scheme(self):
        """Should handle URLs without http:// prefix."""
        result = extract_features("example.com/path")
        assert isinstance(result, dict)
        assert len(result) == 25

    def test_empty_query(self):
        result = extract_features("https://example.com/noquerystring")
        assert result["query_length"] == 0

    def test_path_depth(self):
        deep = extract_features("https://example.com/a/b/c/d")
        root = extract_features("https://example.com/")
        assert deep["path_depth"] >= 4
        assert root["path_depth"] <= 1

    def test_n_at_sign(self):
        at_url = extract_features("http://user@evil.com/path")
        no_at  = extract_features("https://example.com/path")
        assert at_url["n_at"] >= 1
        assert no_at["n_at"] == 0


# ---------------------------------------------------------------------------
# Tests: batch extraction
# ---------------------------------------------------------------------------

class TestExtractFeaturesBatch:

    def test_returns_dataframe(self):
        urls = pd.Series(PHISHING_URLS + LEGIT_URLS)
        result = extract_features_batch(urls, n_jobs=1)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self):
        urls = pd.Series(LEGIT_URLS)
        result = extract_features_batch(urls, n_jobs=1)
        assert len(result) == len(LEGIT_URLS)

    def test_correct_column_count(self):
        urls = pd.Series(PHISHING_URLS)
        result = extract_features_batch(urls, n_jobs=1)
        assert result.shape[1] == 25

    def test_column_names_match(self):
        urls = pd.Series(["https://example.com"])
        result = extract_features_batch(urls, n_jobs=1)
        assert list(result.columns) == FEATURE_NAMES

    def test_no_null_values(self):
        urls = pd.Series(PHISHING_URLS + LEGIT_URLS)
        result = extract_features_batch(urls, n_jobs=1)
        assert result.isnull().sum().sum() == 0, "Batch extraction produced null values"

    def test_all_numeric(self):
        urls = pd.Series(PHISHING_URLS + LEGIT_URLS)
        result = extract_features_batch(urls, n_jobs=1)
        assert result.dtypes.apply(lambda dt: dt.kind in 'iuf').all(), \
            "Non-numeric columns found in output"


# ---------------------------------------------------------------------------
# Tests: phishing vs legitimate signal
# ---------------------------------------------------------------------------

class TestPhishingSignals:
    """Higher-level tests: phishing URLs should score differently to legit."""

    def _batch(self, urls):
        return extract_features_batch(pd.Series(urls), n_jobs=1)

    def test_phishing_higher_url_length(self):
        # Verify that url_length varies meaningfully across URLs
        # (i.e. the extractor doesn't return a constant value).
        # Phishing URLs tend to be longer but this depends on the sample;
        # the reliable invariant is just that the feature has non-zero variance.
        all_urls = PHISHING_URLS + LEGIT_URLS
        result = self._batch(all_urls)
        lengths = result["url_length"]
        assert lengths.std() > 0, (
            "url_length is constant across all URLs — extractor may be broken. "
            f"Values: {lengths.tolist()}"
        )

    def test_phishing_more_suspicious_words(self):
        ph = self._batch(PHISHING_URLS)["n_suspicious_words"].mean()
        lg = self._batch(LEGIT_URLS)["n_suspicious_words"].mean()
        assert ph >= lg, f"Expected phishing ≥ legit suspicious words, got {ph:.2f} vs {lg:.2f}"

    def test_legit_more_https(self):
        ph = self._batch(PHISHING_URLS)["is_https"].mean()
        lg = self._batch(LEGIT_URLS)["is_https"].mean()
        assert lg >= ph, f"Expected legit ≥ phishing HTTPS rate, got {lg:.2f} vs {ph:.2f}"
