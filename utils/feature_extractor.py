"""
feature_extractor.py
Extracts static (no network calls) features from a URL string.
All features are numeric so they can be fed directly to ML / NN models.
"""

import re
import math
from urllib.parse import urlparse, parse_qs


# Keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "confirm", "password", "paypal", "ebay", "amazon",
    "apple", "google", "microsoft", "support", "free", "lucky",
    "winner", "click", "here", "access", "wallet", "alert",
]

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_signs",
    "num_ampersands",
    "num_percent",
    "has_https",
    "has_ip_address",
    "num_subdomains",
    "domain_length",
    "path_length",
    "num_params",
    "num_fragments",
    "num_digits_in_domain",
    "num_suspicious_keywords",
    "has_port",
    "num_special_chars",
    "entropy",
    "tld_length",
    "num_digits",
    "double_slash_redirect",
]


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string."""
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _has_ip(url: str) -> int:
    """Return 1 if URL contains an IPv4 address in host position."""
    ipv4 = re.compile(
        r"((\d{1,3}\.){3}\d{1,3})"
    )
    parsed = urlparse(url)
    return int(bool(ipv4.match(parsed.hostname or "")))


def extract_features(url: str) -> dict:
    """
    Extract a fixed-length feature vector from a URL.
    Returns a dict {feature_name: value}.
    """
    url = url.strip()

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        parsed = urlparse("")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    scheme = parsed.scheme or ""

    # Subdomain count: parts minus TLD and second-level domain
    host_parts = hostname.split(".")
    num_subdomains = max(len(host_parts) - 2, 0)

    # TLD
    tld = host_parts[-1] if host_parts else ""

    # Suspicious keyword count (case-insensitive)
    url_lower = url.lower()
    num_suspicious = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    # Special characters beyond the structural ones
    special_chars = re.findall(r"[!#$%&'*+,;<=>?@\[\]^`{|}~]", url)

    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_signs": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_percent": url.count("%"),
        "has_https": int(scheme == "https"),
        "has_ip_address": _has_ip(url),
        "num_subdomains": num_subdomains,
        "domain_length": len(hostname),
        "path_length": len(path),
        "num_params": len(parse_qs(query)),
        "num_fragments": int(bool(fragment)),
        "num_digits_in_domain": sum(c.isdigit() for c in hostname),
        "num_suspicious_keywords": num_suspicious,
        "has_port": int(parsed.port is not None),
        "num_special_chars": len(special_chars),
        "entropy": round(_shannon_entropy(url), 4),
        "tld_length": len(tld),
        "num_digits": sum(c.isdigit() for c in url),
        "double_slash_redirect": int("//" in path),
    }
    return features


def features_to_list(url: str) -> list:
    """Return feature values as an ordered list matching FEATURE_NAMES."""
    f = extract_features(url)
    return [f[name] for name in FEATURE_NAMES]
