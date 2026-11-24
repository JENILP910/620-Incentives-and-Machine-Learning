"""Shared query/ad feature extraction for CTR training and serving."""
import re
import numpy as np

CATEGORY_TERMS = {
    "laptop": ["laptop", "notebook"], "smartphone": ["phone", "smartphone", "mobile"],
    "tablet": ["tablet", "ipad"], "headphones": ["headphone", "earphone", "headset"],
    "monitor": ["monitor", "display", "screen"], "camera": ["camera"],
    "keyboard": ["keyboard"], "mouse": ["mouse"], "printer": ["printer"],
    "external_ssd": ["external ssd", "portable ssd"], "storage": ["usb drive", "pendrive", "flash drive", "memory card"],
    "computer_component": ["ram", "ssd", "computer component", "pc component"],
    "power_bank": ["power bank", "powerbank"], "router": ["router", "wifi"],
    "smartwatch": ["smartwatch", "smart watch"], "accessory": ["accessory", "laptop stand", "usb hub"],
}


def normalize_query(query):
    q = query.lower()
    q = re.sub(r"\bai\s*/\s*ml\b|\bai\s+ml\b", "machine learning", q)
    q = re.sub(r"\bml\b", "machine learning", q)
    q = re.sub(r"\blightweight\b", "portable lightweight", q)
    q = re.sub(r"\bgood battery\b", "long battery life", q)
    return q


def parse_budget(query):
    q = query.lower().replace(",", "")
    patterns = [
        (r'(?:under|below|less than|within|budget of|upto|up to)\s*(?:₹|rs\.?\s*)?([\d.]+)\s*(lakh|lac|k)?', 1),
        (r'(?:₹|rs\.?\s*)([\d.]+)\s*(lakh|lac|k)?', 1),
    ]
    for pattern, _ in patterns:
        m = re.search(pattern, q)
        if m:
            value = float(m.group(1))
            unit = (m.group(2) or "").lower()
            if unit in ("lakh", "lac"):
                value *= 100000
            elif unit == "k":
                value *= 1000
            return value
    return None


def detect_category(query):
    q = query.lower()
    for category, terms in CATEGORY_TERMS.items():
        if any(term in q for term in terms):
            return category
    return None


def explicit_constraints(query):
    q = query.lower()
    constraints = {}
    ram = re.search(r'(\d+)\s*gb\s*ram', q)
    if ram:
        constraints["ram_gb"] = int(ram.group(1))
    storage = re.search(r'(\d+)\s*(tb|gb)\s*(?:ssd|storage)', q)
    if storage:
        amount = int(storage.group(1)) * (1024 if storage.group(2) == "tb" else 1)
        constraints["storage_gb"] = amount
    return constraints


def parse_ad_specs(row):
    text = f"{row.get('product', '')} {row.get('description', '')}".lower()
    ram = re.search(r'(\d+)\s*gb\s*ram', text)
    storage = re.search(r'(\d+)\s*(tb|gb)\s*(?:nvme\s*)?(?:ssd|storage)', text)
    return {
        "ram_gb": int(ram.group(1)) if ram else None,
        "storage_gb": int(storage.group(1)) * (1024 if storage and storage.group(2) == "tb" else 1) if storage else None,
    }


def lexical_overlap(query, text):
    stop = {"a", "an", "the", "for", "with", "under", "below", "and", "to", "of", "in", "on", "good", "best"}
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1 and t not in stop}
    d_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q_tokens & d_tokens) / max(1, len(q_tokens))


def build_numeric_features(query, row, similarity):
    budget = parse_budget(query)
    category = detect_category(query)
    text = f"{row.get('product', '')} {row.get('category', '')} {row.get('description', '')}"
    price = float(row.get("price_inr", 0) or 0)
    budget_fit = 1.0 if budget is None else float(price <= budget)
    price_ratio = min(price / budget, 3.0) if budget else 0.0
    category_match = float(category is not None and str(row.get("category", "")).lower() == category)
    overlap = lexical_overlap(normalize_query(query), text)
    return np.array([similarity, category_match, overlap, budget_fit, price_ratio, np.log1p(price)], dtype=float)
