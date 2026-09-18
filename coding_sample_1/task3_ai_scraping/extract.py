"""
Purpose:    Turn each saved response into structured counts of brands/products
            and product features via a second Anthropic call. The model is asked
            for STRICT JSON, but models occasionally wrap it in ```json fences or
            add a stray sentence, so parsing is defensive (strip fences, slice to
            the outermost braces, try/except) and degrades to empty lists rather
            than crashing a 10-day run on one malformed reply.
Inputs:     outputs/promptNN_runNN_*.json files produced by collect.py.
Outputs:    The same JSON files, updated in place with brands, features,
            brand_count, feature_count.
Key Steps:  Build a strict-JSON extraction prompt -> call the extraction model ->
            safely parse -> write the four fields back into the response file.
How to Run: python main.py extract   (or run automatically during collect)
"""

import json
import logging
import re

from client import generate
from config import EXTRACTION_MAX_TOKENS, EXTRACTION_MODEL, OUTPUTS_DIR

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = (
    "You extract structured data from a shopping assistant's answer. "
    "Return ONLY a JSON object, no prose and no markdown code fences, with "
    'exactly this shape: {"brands": ["..."], "features": ["..."]}. '
    '"brands" lists distinct brand or product names mentioned (e.g. "Sony WH-1000XM5", "Toyota RAV4"). '
    '"features" lists distinct product features or attributes discussed '
    '(e.g. "battery life", "noise cancellation", "fuel economy"). '
    "Deduplicate entries and use [] when none are present."
)


def enrich_all(client, outputs_dir=OUTPUTS_DIR):
    """Add brand/feature data to every response file that is missing it.

    Re-running is safe: files already enriched are skipped, so an interrupted
    extraction pass resumes cleanly. Returns the number of files updated."""
    updated = 0
    for path in sorted(outputs_dir.glob("prompt*_run*.json")):
        if _enrich_file(client, path):
            updated += 1
    logger.info("Extraction pass complete: %d file(s) updated", updated)
    return updated


def _enrich_file(client, path):
    """Extract and store brands/features for one file. Returns True if written."""
    record = json.loads(path.read_text())
    if "brand_count" in record:
        return False  # already enriched

    brands, features = extract_brands_and_features(client, record["raw_response_text"])
    record.update(
        brands=brands,
        features=features,
        brand_count=len(brands),
        feature_count=len(features),
    )
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    logger.info("Extracted %s: %d brands, %d features", path.name, len(brands), len(features))
    return True


def extract_brands_and_features(client, response_text):
    """Return (brands, features) lists for one response via a strict-JSON call."""
    raw, _usage = generate(
        client,
        model=EXTRACTION_MODEL,
        prompt=f"Extract brands and features from this shopping answer:\n\n{response_text}",
        max_tokens=EXTRACTION_MAX_TOKENS,
        system=_EXTRACTION_SYSTEM,
    )
    parsed = _safe_parse_json(raw)
    return _string_list(parsed.get("brands")), _string_list(parsed.get("features"))


def _safe_parse_json(raw_text):
    """Parse model output that should be JSON but may carry fences or stray text.

    Returns {} on any failure so one bad reply can't abort the run."""
    cleaned = _strip_code_fences(raw_text).strip()
    candidate = _outermost_object(cleaned)
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse extraction JSON; storing empty lists. Raw: %.120r", raw_text)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _strip_code_fences(text):
    """Remove a surrounding ```json ... ``` (or plain ```) fence if present."""
    fence = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return fence.group(1) if fence else text


def _outermost_object(text):
    """Slice from the first '{' to the last '}' so leading/trailing prose is
    dropped; returns the text unchanged if no braces are found."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def _string_list(value):
    """Coerce a parsed field into a clean list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
