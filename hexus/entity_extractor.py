import re
from typing import List, Dict

DEFAULT_PATTERNS = {
    "url": r'https?://[^\s<>"]+',
    "domain": r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
    "email": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    "file_path": r'(?<![\w.-])(?:/[\w.-]+)+\.\w+\b',
    "version": r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "docker_image": r'\b[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)?\b|\b[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+\b',
    "hostname": r'\b[a-zA-Z0-9-]+\.(?:local|lan|home|internal|mesh|host|node)\b|\b[a-zA-Z0-9]+-[a-zA-Z0-9-]+\b|\blocalhost\b',
}


class EntityExtractor:
    def __init__(self, patterns: Dict[str, str] = None, enabled: bool = True):
        self.enabled = enabled
        self.patterns = {**DEFAULT_PATTERNS, **(patterns or {})}
        self._compiled = {t: re.compile(p) for t, p in self.patterns.items()}

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        if not self.enabled or not text:
            return []

        extracted = []
        seen = set()

        # Extract elements matching our patterns.
        # Track spans to avoid extracting domains, hostnames, or other entities
        # from within URLs, while still allowing them to be extracted if they
        # appear elsewhere as standalone entities.
        url_spans = []
        if "url" in self._compiled:
            for match in self._compiled["url"].finditer(text):
                val = match.group(0)
                start, end = match.span()
                url_spans.append((start, end))
                if val not in seen:
                    seen.add(val)
                    extracted.append({"type": "url", "value": val})

        for entity_type, pattern in self._compiled.items():
            if entity_type == "url":
                continue
            for match in pattern.finditer(text):
                val = match.group(0)
                start, end = match.span()

                # Skip if this match is contained within any URL span
                if any(start >= u_start and end <= u_end for u_start, u_end in url_spans):
                    continue

                if val not in seen:
                    seen.add(val)
                    extracted.append({"type": entity_type, "value": val})

        return extracted
