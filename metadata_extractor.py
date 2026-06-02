"""
MetadataExtractor: Parses title, authors, abstract, year, and keywords
from raw PDF text using heuristics and regex patterns.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured metadata for a research paper."""
    title: str = "Unknown Title"
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    keywords: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "keywords": self.keywords,
            "source": self.source,
        }

    def display_name(self) -> str:
        year_str = f" ({self.year})" if self.year else ""
        return f"{self.title}{year_str}"


class MetadataExtractor:
    """
    Extracts structured metadata from the first ~2000 characters of paper text.

    Heuristics used:
    - Title: first non-trivial line before authors / abstract
    - Authors: lines before abstract that look like name lists
    - Abstract: text between "Abstract" heading and next section
    - Year: four-digit year found in header region
    - Keywords: explicit keyword lines or extracted from abstract
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, text: str, source: str = "") -> PaperMetadata:
        """
        Extract metadata from full paper text.

        Args:
            text: Full cleaned document text.
            source: File path or URL for reference.

        Returns:
            PaperMetadata instance.
        """
        header = text[:3000]  # Focus on front matter

        meta = PaperMetadata(source=source)
        meta.title = self._extract_title(header)
        meta.authors = self._extract_authors(header)
        meta.abstract = self._extract_abstract(header)
        meta.year = self._extract_year(header)
        meta.keywords = self._extract_keywords(header)

        return meta

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_title(self, header: str) -> str:
        lines = [l.strip() for l in header.splitlines() if l.strip()]

        # Skip lines that are clearly not titles
        skip_patterns = re.compile(
            r'^(abstract|introduction|keywords|email|received|published|doi|arxiv|preprint|'
            r'volume|issue|journal|proceedings|\d{4}|©|copyright|\*)',
            re.IGNORECASE,
        )

        candidates: list[str] = []
        for line in lines[:20]:
            if skip_patterns.match(line):
                continue
            if len(line) < 8 or len(line) > 200:
                continue
            # Titles usually have mixed case, not all-caps (typically)
            candidates.append(line)

        if candidates:
            # Prefer the longest candidate in the first 5 plausible lines
            return max(candidates[:5], key=len)

        return lines[0] if lines else "Unknown Title"

    def _extract_authors(self, header: str) -> list[str]:
        """Look for author-like lines after the title and before abstract."""
        abstract_pos = self._find_abstract_start(header)
        search_area = header[:abstract_pos] if abstract_pos else header[:800]

        lines = [l.strip() for l in search_area.splitlines() if l.strip()]

        author_pattern = re.compile(
            r'^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+'  # "First [M.] Last"
        )
        # Also handle "A. Einstein, N. Bohr" style
        comma_names = re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)+)')

        authors: list[str] = []
        for line in lines[1:15]:  # skip title line
            if re.search(r'abstract|introduction|keywords', line, re.IGNORECASE):
                break
            # Comma-separated list
            m = comma_names.search(line)
            if m:
                parts = [p.strip() for p in m.group().split(',')]
                authors.extend(p for p in parts if author_pattern.match(p))
                continue
            if author_pattern.match(line):
                authors.append(line)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for a in authors:
            if a not in seen:
                seen.add(a)
                unique.append(a)

        return unique[:10]  # Cap at 10

    def _extract_abstract(self, header: str) -> str:
        pos = self._find_abstract_start(header)
        if pos is None:
            return ""

        abstract_text = header[pos:]
        # Take text up to next major section heading
        section_end = re.search(
            r'\n\s*(1\.?\s+Introduction|1\s+Introduction|Keywords|Index Terms)',
            abstract_text, re.IGNORECASE
        )
        if section_end:
            abstract_text = abstract_text[:section_end.start()]

        # Remove the "Abstract" heading itself
        abstract_text = re.sub(r'^abstract[\s:—–-]*', '', abstract_text, flags=re.IGNORECASE)
        return abstract_text.strip()[:1500]

    def _find_abstract_start(self, text: str) -> Optional[int]:
        m = re.search(r'\babstract\b', text, re.IGNORECASE)
        if m:
            # Move past the heading word
            return m.end()
        return None

    def _extract_year(self, header: str) -> Optional[int]:
        # Look for 4-digit year in range 1950–2030
        matches = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', header)
        if matches:
            years = [int(y) for y in matches]
            # Return the most recent plausible year in the header
            return max(y for y in years if 1950 <= y <= 2030)
        return None

    def _extract_keywords(self, header: str) -> list[str]:
        # Explicit keyword line
        kw_match = re.search(
            r'(?:keywords?|index terms?|key terms?)[\s:—–-]+(.+?)(?:\n\n|\n[A-Z])',
            header, re.IGNORECASE | re.DOTALL
        )
        if kw_match:
            raw = kw_match.group(1)
            # Split on commas, semicolons, or bullets
            keywords = re.split(r'[,;•·]', raw)
            cleaned = [k.strip().lower() for k in keywords if 2 < len(k.strip()) < 50]
            return cleaned[:15]

        return []
