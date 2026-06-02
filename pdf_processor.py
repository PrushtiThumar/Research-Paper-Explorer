"""
PDFProcessor: Handles extraction and cleaning of text from PDF research papers.
Supports local files and arXiv URLs.
"""

import re
import io
import logging
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    """Holds raw extracted content from a PDF."""
    raw_text: str
    pages: list[str]
    num_pages: int
    source: str  # file path or URL
    chunks: list[str] = field(default_factory=list)
    cleaned_text: str = ""


class PDFProcessor:
    """
    Extracts, cleans, and chunks text from PDF research papers.

    Supports:
    - Local PDF files
    - arXiv paper URLs (auto-downloads PDF)
    - Raw PDF bytes

    Chunk strategy: sliding window with configurable size and overlap.
    """

    ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
    ARXIV_ABS_BASE = "https://arxiv.org/abs/"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not PYMUPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError("Install PyMuPDF or pdfplumber: pip install PyMuPDF pdfplumber")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_file(self, path: str | Path) -> ExtractedDocument:
        """Process a local PDF file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        raw_bytes = path.read_bytes()
        doc = self._extract(raw_bytes, source=str(path))
        doc.cleaned_text = self._clean(doc.raw_text)
        doc.chunks = self._chunk(doc.cleaned_text)
        return doc

    def process_url(self, url: str) -> ExtractedDocument:
        """Process a PDF from a URL (including arXiv abstract/PDF links)."""
        pdf_url = self._resolve_arxiv_url(url)
        logger.info(f"Downloading PDF from: {pdf_url}")

        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()

        doc = self._extract(resp.content, source=pdf_url)
        doc.cleaned_text = self._clean(doc.raw_text)
        doc.chunks = self._chunk(doc.cleaned_text)
        return doc

    def process_bytes(self, data: bytes, source: str = "uploaded") -> ExtractedDocument:
        """Process raw PDF bytes (e.g., Streamlit file uploader)."""
        doc = self._extract(data, source=source)
        doc.cleaned_text = self._clean(doc.raw_text)
        doc.chunks = self._chunk(doc.cleaned_text)
        return doc

    # ------------------------------------------------------------------ #
    #  Extraction                                                          #
    # ------------------------------------------------------------------ #

    def _extract(self, data: bytes, source: str) -> ExtractedDocument:
        """Try PyMuPDF first, fall back to pdfplumber."""
        if PYMUPDF_AVAILABLE:
            return self._extract_pymupdf(data, source)
        return self._extract_pdfplumber(data, source)

    def _extract_pymupdf(self, data: bytes, source: str) -> ExtractedDocument:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return ExtractedDocument(
            raw_text="\n".join(pages),
            pages=pages,
            num_pages=len(pages),
            source=source,
        )

    def _extract_pdfplumber(self, data: bytes, source: str) -> ExtractedDocument:
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return ExtractedDocument(
            raw_text="\n".join(pages),
            pages=pages,
            num_pages=len(pages),
            source=source,
        )

    # ------------------------------------------------------------------ #
    #  Cleaning                                                            #
    # ------------------------------------------------------------------ #

    def _clean(self, text: str) -> str:
        """Remove noise common in academic PDFs."""
        # Remove references section
        text = re.split(r'\n\s*references\s*\n', text, flags=re.IGNORECASE)[0]

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Remove figure/table captions (heuristic)
        text = re.sub(r'(figure|fig\.|table)\s+\d+[:\.].*?\n', '', text, flags=re.IGNORECASE)

        # Remove lone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Collapse multiple spaces / blank lines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

        # Remove duplicate sentences (exact duplicates)
        sentences = text.split('. ')
        seen: set[str] = set()
        deduped: list[str] = []
        for s in sentences:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        text = '. '.join(deduped)

        return text.strip()

    # ------------------------------------------------------------------ #
    #  Chunking                                                            #
    # ------------------------------------------------------------------ #

    def _chunk(self, text: str) -> list[str]:
        """Sliding-window word chunker with overlap."""
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_arxiv_url(self, url: str) -> str:
        """Convert arXiv abstract URL to PDF URL if needed."""
        # Handle arxiv.org/abs/XXXX
        abs_match = re.search(r'arxiv\.org/abs/(\S+)', url)
        if abs_match:
            paper_id = abs_match.group(1).rstrip('/')
            return f"{self.ARXIV_PDF_BASE}{paper_id}.pdf"

        # Handle arxiv.org/pdf/XXXX (already PDF)
        if 'arxiv.org/pdf/' in url:
            return url if url.endswith('.pdf') else url + '.pdf'

        # Bare arXiv ID like "2301.00001"
        if re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', url.strip()):
            return f"{self.ARXIV_PDF_BASE}{url.strip()}.pdf"

        return url
