"""Example PDF processing skill for agentu."""

import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text


def extract_tables_from_pdf(pdf_path: str, page_number: int = 0) -> list:
    """
    Extract tables from a PDF page.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Which page to extract from (0-indexed)
        
    Returns:
        List of tables as list of rows
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            return []
        return pdf.pages[page_number].extract_tables()
