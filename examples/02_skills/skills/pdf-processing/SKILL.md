---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
---

# PDF Processing Skill

## Quick Start

Use pdfplumber to extract text from PDFs:

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    text = pdf.pages[0].extract_text()
    print(text)
```

## Capabilities

### 1. Text Extraction
Extract all text from a PDF document:

```python
def extract_all_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() or ""
        return full_text
```

### 2. Table Extraction
Extract tables from PDF pages. For detailed table extraction, see [[TABLES.md]].

Basic example:
```python
with pdfplumber.open("document.pdf") as pdf:
    tables = pdf.pages[0].extract_tables()
    for table in tables:
        print(table)
```

### 3. Form Filling
Fill PDF forms programmatically. For comprehensive form-filling guide, see [[FORMS.md]].

## Best Practices

- **Performance**: For large PDFs, process page-by-page to avoid memory issues
- **OCR**: For scanned PDFs without text layer, recommend using OCR tools first
- **Encoding**: Handle UTF-8 encoding properly when extracting text

## Common Use Cases

- Invoice text extraction
- Table data scraping from reports
- PDF form automation
- Document merging and splitting
