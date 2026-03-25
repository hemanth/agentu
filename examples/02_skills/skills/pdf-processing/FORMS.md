# PDF Form Filling Guide

## Using PyPDF2 for Form Filling

```python
from PyPDF2 import PdfReader, PdfWriter

def fill_pdf_form(input_path, output_path, field_values):
    """
    Fill a PDF form with provided values.
    
    Args:
        input_path: Path to template PDF
        output_path: Path for filled PDF
        field_values: Dict mapping field names to values
    """
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    # Get the form fields
    page = reader.pages[0]
    
    # Update field values
    writer.add_page(page)
    writer.update_page_form_field_values(
        writer.pages[0],
        field_values
    )
    
    # Write output
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

# Example usage
form_data = {
    "Name": "John Doe",
    "Email": "john@example.com",
    "Date": "2024-01-15"
}

fill_pdf_form("template.pdf", "filled.pdf", form_data)
```

## Tips

- List all form fields with `reader.get_form_text_fields()`
- Check field names before filling
- Flatten forms after filling to prevent edits
