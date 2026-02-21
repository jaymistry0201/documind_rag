import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    """Extract text from each page of a PDF as a list of dictionaries."""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page_num": i+1, "text": text})
    return pages
