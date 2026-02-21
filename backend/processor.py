import uuid

def chunk_texts(pages, chunk_size=500, overlap=50):
    """Split PDF pages into overlapping text chunks with unique IDs."""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                "id": str(uuid.uuid4()),
                "page": page["page_num"],
                "start": start,
                "end": end,
                "chunk": chunk_text
            })
            start += chunk_size - overlap
    return chunks
