from sdk.time_logger import TimeLogger
import time


class DocumentProcessor:
    """Loads the document, splits it into chunks, and processes text."""
    def __init__(self, doc_path, max_tokens=80):
        start_time = time.time()
        self.doc_path = doc_path
        self.max_tokens = max_tokens
        self.text = self.load_document()
        self.chunks = self.chunk_text()
        TimeLogger.log("Document processing completed", start_time)

    def load_document(self):
        """Reads the document from file."""
        with open(self.doc_path, "r", encoding="utf-8") as file:
            return file.read()

    def chunk_text(self):
        """Splits the text into chunks based on token count."""
        sentences = self.text.split("\n")
        chunks, chunk = [], ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= self.max_tokens:
                chunk += sentence + " "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
        if chunk:
            chunks.append(chunk.strip())
        return chunks
