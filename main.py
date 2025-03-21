import json
import os
import numpy as np
from pathlib import Path

from sdk.answer_generator import AnswerGenerator
from sdk.args import Args
from sdk.document_processor import DocumentProcessor
from sdk.embedding_retriever import EmbeddingRetriever
from sdk.query_processor import QueryProcessor

# File Paths
INPUT_DIR = "input/"
OUTPUT_DIR = "output/"
DOC_PATH = f"{INPUT_DIR}/medicare_comparison.md"
QUERY_PATH = f"{INPUT_DIR}/queries.json"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "answers.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    args = Args()
    doc_processor = DocumentProcessor(DOC_PATH)
    query_processor = QueryProcessor(QUERY_PATH, args.qid)
    retriever = EmbeddingRetriever(doc_processor.chunks)
    
    # Retrieve context
    context, source_indices = retriever.retrieve_context(query_processor.query["text"])
    generator = AnswerGenerator(query_processor.query["text"], "\n".join(context))
    
    # Generate answer
    answer = generator.generate_answer()
    
    # Store output
    output = {
        "query_id": query_processor.query["id"],
        "query_text": query_processor.query["text"],
        "answer": answer,
        "source_chunks": source_indices,
        "source_text": context
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"Answer saved to {OUTPUT_PATH}")