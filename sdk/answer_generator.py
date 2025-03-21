from sdk.time_logger import TimeLogger
import time
from transformers import pipeline


class AnswerGenerator:
    """Uses an LLM (Flan-T5) to generate answers based on retrieved context."""
    def __init__(self, query, context):
        self.query = query
        self.context = context
        self.llm = pipeline("text-generation", model="google/flan-t5-base")

    def generate_answer(self):
        """Generates an answer using Flan-T5."""
        start_time = time.time()
        if not self.context:
            return "No relevant information found in the document."
        prompt = f"Answer the query based only on the given context.\n\nContext:\n{self.context}\n\nQuery: {self.query}\nAnswer:"
        response = self.llm(prompt, max_length=100, do_sample=True)[0]["generated_text"].strip()
        TimeLogger.log("Answer generation completed", start_time)
        return response

