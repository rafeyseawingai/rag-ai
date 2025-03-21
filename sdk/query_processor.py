
import json
import time
from sdk.time_logger import TimeLogger


class QueryProcessor:
    """Loads queries and extracts the required query based on QID."""
    def __init__(self, query_path, qid):
        start_time = time.time()
        self.query_path = query_path
        self.qid = qid
        self.query = self.load_query()
        TimeLogger.log("Query processing completed", start_time)

    def load_query(self):
        """Finds the query matching the given QID."""
        with open(self.query_path, "r", encoding="utf-8") as file:
            queries = json.load(file)
        for query in queries:
            if query["id"].upper() == self.qid:
                return query
        raise ValueError(f"Query ID {self.qid} not found in queries.json")
