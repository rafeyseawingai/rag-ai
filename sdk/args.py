import argparse


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Process Medicare queries.")
        parser.add_argument("--qid", required=True, type=str, help="Question Id in queries.json (case insensitive)")
        self.args = parser.parse_args()
        self.qid = self.args.qid.upper()