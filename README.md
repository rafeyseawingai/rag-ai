# Introduction

Build a simple Retrieval-Augmented Generation (RAG) System

See [problem statement](./problem.md)

## Setup

Install the following dependencies:

```sh
pip install sentence-transformers faiss-cpu numpy transformers argparse torch
```

1. `sentence-transformers`:  Generates embeddings for document chunks.  
2. `faiss-cpu`:  Efficient similarity search for retrieval.  
3. `numpy`:  Handles numerical operations.  
4. `transformers`:  Loads and runs the Flan-T5 model.  
5. `argparse`:  Parses command-line arguments (`--qid`).  
6. `torch`:  Required by transformers for deep learning model execution.  

## Run

Use following command to run application:

```sh
python main.py --qid Q1
```

Or

```sh
py main.py --qid Q1
```
