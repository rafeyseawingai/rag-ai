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

## Model

Model `free` alternatives depending on **hardware, accuracy needs, and speed requirements**. Since we have **16GB RAM and an Intel Core i7**, you can run **mid-sized models efficiently**, though GPU-based models may still be slow.  

`Flan-T5 (Large or XL)` is good for Question Answer use case as we have.

| **Model**                   | **Size**  | **Pros**                            | **Cons**                                | **Ideal For**                   |
| --------------------------- | --------- | ----------------------------------- | --------------------------------------- | ------------------------------- |
| **Flan-T5 (Base/Large/XL)** | 250M - 3B | Lightweight, free, good for Q&A     | Less powerful than GPT-4                | General NLP, Question Answering |
| **Mistral-7B**              | 7B        | High accuracy, better than LLaMA-7B | Needs 8GB+ VRAM for good speed          | Chatbots, Reasoning             |
| **LLaMA 2-7B/13B**          | 7B / 13B  | Strong general performance          | Slower on CPU                           | Research, Chatbots              |
| **Falcon-7B/40B**           | 7B / 40B  | Open-source, good for text tasks    | Slow on CPU, 40B too large for 16GB RAM | Content Generation              |
| **Gemma-7B** *(by Google)*  | 7B        | Optimized for instruction-following | Requires GPU for fast results           | AI Assistants                   |
| **GPT-J-6B**                | 6B        | Reasonably fast, free               | Lower accuracy vs. newer models         | Creative Writing                |
| **GPT-NeoX-20B**            | 20B       | Powerful for reasoning              | Too large for 16GB RAM alone            | Advanced Research               |
| **Phi-2 (by Microsoft)**    | 2.7B      | Optimized, small, powerful          | Not as tested as Mistral                | Small-scale AI Apps             |

---

## **Best Choice for QA, Intel i7 + 16GB RAM Setup**
1️⃣ **🚀 Fastest (CPU-Friendly) Choice:**  
   - **Flan-T5 (Large)** → Works well on CPU, good accuracy.  

2️⃣ **⚡ More Powerful But Heavier:**  
   - **Mistral-7B** → Better than LLaMA-7B, but slower on CPU.  
   - **LLaMA 2-7B** → Great, but needs more resources.  

3️⃣ **💡 Best for Instruction-Following (Chatbots):**  
   - **Gemma-7B** → Google’s latest model.  
   - **Phi-2** → Small but powerful.  
