# Ambedkar Speech Q&A

A minimal command-line Retrieval-Augmented Generation (RAG) prototype that answers questions about Dr. B. R. Ambedkar's speech. The pipeline is built with LangChain, uses HuggingFace embeddings stored in a local ChromaDB vector store, and queries the local Mistral 7B model running through Ollama.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed locally
- `mistral` model available via `ollama pull mistral`
- LangChain 0.3+ plus the `langchain-community` integrations (installed through `requirements.txt`)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
ollama pull mistral
```

## Usage

1. Ensure `speech.txt` (provided) is in the project root.
2. Run the CLI and start asking questions:

```bash
python main.py
```

Type your question and press enter. Use `exit`, `quit`, or `q` to close the program. Each run reloads the speech, splits it into chunks, rebuilds the Chroma vector store, and feeds retrieved context to Mistral 7B through LangChain's `RetrievalQA` chain.

## Project Structure

- `main.py` – minimal RAG pipeline and interactive prompt.
- `speech.txt` – Dr. Ambedkar speech used as the knowledge base.
- `requirements.txt` – Python dependencies.
- `README.md` – setup and usage instructions.
