# Using main_large.py for Large Document Collections

This variant handles 8,000+ pages efficiently with persistent storage and batch processing.

## Key Differences from main.py

- **Persistent vector store** - Build once, reuse forever (no rebuilding on each run)
- **Batch processing** - Handles large datasets without memory issues
- **RecursiveCharacterTextSplitter** - Better semantic chunking for academic text
- **Directory-based loading** - Process multiple files from a `documents/` folder
- **Increased retrieval** - Fetches 10 chunks instead of 3 for comprehensive answers
- **Progress indicators** - Shows indexing progress for large datasets

## Setup

Same as main.py, but create a `documents` folder:

```bash
mkdir documents
```

## Usage

### First Time: Build the Index

```bash
python main_large.py --index
```

This will:
- Scan all `.txt` files in the `documents/` folder
- Split them into chunks
- Create embeddings (may take 10-30 minutes for 8,000 pages)
- Save the vector store to `chroma_store_large/`

### Query the Indexed Documents

```bash
python main_large.py
```

Ask questions as usual. The system loads the pre-built index instantly.

### Rebuild the Index

If you add new documents:

```bash
python main_large.py --reindex
```

## Performance Estimates

- **8,000 pages** → ~50,000-100,000 chunks
- **Indexing time**: 15-45 minutes (one-time)
- **Query time**: 3-10 seconds per question
- **Disk space**: ~200-500 MB for vector store
- **RAM usage**: ~2-4 GB during indexing, ~1 GB during queries
