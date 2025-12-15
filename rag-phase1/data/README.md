# Data Directory

This directory is for storing PDF documents that will be ingested into the RAG system.

## Usage

1. Place any `.pdf` files in this directory (subdirectories are also supported)
2. Run the RAG demo with `--rebuild` to include the PDFs:
   ```bash
   python rag_demo.py --rebuild
   ```

## Supported Formats

- Currently only PDF files (`.pdf`) are supported
- The system recursively scans all subdirectories
- Each PDF will be chunked and indexed alongside the web content

## Example Structure

```
data/
├── technical_docs/
│   ├── guide_1.pdf
│   └── guide_2.pdf
├── research_papers/
│   └── paper.pdf
└── README.md
```

## Notes

- PDFs will be tagged with their file path in metadata
- You can filter by source in queries using metadata filters
- Large PDFs may take time to process during initial indexing

