#!/usr/bin/env python
"""
AmbedkarGPT - Main Entry Point

Usage:
    python run.py --ingest              # Ingest PDF and build knowledge graph
    python run.py --load --interactive  # Load data and start interactive mode
    python run.py --load --query "..."  # Load data and answer a single query
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.ambedkargpt import AmbedkarGPT


def main():
    parser = argparse.ArgumentParser(
        description="AmbedkarGPT - SEMRAG-based RAG System for Dr. B.R. Ambedkar's Works",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --ingest                    # Process PDF and build knowledge graph
  python run.py --load --interactive        # Start interactive Q&A
  python run.py --load --query "Who was Ambedkar?"
  python run.py --ingest --pdf custom.pdf   # Use custom PDF path
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline (chunking, graph building, etc.)"
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to ingest (default: data/Ambedkar_book.pdf)"
    )
    
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load previously processed data"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to answer"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive Q&A mode"
    )
    
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local search (skip global)"
    )
    
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Use only global search (skip local)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ingest and not args.load:
        if args.query or args.interactive:
            print("Note: --load flag added automatically since data is needed")
            args.load = True
    
    if not args.ingest and not args.load and not args.query and not args.interactive:
        parser.print_help()
        return
    
    # Initialize
    config_path = Path(__file__).parent / args.config
    print(f"\n{'='*60}")
    print("AMBEDKARGPT - SEMRAG-based RAG System")
    print(f"{'='*60}\n")
    
    gpt = AmbedkarGPT(config_path=str(config_path))
    
    # Determine PDF path
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        # Default locations to check
        possible_paths = [
            Path(__file__).parent / "data" / "Ambedkar_book.pdf",
            Path(__file__).parent.parent / "data" / "Ambedkar_book.pdf",
        ]
        pdf_path = None
        for p in possible_paths:
            if p.exists():
                pdf_path = p
                break
        
        if pdf_path is None and args.ingest:
            print("Error: Could not find Ambedkar_book.pdf")
            print("Please provide path with --pdf argument")
            print("Or copy PDF to: ambedkargpt/data/Ambedkar_book.pdf")
            return
    
    # Run ingestion if requested
    if args.ingest:
        if pdf_path and pdf_path.exists():
            print(f"Ingesting document: {pdf_path}")
            gpt.ingest_document(str(pdf_path))
        else:
            print(f"Error: PDF not found at {pdf_path}")
            return
    
    # Load processed data if requested
    if args.load:
        try:
            gpt.load_processed_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Run with --ingest first to process the document")
            return
    
    # Determine search settings
    use_local = not args.global_only
    use_global = not args.local_only
    
    # Handle query
    if args.query:
        print(f"\nQuery: {args.query}\n")
        result = gpt.query(
            args.query,
            use_local=use_local,
            use_global=use_global
        )
        
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(result["answer"])
        
        if result.get("citations"):
            print(f"\n{'='*60}")
            print("SOURCES:")
            print(f"{'='*60}")
            for i, cite in enumerate(result["citations"][:5], 1):
                source = cite.get("source", "")
                chunk_id = cite.get("chunk_id", "")
                score = cite.get("score", 0)
                print(f"{i}. [{source}] {chunk_id} (relevance: {score:.2f})")
    
    # Interactive mode
    if args.interactive:
        gpt.interactive_mode()


def demo_queries():
    """Run demo with sample queries."""
    sample_queries = [
        "Who was Dr. B.R. Ambedkar?",
        "What were Ambedkar's views on caste?",
        "What role did Ambedkar play in the Indian Constitution?",
        "What is the significance of education according to Ambedkar?",
        "How did Ambedkar fight for social justice?"
    ]
    
    print("\n" + "="*60)
    print("DEMO MODE - Sample Queries")
    print("="*60)
    
    config_path = Path(__file__).parent / "config.yaml"
    gpt = AmbedkarGPT(config_path=str(config_path))
    
    try:
        gpt.load_processed_data()
    except:
        print("Error: No processed data found. Run --ingest first.")
        return
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query}")
        print("="*60)
        
        result = gpt.query(query)
        print(f"\nANSWER:\n{result['answer'][:500]}...")
        
        if input("\nPress Enter for next query (or 'q' to quit): ").lower() == 'q':
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_queries()
    else:
        main()
