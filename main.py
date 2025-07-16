# main.py

import gc
import time

import torch
from src.chunker import PDFChunker
from src.embedder import Embedder
from src.langchain_rag import HRDocumentRAG
import json
import os
import signal
import sys
from src.summarizer import HRPolicySummarizer, SummarizationConfig, ModelManager, create_test_document
from rich.console import Console
import chromadb
from langchain.schema import Document

console = Console()

gc.collect()
torch.cuda.empty_cache()
class ProgressTracker:
    def __init__(self, progress_file="summarization_progress.json"):
        self.progress_file = progress_file
        self.processed_docs = set()
        self.load_progress()
    
    def load_progress(self):
        """Load previously processed document IDs"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.processed_docs = set(data.get("processed_docs", []))
                    console.print(f"📂 [blue]Loaded progress: {len(self.processed_docs)} documents already processed[/blue]")
            except json.JSONDecodeError:
                console.print(f"⚠️ [yellow]Progress file corrupted, starting fresh[/yellow]")
                self.processed_docs = set()
    
    def save_progress(self):
        """Save current progress"""
        data = {
            "processed_docs": list(self.processed_docs),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def mark_processed(self, doc_id):
        """Mark a document as processed"""
        self.processed_docs.add(doc_id)
        self.save_progress()
    
    def is_processed(self, doc_id):
        """Check if document was already processed"""
        return doc_id in self.processed_docs
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            "processed_count": len(self.processed_docs),
            "processed_docs": list(self.processed_docs)
        }

def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully"""
    console.print(f"\n🛑 [yellow]Received interrupt signal. Saving progress and exiting gracefully...[/yellow]")
    console.print(f"💾 [blue]You can restart the process later and it will continue from where it left off.[/blue]")
    sys.exit(0)

def generate_doc_id(doc):
    """Generate a unique ID for a document based on its content and metadata"""
    # Use subtopic_title as the primary identifier since it's unique
    metadata = doc.metadata
    subtopic_title = metadata.get('subtopic_title', 'unknown')
    chapter_title = metadata.get('chapter_title', 'unknown')
    
    # Create a unique ID using subtopic and chapter titles
    # Clean the strings to make them filesystem-safe
    clean_subtopic = str(subtopic_title).replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("-", "_")
    clean_chapter = str(chapter_title).replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("-", "_")
    
    doc_id = f"{clean_chapter}_{clean_subtopic}"
    return doc_id

def main():
    """Complete RAG pipeline: PDF processing -> Chunking -> Embeddings -> Vector Store"""
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configuration
    pdf_path = "data/Sample HR Policy Manual.pdf"
    chunks_file = "chunks.json"
    embeddings_file = "embeddings.json"
    chroma_db_path = "./hr_chroma_db"
    
    try:
        # Phase 1: Document Processing
        # print("📄 Phase 1: Document Processing")
        # chunker = PDFChunker(pdf_path)
        # toc = chunker.parse_toc_structure()
        # subtopic_chunks, chapter_chunks = chunker.extract_subtopic_and_chapter_chunks(toc)
        
        # print(f"   ✅ Extracted {len(subtopic_chunks)} subtopics, {len(chapter_chunks)} chapters")
        
        # # Save chunks
        # chunker.save_chunks(subtopic_chunks, chapter_chunks, chunks_file)
        # chunker.close()
        
        # # Phase 2: Embeddings (Optional - for standalone use)
        # print("\n🔮 Phase 2: Creating Embeddings")
        # embedder = Embedder()
        # subtopic_embeddings = embedder.embed_chunks(subtopic_chunks)
        # chapter_embeddings = embedder.embed_chunks(chapter_chunks)
        # embedder.save_embeddings(subtopic_embeddings, chapter_embeddings, embeddings_file)
        
        # print(f"   ✅ Created {len(subtopic_embeddings + chapter_embeddings)} embeddings")
        
        # Phase 3: RAG System Setup
        print("\n🚀 Phase 3: Building RAG System")
        rag = HRDocumentRAG(
            chunks_file=chunks_file,
            chroma_db_path=chroma_db_path,
            collection_name="hr_policies"
        )
        
        # Build the complete RAG system
        rag.build_rag_system(force_rebuild=False)

        # Phase 4: System Summary
        console.print("\n🚀 [bold cyan]HR Policy Summarizer - Refactored Version[/bold cyan]")
        # Configuration
        config = SummarizationConfig(
            temperature=0.1,
            repetition_penalty=1.1,
            preserve_structure=True,
            redact_sensitive_info=True
        )
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker()
        
        # Initialize components
        model_manager = ModelManager()
        try:
            # Load model
            if not model_manager.load_model(config):
                console.print("❌ [red]Failed to load model[/red]")
                return
            
            llm = model_manager.get_llm()
            if not llm:
                console.print("❌ [red]Failed to get LLM wrapper[/red]")
                return
            
            # Create summarizer
            summarizer = HRPolicySummarizer(llm, config)
            
            # # Test with sample document
            # test_doc = create_test_document()
            # console.print("\n📄 [yellow]Processing test document...[/yellow]")
            
            # result = summarizer.summarize_document(test_doc)
            
            # summarize all documents in the RAG system
            console.print("\n📄 [yellow]Processing all documents in RAG system...[/yellow]")
            # Load all documents directly from ChromaDB vector store
            client = chromadb.PersistentClient(path=chroma_db_path)
            collection = client.get_collection("hr_policies")

            # Retrieve both content and metadata
            chroma_docs = collection.get(include=["documents", "metadatas"])
            raw_docs = chroma_docs.get("documents", [])
            raw_metas = chroma_docs.get("metadatas", [])

            if not raw_docs:
                console.print("❌ [red]No documents found in the vector store[/red]")
                return

            # Restore Document objects, filtering out chapter-level docs
            all_docs = []
            for content, metadata in zip(raw_docs, raw_metas): # type: ignore
                if metadata.get("type") != "subtopic":
                    continue  # Skip chapter-level documents

                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                all_docs.append(doc)
            
            # Filter out already processed documents
            docs_to_process = []
            for doc in all_docs:
                doc_id = generate_doc_id(doc)
                if not progress_tracker.is_processed(doc_id):
                    docs_to_process.append((doc, doc_id))
            
            total_docs = len(all_docs)
            remaining_docs = len(docs_to_process)
            processed_docs = total_docs - remaining_docs
            
            console.print(f"\n📊 [cyan]Processing Status:[/cyan]")
            console.print(f"   • Total documents: {total_docs}")
            console.print(f"   • Already processed: {processed_docs}")
            console.print(f"   • Remaining to process: {remaining_docs}")
            
            if remaining_docs == 0:
                console.print("\n🎉 [green]All documents have been processed![/green]")
                return
            
            console.print(f"\n🔄 [yellow]Resuming from document {processed_docs + 1}...[/yellow]")

            def get_individual_summary(result, idx, total):
                doc, doc_id = docs_to_process[idx]
                actual_doc_num = processed_docs + idx + 1
                
                console.print(f"\n🔍 [yellow]Processing document {actual_doc_num}/{total_docs} (Batch: {idx + 1}/{total})...[/yellow]")
                console.print(f"📄 [blue]Document: {doc.metadata.get('subtopic', 'Unknown')}[/blue]")
                
                if result.get('processing_successful'):
                    # Mark as processed
                    progress_tracker.mark_processed(doc_id)
                    
                    stats = result['metadata']['summary_stats']
                    console.print(f"\n📊 [green]Results:[/green]")
                    console.print(f"Original: {stats['original_words']} words")
                    console.print(f"Summary: {stats['summary_words']} words")
                    console.print(f"Compression: {stats['compression_ratio']:.2f}")
                    console.print(f"Privacy Redacted: {result['privacy_redacted']}")
                    
                    # Save and append all summaries in a JSON file
                    summaries_file = "summaries.json"
                    summaries = []

                    # Load existing summaries if file exists
                    if os.path.exists(summaries_file):
                        with open(summaries_file, "r", encoding="utf-8") as f:
                            try:
                                summaries = json.load(f)
                            except json.JSONDecodeError:
                                summaries = []

                    # Add document ID to result for tracking
                    result['doc_id'] = doc_id
                    
                    # Append the full result dictionary
                    summaries.append(result)

                    # Save back to file
                    with open(summaries_file, "w", encoding="utf-8") as f:
                        json.dump(summaries, f, indent=2, ensure_ascii=False)

                    console.print(f"\n📝 [blue]Summary saved to {summaries_file}[/blue]")
                    console.print(f"✅ [green]Progress saved - {len(progress_tracker.processed_docs)} documents completed[/green]")
                else:
                    console.print(f"❌ [red]Processing failed: {result.get('error', 'Unknown error')}[/red]")
                    console.print(f"🔄 [yellow]This document will be retried on next run[/yellow]")

            # Extract just the documents for processing
            docs = [doc for doc, _ in docs_to_process]
            
            # Summarize remaining documents
            console.print(f"\n🚀 [green]Starting batch processing of {remaining_docs} documents...[/green]")
            console.print(f"💡 [blue]Tip: You can safely interrupt (Ctrl+C) and resume later[/blue]")
            
            summarizer.summarize_batch(docs, on_result_callback=get_individual_summary) 

            # Display final results
            console.print(f"\n🎉 [green]All documents processed successfully![/green]")
            final_stats = progress_tracker.get_stats()
            console.print(f"📊 [cyan]Final Statistics:[/cyan]")
            console.print(f"   • Total processed: {final_stats['processed_count']} documents")
            
            # Brief pause before cleanup
            time.sleep(3)
            
        except KeyboardInterrupt:
            console.print(f"\n🛑 [yellow]Process interrupted by user[/yellow]")
            console.print(f"💾 [blue]Progress has been saved. Restart to continue from where you left off.[/blue]")
            return
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
        
        finally:
            model_manager.cleanup()
            console.print("\n✅ [green]Process completed![/green]")


        # stats = rag.get_stats()
        
        # file_sizes = {}
        # for file_path in [chunks_file, embeddings_file]:
        #     if os.path.exists(file_path):
        #         file_sizes[file_path] = f"{os.path.getsize(file_path) / 1024:.1f} KB"
        
        # print(f"   • Subtopics: {stats['total_subtopics']}")
        # print(f"   • Chapters: {stats['total_chapters']}")
        # print(f"   • Vector Store: {stats['documents_in_vectorstore']} documents")
        # print(f"   • Files Created:")
        # for file_path, size in file_sizes.items():
        #     print(f"     - {file_path}: {size}")
        
        # print(f"\n✅ RAG System Ready!")
        # print(f"   Vector Store: {chroma_db_path}")
        # print(f"   Use the HRDocumentRAG class for queries")
        
        # return rag
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Make sure the PDF file exists in the data/ directory")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()