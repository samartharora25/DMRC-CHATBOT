# main.py

import time
from src.chunker import PDFChunker
from src.embedder import Embedder
from src.langchain_rag import HRDocumentRAG
import json
import os
from src.summarizer import HRPolicySummarizer, SummarizationConfig, ModelManager, create_test_document
from rich.console import Console
import chromadb
from langchain.schema import Document

console = Console()

def main():
    """Complete RAG pipeline: PDF processing -> Chunking -> Embeddings -> Vector Store"""
    
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
            console.print("\n📄 [yellow]Processing all documents in RAG system...[yellow]")
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
            docs = []
            for content, metadata in zip(raw_docs, raw_metas): # type: ignore
                if metadata.get("type") != "subtopic":
                    continue  # Skip chapter-level documents

                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                docs.append(doc)

            # Summarize all documents
            result = summarizer.summarize_batch(docs) # type: ignore

            # Display results
            for res in result:
                if res.get('processing_successful'):
                    stats = res['metadata']['summary_stats']
                    console.print(f"\n📊 [green]Results:[/green]")
                    console.print(f"Original: {stats['original_words']} words")
                    console.print(f"Summary: {stats['summary_words']} words")
                    console.print(f"Compression: {stats['compression_ratio']:.2f}")
                    console.print(f"Privacy Redacted: {res['privacy_redacted']}")
                    
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

                    # Append the full result dictionary
                    summaries.append(res)

                    # Save back to file
                    with open(summaries_file, "w", encoding="utf-8") as f:
                        json.dump(summaries, f, indent=2, ensure_ascii=False)

                    console.print(f"\n📝 [blue]Summary saved to {summaries_file}[/blue]")
                else:
                    console.print(f"❌ [red]Processing failed: {res.get('error', 'Unknown error')}[/red]")
            
            # Brief pause before cleanup
            time.sleep(3)
            
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