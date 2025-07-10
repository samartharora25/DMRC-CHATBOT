from src.chunker import PDFChunker
from src.embedder import Embedder
from src.langchain_rag import HRDocumentRAG
import json
import os

def main():
    """Complete RAG pipeline: PDF processing -> Chunking -> Embeddings -> Vector Store"""
    
    # Configuration
    pdf_path = "data/Sample HR Policy Manual.pdf"
    chunks_file = "chunks.json"
    embeddings_file = "embeddings.json"
    chroma_db_path = "./hr_chroma_db"
    
    try:
        # Phase 1: Document Processing
        print("📄 Phase 1: Document Processing")
        chunker = PDFChunker(pdf_path)
        toc = chunker.parse_toc_structure()
        subtopic_chunks, chapter_chunks = chunker.extract_subtopic_and_chapter_chunks(toc)
        
        print(f"   ✅ Extracted {len(subtopic_chunks)} subtopics, {len(chapter_chunks)} chapters")
        
        # Save chunks
        chunker.save_chunks(subtopic_chunks, chapter_chunks, chunks_file)
        chunker.close()
        
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
        rag.build_rag_system(force_rebuild=True)
        
        # Phase 4: System Summary
        print("\n📊 Phase 4: System Summary")
        stats = rag.get_stats()
        
        file_sizes = {}
        for file_path in [chunks_file, embeddings_file]:
            if os.path.exists(file_path):
                file_sizes[file_path] = f"{os.path.getsize(file_path) / 1024:.1f} KB"
        
        print(f"   • Subtopics: {stats['total_subtopics']}")
        print(f"   • Chapters: {stats['total_chapters']}")
        print(f"   • Vector Store: {stats['documents_in_vectorstore']} documents")
        print(f"   • Files Created:")
        for file_path, size in file_sizes.items():
            print(f"     - {file_path}: {size}")
        
        print(f"\n✅ RAG System Ready!")
        print(f"   Vector Store: {chroma_db_path}")
        print(f"   Use the HRDocumentRAG class for queries")
        
        return rag
        
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