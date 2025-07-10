#!/usr/bin/env python3
"""
Test script for HR Document RAG system
Run this to test your RAG implementation step by step
"""

import os
import sys
import json
from pathlib import Path

# Add your project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.langchain_rag import HRDocumentRAG


def test_chunk_loading():
    """Test 1: Load chunks from file"""
    print("🧪 TEST 1: Loading chunks from file")
    print("-" * 50)
    
    # Check if chunks file exists
    chunks_file = "chunks.json"  # Update with your actual file name
    if not os.path.exists(chunks_file):
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Please run your chunker first to create the chunks file")
        return False
    
    try:
        # Test loading chunks
        rag = HRDocumentRAG(chunks_file=chunks_file)
        print(f"✅ Successfully loaded {len(rag.subtopics)} subtopics and {len(rag.chapters)} chapters")
        return rag
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        return False


def test_document_creation(rag):
    """Test 2: Create LangChain documents"""
    print("\n🧪 TEST 2: Creating LangChain documents")
    print("-" * 50)
    
    try:
        documents = rag.create_langchain_documents()
        print(f"✅ Created {len(documents)} LangChain documents")
        
        # Show sample document
        if documents:
            sample_doc = documents[0]
            print(f"\n📄 Sample document:")
            print(f"Type: {sample_doc.metadata.get('type')}")
            print(f"Chapter: {sample_doc.metadata.get('chapter_title')}")
            print(f"Content preview: {sample_doc.page_content[:200]}...")
        
        return documents
    except Exception as e:
        print(f"❌ Error creating documents: {e}")
        return False


def test_embeddings(rag):
    """Test 3: Test embeddings"""
    print("\n🧪 TEST 3: Testing embeddings")
    print("-" * 50)
    
    try:
        # Test single embedding
        test_text = "Employee onboarding process"
        embedding = rag.embeddings.embed_query(test_text)
        print(f"✅ Generated embedding for '{test_text}'")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Embedding preview: {embedding[:5]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error with embeddings: {e}")
        return False


def test_vector_store_creation(rag, documents):
    """Test 4: Create vector store"""
    print("\n🧪 TEST 4: Creating vector store")
    print("-" * 50)
    
    try:
        # Remove existing vector store for clean test
        import shutil
        if os.path.exists(rag.chroma_db_path):
            shutil.rmtree(rag.chroma_db_path)
            print("🗑️ Removed existing vector store")
        
        # Create new vector store
        rag.setup_vector_store(documents)
        print(f"✅ Vector store created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False


def test_retrieval(rag):
    """Test 5: Test document retrieval"""
    print("\n🧪 TEST 5: Testing document retrieval")
    print("-" * 50)
    
    try:
        # Test queries
        test_queries = [
            "employee onboarding",
            "performance evaluation",
            "leave policy",
            "dress code"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = rag.search_documents(query, k=2)
            
            if results:
                print(f"✅ Found {len(results)} relevant documents")
                for i, doc in enumerate(results, 1):
                    print(f"  {i}. {doc.metadata.get('type', 'unknown')} - {doc.metadata.get('chapter_title', 'unknown')}")
                    if doc.metadata.get('subtopic_title'):
                        print(f"     Section: {doc.metadata.get('subtopic_title')}")
            else:
                print("⚠️ No documents found")
        
        return True
    except Exception as e:
        print(f"❌ Error with retrieval: {e}")
        return False


def test_full_rag_system():
    """Test 6: Full RAG system build"""
    print("\n🧪 TEST 6: Full RAG system build")
    print("-" * 50)
    
    try:
        # Initialize with force rebuild
        rag = HRDocumentRAG(
            chunks_file="chunks.json",  # Update with your file
            chroma_db_path="./test_chroma_db",
            collection_name="hr_test"
        )
        
        # Build complete system
        rag.build_rag_system(force_rebuild=True)
        
        # Test the system
        test_query = "What is the employee onboarding process?"
        results = rag.query(test_query)
        
        print(f"✅ Full RAG system working! Found {len(results)} relevant documents")
        return True
        
    except Exception as e:
        print(f"❌ Error with full RAG system: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 HR Document RAG System Test Suite")
    print("=" * 60)
    
    # Test 1: Load chunks
    rag = test_chunk_loading()
    if not rag:
        print("\n❌ Cannot proceed without chunks. Please fix chunk loading first.")
        return
    
    # Test 2: Create documents
    documents = test_document_creation(rag)
    if not documents:
        print("\n❌ Cannot proceed without documents.")
        return
    
    # Test 3: Test embeddings
    if not test_embeddings(rag):
        print("\n❌ Cannot proceed without working embeddings.")
        return
    
    # Test 4: Create vector store
    if not test_vector_store_creation(rag, documents):
        print("\n❌ Cannot proceed without vector store.")
        return
    
    # Test 5: Test retrieval
    if not test_retrieval(rag):
        print("\n❌ Retrieval not working.")
        return
    
    # Test 6: Full system test
    if not test_full_rag_system():
        print("\n❌ Full system test failed.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Your RAG system is working correctly.")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Choose and integrate a local LLM (Llama.cpp or HuggingFace)")
    print("2. Test end-to-end question answering")
    print("3. Optimize retrieval parameters")
    print("4. Add evaluation metrics")


if __name__ == "__main__":
    main()