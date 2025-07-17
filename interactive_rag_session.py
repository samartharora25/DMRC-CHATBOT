# interactive_rag_session.py

from src.langchain_rag import HRDocumentRAG
import os
import time

def interactive_rag_session():
    """Interactive session for asking multiple questions with enhanced debugging"""
    print("🚀 Starting interactive RAG session...")
    
    # Setup RAG with enhanced error handling
    try:
        rag = HRDocumentRAG(
            chunks_file="chunks.json",
            chroma_db_path="./hr_chroma_db"
        )
        
        # Build system with debugging enabled
        print("\n🔧 Building RAG system...")
        rag.build_rag_system(
            force_rebuild=False,  # Set to True if you want to rebuild
            groq_model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=1024
        )
        
        # Show system stats
        print("\n📊 System initialized successfully!")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  ✅ {key}: {value}")
        
        # Test the system with a simple query
        print("\n🧪 Testing system with sample query...")
        test_results = rag.debug_search("policy", k=3)
        
        if not test_results:
            print("⚠️ WARNING: No documents found in test search!")
            print("   This suggests an embedding or vector store issue.")
            print("   Try running with force_rebuild=True")
        else:
            print(f"✅ Test successful - found {len(test_results)} documents")
            
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("   Please check your chunks.json file and vector store.")
        return
    
    print("\n💬 HR Document Assistant Ready!")
    print("Available commands:")
    print("  - Ask any question about HR policies")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'stats' for system info")
    print("  - 'debug <query>' to see search results without LLM")
    print("  - 'rebuild' to rebuild vector store")
    print("  - 'test' to run system tests")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            elif question.lower() == 'stats':
                stats = rag.get_stats()
                print("\n📊 System Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
                
            elif question.lower() == 'rebuild':
                print("\n🔨 Rebuilding vector store...")
                try:
                    rag.build_rag_system(force_rebuild=True)
                    print("✅ Vector store rebuilt successfully!")
                except Exception as e:
                    print(f"❌ Error rebuilding: {e}")
                continue
                
            elif question.lower() == 'test':
                print("\n🧪 Running system tests...")
                run_system_tests(rag)
                continue
                
            elif question.lower().startswith('debug '):
                debug_query = question[6:].strip()
                if debug_query:
                    print(f"\n🔍 Debug search for: '{debug_query}'")
                    results = rag.debug_search(debug_query, k=5)
                    if not results:
                        print("❌ No documents found")
                    else:
                        print(f"✅ Found {len(results)} documents")
                continue
                
            elif not question:
                continue
            
            # Process regular question
            print(f"\n🔍 Processing: '{question}'")
            start_time = time.time()
            
            # First check if we can find relevant documents
            search_results = rag.search_documents(question, k=3)
            
            if not search_results:
                print("❌ No relevant documents found for your question.")
                print("   This could mean:")
                print("   1. Your question topic isn't covered in the documents")
                print("   2. There's an embedding mismatch issue")
                print("   3. Try rephrasing your question with different keywords")
                
                # Suggest related topics if possible
                print("\n💡 Try searching for more general terms like:")
                print("   - 'employee policies'")
                print("   - 'HR procedures'")
                print("   - 'company guidelines'")
                continue
            
            # If documents found, proceed with full query
            print(f"📖 Found {len(search_results)} relevant documents")
            response = rag.query_with_details(question)
            
            processing_time = time.time() - start_time
            print(f"\n⏱️ Processing time: {processing_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            print("   Please try again or type 'quit' to exit.")


def run_system_tests(rag):
    """Run comprehensive system tests"""
    print("\n🧪 Running system tests...")
    
    # Test 1: Check embeddings
    print("\n1. Testing embeddings...")
    try:
        test_embedding = rag.embeddings.embed_query("test query")
        if test_embedding and len(test_embedding) > 0:
            print(f"   ✅ Embeddings working - dimension: {len(test_embedding)}")
        else:
            print("   ❌ Embeddings not working properly")
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")
    
    # Test 2: Check vector store
    print("\n2. Testing vector store...")
    try:
        if rag.vectorstore:
            collection = rag.vectorstore._collection
            doc_count = collection.count()
            print(f"   ✅ Vector store loaded - {doc_count} documents")
        else:
            print("   ❌ Vector store not initialized")
    except Exception as e:
        print(f"   ❌ Vector store test failed: {e}")
    
    # Test 3: Check document search
    print("\n3. Testing document search...")
    try:
        test_queries = ["policy", "employee", "HR", "procedure"]
        for query in test_queries:
            results = rag.search_documents(query, k=1)
            if results:
                print(f"   ✅ '{query}' found {len(results)} documents")
            else:
                print(f"   ❌ '{query}' found 0 documents")
    except Exception as e:
        print(f"   ❌ Document search test failed: {e}")
    
    # Test 4: Check Groq API
    print("\n4. Testing Groq API...")
    try:
        if rag.groq_api_key:
            print("   ✅ Groq API key configured")
            if rag.qa_chain:
                print("   ✅ QA chain initialized")
            else:
                print("   ❌ QA chain not initialized")
        else:
            print("   ❌ Groq API key not found")
    except Exception as e:
        print(f"   ❌ Groq API test failed: {e}")
    
    # Test 5: End-to-end test
    print("\n5. Testing end-to-end query...")
    try:
        test_query = "What are the policies?"
        response = rag.query(test_query)
        if isinstance(response, dict) and response.get('answer'):
            print("   ✅ End-to-end query successful")
        else:
            print("   ❌ End-to-end query failed")
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
    
    print("\n✅ System tests completed!")


def quick_test():
    """Quick test function for debugging"""
    print("🧪 Quick RAG system test...")
    
    try:
        rag = HRDocumentRAG(
            chunks_file="chunks.json",
            chroma_db_path="./hr_chroma_db"
        )
        
        print("\n📊 System stats:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🔍 Testing search...")
        results = rag.debug_search("employee", k=3)
        print(f"Found {len(results)} documents")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")


if __name__ == "__main__":
    # Choose what to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        interactive_rag_session()