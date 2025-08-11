# src/langchain_rag.py

import json
import torch
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# HuggingFace imports
from transformers import AutoTokenizer, AutoModel

# Your existing classes
from src.chunker import PDFChunker


class CustomEmbeddings(Embeddings):
    """Custom LangChain embeddings wrapper for your local BGE model"""
    
    def __init__(self, model_path: str = "BAAI/bge-large-en-v1.5"):
        print(f"🔧 Initializing CustomEmbeddings with model: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            print("✅ Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ Using CPU for embeddings")
            
        # Test embedding to ensure model works
        test_embedding = self._get_embedding("test")
        print(f"✅ Test embedding successful - shape: {test_embedding.shape}, sample values: {test_embedding[:3]}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        print(f"📄 Embedding {len(texts)} documents...")
        embeddings = []
        
        for i, text in enumerate(texts):
            if i % 10 == 0:  # Progress indicator
                print(f"  Processing document {i+1}/{len(texts)}")
            
            # Truncate very long texts to prevent issues
            truncated_text = text[:2000] if len(text) > 2000 else text
            embedding = self._get_embedding(truncated_text)
            embeddings.append(embedding.tolist())
        
        print(f"✅ Documents embedded successfully - sample embedding shape: {len(embeddings[0]) if embeddings else 0}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        print(f"🔍 Embedding query: '{text[:50]}...' (length: {len(text)})")
        
        # Truncate very long queries
        truncated_text = text[:500] if len(text) > 500 else text
        embedding = self._get_embedding(truncated_text)
        
        print(f"✅ Query embedded - shape: {embedding.shape}, sample values: {embedding[:3]}")
        return embedding.tolist()
    
    @torch.no_grad()
    def _get_embedding(self, text: str):
        """Get embedding for a single text"""
        if not text or not text.strip():
            print("⚠️ Empty text provided for embedding")
            # Return zero vector for empty text
            return torch.zeros(1024)  # Adjust size based on your model
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            
            # Normalize the embedding for better similarity search
            embedding = embedding / np.linalg.norm(embedding)
            
            return torch.tensor(embedding)
            
        except Exception as e:
            print(f"❌ Error in _get_embedding: {e}")
            print(f"   Text length: {len(text)}")
            print(f"   Text preview: {text[:100]}...")
            raise


class HRDocumentRAG:
    """Main RAG system for HR documents"""
    
    def __init__(self, 
            chunks_file: str = "",
            pdf_path: str = "",
            chroma_db_path: str = "./chroma_db",
            collection_name: str = "hr_documents"):
        
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize embeddings
        print("🔧 Initializing embeddings...")
        self.embeddings = CustomEmbeddings()
        
        # Initialize vector store as None (will be set later)
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
        # Load Groq API key from environment
        if not self.groq_api_key:
            print("⚠️ Warning: GROQ_API_KEY not found in environment variables.")
            print("   Please ensure your .env file contains GROQ_API_KEY=your_key_here")
        else:
            print("✅ Groq API key loaded from environment")
        
        # Initialize chunk data
        self.subtopics = []
        self.chapters = []
        
        # Load or create chunks
        if chunks_file:
            self.load_chunks_from_file(chunks_file)
        elif pdf_path:
            self.create_chunks_from_pdf(pdf_path)
        else:
            raise ValueError("Either chunks_file or pdf_path must be provided")
    
    def load_chunks_from_file(self, filename: str):
        """Load existing chunks from JSON file"""
        print(f"📖 Loading chunks from {filename}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.subtopics = data.get('subtopics', [])
            self.chapters = data.get('chapters', [])
            
            print(f"✅ Loaded {len(self.subtopics)} subtopics and {len(self.chapters)} chapters")
            
            # Debug: Print sample content
            if self.subtopics:
                print(f"📝 Sample subtopic content: {self.subtopics[0].get('text', '')[:100]}...")
            if self.chapters:
                print(f"📚 Sample chapter content: {self.chapters[0].get('text', '')[:100]}...")
                
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            raise
    
    def create_chunks_from_pdf(self, pdf_path: str):
        """Create chunks from PDF using your existing chunker"""
        print(f"📄 Processing PDF: {pdf_path}")
        
        try:
            chunker = PDFChunker(pdf_path)
            toc = chunker.parse_toc_structure()
            subtopics, chapters = chunker.extract_subtopic_and_chapter_chunks(toc)
            chunker.close()
            
            # Convert to dict format for consistency
            self.subtopics = [sub.__dict__ for sub in subtopics]
            self.chapters = [chap.__dict__ for chap in chapters]
            
            print(f"✅ Created {len(self.subtopics)} subtopics and {len(self.chapters)} chapters")
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            raise
    
    def create_langchain_documents(self) -> List[Document]:
        """Convert chunks to LangChain Documents"""
        print("🔄 Converting chunks to LangChain Documents...")
        
        documents = []
        
        # Process subtopics
        for i, subtopic in enumerate(self.subtopics):
            try:
                # Create rich metadata
                metadata = {
                    "type": "subtopic",
                    "chunk_id": f"subtopic_{i}",
                    "chapter_id": subtopic.get("chapter_id", "unknown"),
                    "chapter_title": subtopic.get("chapter_title", "unknown"),
                    "subtopic_title": subtopic.get("subtopic_title", "unknown"),
                    "page_range": f"{subtopic.get('page_range', [0, 0])[0]}-{subtopic.get('page_range', [0, 0])[1]}",
                    "source": "hr_document",
                    "content_type": "policy"
                }
                
                # Create document with enhanced content
                content = f"""Chapter: {subtopic.get('chapter_title', 'Unknown')}
Section: {subtopic.get('subtopic_title', 'Unknown')}
Content: {subtopic.get('text', '')}""".strip()
                
                if not content or len(content.strip()) < 10:
                    print(f"⚠️ Skipping empty/short subtopic content at index {i}")
                    continue
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"❌ Error processing subtopic {i}: {e}")
                continue
        
        # Process chapters (with size limit to avoid overwhelming)
        for i, chapter in enumerate(self.chapters):
            try:
                metadata = {
                    "type": "chapter",
                    "chunk_id": f"chapter_{i}",
                    "chapter_id": chapter.get("chapter_id", "unknown"),
                    "chapter_title": chapter.get("chapter_title", "unknown"),
                    "page_range": f"{chapter.get('page_range', [0, 0])[0]}-{chapter.get('page_range', [0, 0])[1]}",
                    "source": "hr_document",
                    "content_type": "policy",
                    "subtopic_count": len(chapter.get("subtopics", []))
                }
                
                # Create summarized chapter content (limit size)
                chapter_text = chapter.get('text', '')
                if len(chapter_text) > 3000:  # Limit chapter content size
                    chapter_text = chapter_text[:3000] + "..."
                
                content = f"""Chapter: {chapter.get('chapter_title', 'Unknown')}
Summary: This chapter contains {len(chapter.get('subtopics', []))} subtopics covering various HR policies.
Content: {chapter_text}""".strip()
                
                if not content or len(content.strip()) < 10:
                    print(f"⚠️ Skipping empty/short chapter content at index {i}")
                    continue
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"❌ Error processing chapter {i}: {e}")
                continue
        
        print(f"✅ Created {len(documents)} LangChain documents")
        
        # Debug: Show sample document
        if documents:
            print(f"📝 Sample document content: {documents[0].page_content[:200]}...")
            print(f"📝 Sample document metadata: {documents[0].metadata}")
        
        return documents
    
    def setup_vector_store(self, documents: List[Document]):
        """Create and populate ChromaDB vector store"""
        print("🗄️ Setting up ChromaDB vector store...")
        
        if not documents:
            raise ValueError("No documents provided for vector store setup")
        
        try:
            # Remove existing vector store if it exists
            if os.path.exists(self.chroma_db_path):
                import shutil
                shutil.rmtree(self.chroma_db_path)
                print(f"🗑️ Removed existing vector store at {self.chroma_db_path}")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_path,
                collection_name=self.collection_name
            )
            
            # Persist the database
            self.vectorstore.persist()
            print(f"✅ Vector store created with {len(documents)} documents")
            
            # Verify the vector store has documents
            collection = self.vectorstore._collection
            doc_count = collection.count()
            print(f"✅ Vector store verification - contains {doc_count} documents")
            
            # Test the vector store
            test_results = self.vectorstore.similarity_search("test query", k=1)
            print(f"✅ Vector store test successful - found {len(test_results)} results")
            
        except Exception as e:
            print(f"❌ Error setting up vector store: {e}")
            raise
    
    def load_existing_vector_store(self):
        """Load existing ChromaDB vector store"""
        print("📂 Loading existing vector store...")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test the loaded vector store
            collection = self.vectorstore._collection
            doc_count = collection.count()
            print(f"✅ Vector store loaded successfully - contains {doc_count} documents")
            
            if doc_count == 0:
                print("⚠️ WARNING: Vector store is empty! This will cause issues.")
                return False
            
            # Test search functionality
            test_results = self.vectorstore.similarity_search("test", k=1)
            print(f"✅ Vector store search test successful - found {len(test_results)} results")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def setup_retrieval_chain(self, 
                            groq_model: str = "llama-3.3-70b-versatile",
                            temperature: float = 0.1,
                            max_tokens: int = 1024):
        """Setup the retrieval QA chain with Groq API"""
        print("🔗 Setting up retrieval chain with Groq...")
        
        # Check if vector store is initialized
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call setup_vector_store() or load_existing_vector_store() first.")
        
        # Check if Groq API key is available
        if not self.groq_api_key:
            raise ValueError("Groq API key not found. Please ensure your .env file contains GROQ_API_KEY=your_key_here")
        
        try:
            # Setup retriever with corrected parameters (removed score_threshold)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5  # Number of documents to retrieve
                }
            )
            
            # Test retriever with invoke method instead of deprecated get_relevant_documents
            test_docs = self.retriever.invoke("test query")
            print(f"✅ Retriever test successful - found {len(test_docs)} documents")
            
            # Initialize Groq LLM
            llm = ChatGroq(
                api_key=self.groq_api_key, # type: ignore
                model=groq_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✅ Groq LLM initialized with model: {groq_model}")
            
            # Enhanced prompt template for HR queries
            prompt_template = """You are an HR assistant that helps employees understand company policies and procedures.
Use ONLY the following context to answer the question. Do not use any external knowledge.

Context: {context}

Question: {question}

Instructions:
- Answer ONLY based on the provided context
- If the context doesn't contain relevant information, say "I don't have information about that in the provided documents"
- Be specific and reference the relevant sections
- Provide direct quotes when helpful
- Keep responses professional and helpful

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create the RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            print("✅ Retrieval chain setup complete with Groq integration")
            
        except Exception as e:
            print(f"❌ Error setting up retrieval chain: {e}")
            raise
    
    def debug_search(self, query: str, k: int = 5):
        """Debug search functionality"""
        print(f"\n🔍 DEBUG: Searching for '{query}'")
        
        if not self.vectorstore:
            print("❌ Vector store not initialized")
            return []
        
        try:
            # Test similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"✅ Found {len(results)} documents")
            
            for i, doc in enumerate(results):
                print(f"\n--- Document {i+1} ---")
                print(f"Type: {doc.metadata.get('type', 'unknown')}")
                print(f"Chapter: {doc.metadata.get('chapter_title', 'unknown')}")
                print(f"Content preview: {doc.page_content[:150]}...")
                
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call setup_vector_store() first.")
        
        print(f"🔍 Searching for: '{query}'")
        results = self.vectorstore.similarity_search(query, k=k)
        print(f"✅ Found {len(results)} documents")
        
        return results
    
    def query(self, question: str, return_source_docs: bool = True):
        """Query the RAG system with LLM integration"""
        print(f"🔍 Processing query: {question}")
        
        # First, test document retrieval
        print("📖 Testing document retrieval...")
        test_docs = self.search_documents(question, k=3)
        
        if not test_docs:
            print("⚠️ No documents found for query - this indicates an embedding mismatch issue")
            return {
                "answer": "No relevant documents found. This may indicate an issue with the embedding system.",
                "source_documents": [],
                "num_sources": 0
            }
        
        if not self.qa_chain:
            print("⚠️ QA chain not initialized. Returning document search results only...")
            return test_docs
        
        try:
            # Run the query through the full RAG pipeline
            result = self.qa_chain({"query": question})
            
            response = {
                "answer": result["result"],
                "source_documents": result["source_documents"] if return_source_docs else None,
                "num_sources": len(result["source_documents"])
            }
            
            print(f"✅ Query processed successfully using {response['num_sources']} source documents")
            return response
            
        except Exception as e:
            print(f"❌ Error during query processing: {e}")
            # Fallback to document search
            print("🔄 Falling back to document search...")
            return test_docs
    
    def query_with_details(self, question: str):
        """Query with detailed response including source information"""
        result = self.query(question, return_source_docs=True)
        
        if isinstance(result, dict):
            print(f"\n📋 Answer: {result['answer']}")
            print(f"\n📚 Sources ({result['num_sources']} documents):")
            
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\n--- Source {i} ---")
                print(f"Type: {doc.metadata.get('type', 'unknown')}")
                print(f"Chapter: {doc.metadata.get('chapter_title', 'unknown')}")
                if doc.metadata.get('type') == 'subtopic':
                    print(f"Section: {doc.metadata.get('subtopic_title', 'unknown')}")
                print(f"Pages: {doc.metadata.get('page_range', 'unknown')}")
                print(f"Content Preview: {doc.page_content[:200]}...")
        else:
            print(f"\n📋 Fallback search results: {len(result)} documents found")
            for i, doc in enumerate(result, 1):
                print(f"\n--- Document {i} ---")
                print(f"Content: {doc.page_content[:200]}...")
        
        return result
    
    def build_rag_system(self, 
                        force_rebuild: bool = False,
                        groq_model: str = "llama3-8b-8192",
                        temperature: float = 0.1,
                        max_tokens: int = 1024):
        """Complete RAG system setup with Groq integration"""
        print("🚀 Building RAG system with Groq...")
        
        # Check if vector store already exists and is not empty
        vector_store_loaded = False
        if os.path.exists(self.chroma_db_path) and not force_rebuild:
            print("📂 Found existing vector store, loading...")
            vector_store_loaded = self.load_existing_vector_store()
            
            if not vector_store_loaded:
                print("⚠️ Existing vector store is empty or corrupted, rebuilding...")
                force_rebuild = True
        
        if not os.path.exists(self.chroma_db_path) or force_rebuild or not vector_store_loaded:
            print("🔨 Creating new vector store...")
            documents = self.create_langchain_documents()
            
            if not documents:
                raise ValueError("No documents created - check your chunk data")
            
            self.setup_vector_store(documents)
        
        # Setup retrieval chain with Groq
        self.setup_retrieval_chain(
            groq_model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print("✅ RAG system with Groq integration ready!")
        
        # Run a quick test
        print("\n🧪 Running system test...")
        test_query = "What are the policies?"
        self.debug_search(test_query, k=3)
    
    def get_stats(self):
        """Get system statistics"""
        stats = {
            "total_subtopics": len(self.subtopics),
            "total_chapters": len(self.chapters),
            "vector_store_initialized": self.vectorstore is not None,
            "qa_chain_initialized": self.qa_chain is not None,
            "groq_api_configured": self.groq_api_key is not None
        }
        
        if self.vectorstore:
            try:
                # Get collection stats
                collection = self.vectorstore._collection
                stats["documents_in_vectorstore"] = collection.count()
            except:
                stats["documents_in_vectorstore"] = "unknown"
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Starting HR Document RAG System...")
    
    # Initialize RAG system (Groq API key loaded from .env automatically)
    rag = HRDocumentRAG(
        chunks_file="your_chunks.json",  # or pdf_path="your_document.pdf"
        chroma_db_path="./hr_chroma_db",
        collection_name="hr_policies"
    )
    
    # Build the system with Groq integration (force rebuild to ensure fresh embeddings)
    rag.build_rag_system(
        force_rebuild=True,  # Set to True to rebuild vector store
        groq_model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=1024
    )
    
    # Print system stats
    print(f"\n📊 System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test queries with debugging
    test_queries = [
        "What is the employee onboarding process?",
        "How are performance evaluations conducted?",
        "What are the leave policies?",
        "What is the dress code policy?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        # First run debug search to see what documents are found
        debug_results = rag.debug_search(query, k=3)
        
        if debug_results:
            print(f"\n🔍 Found {len(debug_results)} documents, proceeding with full query...")
            # Get full response with LLM
            response = rag.query_with_details(query)
        else:
            print("❌ No documents found - check your embeddings and data")