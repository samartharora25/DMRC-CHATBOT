# src/langchain_rag.py

import json
import torch
from typing import List, Dict, Any
from dataclasses import dataclass

# LangChain imports
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

# from langchain.llms import LlamaCpp  # or whatever local LLM you're using
# from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# HuggingFace imports
from transformers import AutoTokenizer, AutoModel

# Your existing classes
from src.chunker import PDFChunker


class CustomEmbeddings(Embeddings):
    """Custom LangChain embeddings wrapper for your local BGE model"""
    
    def __init__(self, model_path: str = "models/BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ Using CPU for embeddings")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self._get_embedding(text)
        return embedding.tolist()
    
    @torch.no_grad()
    def _get_embedding(self, text: str):
        """Get embedding for a single text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        # Use CLS token embedding
        return outputs.last_hidden_state[:, 0].cpu().numpy()[0]


class HRDocumentRAG:
    """Main RAG system for HR documents"""
    
    def __init__(self, 
            chunks_file: str = "",
            pdf_path: str = "",
            chroma_db_path: str = "./chroma_db",
            collection_name: str = "hr_documents"):
        
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = CustomEmbeddings()
        
        # Initialize vector store as None (will be set later)
        self.vectorstore = None
        self.retriever = None
        
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
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.subtopics = data.get('subtopics', [])
        self.chapters = data.get('chapters', [])
        
        print(f"✅ Loaded {len(self.subtopics)} subtopics and {len(self.chapters)} chapters")
    
    def create_chunks_from_pdf(self, pdf_path: str):
        """Create chunks from PDF using your existing chunker"""
        print(f"📄 Processing PDF: {pdf_path}")
        
        chunker = PDFChunker(pdf_path)
        toc = chunker.parse_toc_structure()
        subtopics, chapters = chunker.extract_subtopic_and_chapter_chunks(toc)
        chunker.close()
        
        # Convert to dict format for consistency
        self.subtopics = [sub.__dict__ for sub in subtopics]
        self.chapters = [chap.__dict__ for chap in chapters]
        
        print(f"✅ Created {len(self.subtopics)} subtopics and {len(self.chapters)} chapters")
    
    def create_langchain_documents(self) -> List[Document]:
        """Convert chunks to LangChain Documents"""
        print("🔄 Converting chunks to LangChain Documents...")
        
        documents = []
        
        # Process subtopics
        for subtopic in self.subtopics:
            # Create rich metadata
            metadata = {
                "type": "subtopic",
                "chapter_id": subtopic["chapter_id"],
                "chapter_title": subtopic["chapter_title"],
                "subtopic_title": subtopic["subtopic_title"],
                "page_range": f"{subtopic['page_range'][0]}-{subtopic['page_range'][1]}",
                "source": "hr_document",
                "content_type": "policy"
            }
            
            # Create document with enhanced content
            content = f"""
Chapter: {subtopic['chapter_title']}
Section: {subtopic['subtopic_title']}
Content: {subtopic['text']}
            """.strip()
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        # Optionally add chapter-level documents for broader context
        for chapter in self.chapters:
            metadata = {
                "type": "chapter",
                "chapter_id": chapter["chapter_id"],
                "chapter_title": chapter["chapter_title"],
                "page_range": f"{chapter['page_range'][0]}-{chapter['page_range'][1]}",
                "source": "hr_document",
                "content_type": "policy",
                "subtopic_count": len(chapter.get("subtopics", []))
            }
            
            # Create summarized chapter content
            content = f"""
Chapter: {chapter['chapter_title']}
Summary: This chapter contains {len(chapter.get('subtopics', []))} subtopics covering various HR policies.
Full Content: {chapter['text'][:1000]}...
            """.strip()
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} LangChain documents")
        return documents
    
    def setup_vector_store(self, documents: List[Document]):
        """Create and populate ChromaDB vector store"""
        print("🗄️ Setting up ChromaDB vector store...")
        
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
    
    def load_existing_vector_store(self):
        """Load existing ChromaDB vector store"""
        print("📂 Loading existing vector store...")
        
        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        print("✅ Vector store loaded successfully")
    
    def setup_retrieval_chain(self, llm_model_path: str = ""):
        """Setup the retrieval QA chain"""
        print("🔗 Setting up retrieval chain...")
        
        # Check if vector store is initialized
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call setup_vector_store() or load_existing_vector_store() first.")
        
        # Setup retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Return top 3 most relevant chunks
        )
        
        # Custom prompt template for HR queries
        prompt_template = """
You are an HR assistant that helps employees understand company policies and procedures.
Use the following context to answer the question accurately and helpfully.

Context: {context}

Question: {question}

Instructions:
- Provide accurate information based solely on the context provided
- If information is not available in the context, clearly state so
- Be concise but comprehensive
- Include relevant policy section references when applicable
- Use a professional, helpful tone

Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # For now, we'll use a simple setup without LLM
        # You can add your local LLM here
        print("⚠️ Note: Add your local LLM configuration here")
        
        self.retriever = retriever
        
        print("✅ Retrieval chain setup complete")
    
    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call setup_vector_store() first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def query(self, question: str):
        """Query the RAG system"""
        print(f"🔍 Searching for: {question}")
        
        # Get relevant documents
        relevant_docs = self.search_documents(question, k=3)
        
        print(f"📋 Found {len(relevant_docs)} relevant documents:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Type: {doc.metadata.get('type', 'unknown')}")
            print(f"Chapter: {doc.metadata.get('chapter_title', 'unknown')}")
            if doc.metadata.get('type') == 'subtopic':
                print(f"Section: {doc.metadata.get('subtopic_title', 'unknown')}")
            print(f"Pages: {doc.metadata.get('page_range', 'unknown')}")
            print(f"Content Preview: {doc.page_content[:200]}...")
        
        return relevant_docs
    
    def build_rag_system(self, force_rebuild: bool = False):
        """Complete RAG system setup"""
        print("🚀 Building RAG system...")
        
        # Check if vector store already exists
        import os
        if os.path.exists(self.chroma_db_path) and not force_rebuild:
            print("📂 Found existing vector store, loading...")
            self.load_existing_vector_store()
        else:
            print("🔨 Creating new vector store...")
            documents = self.create_langchain_documents()
            self.setup_vector_store(documents)
        
        # Setup retrieval chain
        # self.setup_retrieval_chain()
        
        print("✅ RAG system ready!")
    
    def get_stats(self):
        """Get system statistics"""
        stats = {
            "total_subtopics": len(self.subtopics),
            "total_chapters": len(self.chapters),
            "vector_store_initialized": self.vectorstore is not None
        }
        
        if self.vectorstore:
            # Get collection stats
            collection = self.vectorstore._collection
            stats["documents_in_vectorstore"] = collection.count()
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = HRDocumentRAG(
        chunks_file="your_chunks.json",  # or pdf_path="your_document.pdf"
        chroma_db_path="./hr_chroma_db",
        collection_name="hr_policies"
    )
    
    # Build the system
    rag.build_rag_system()
    
    # Test queries
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
        
        results = rag.query(query)
        
        # Here you would normally pass to your LLM for final answer generation
        print(f"\n💡 Retrieved {len(results)} relevant sections for this query")
    
    # Print system stats
    print(f"\n📊 System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")