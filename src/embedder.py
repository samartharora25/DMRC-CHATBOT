from transformers import AutoTokenizer, AutoModel
import torch
import json
import torch.version

cuda_enabled = torch.cuda.is_available()
cuda_device = torch.cuda.current_device() if cuda_enabled else None
cuda_version = torch.version.cuda if cuda_enabled else None

if cuda_enabled:
    print(f"✅ CUDA is enabled. Device: {torch.cuda.get_device_name(cuda_device)}")
    model_path = "models/BAAI/bge-large-en-v1.5"
    print("🔁 Loading BAAI/bge-large-en-v1.5 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to("cuda").eval()
    print("✅ Model loaded successfully.")
    
else:
    print("❌ CUDA is not enabled. Using CPU.")

class Embedder:
    def __init__(self, subtopics, chapters):
        self.tokenizer = tokenizer
        self.model = model
        self.subtopics = subtopics
        self.chapters = chapters
        self.embedded_chunks = []

    @torch.no_grad()
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0].cpu().numpy()[0]

    def embed(self):
        self.embedded_chunks = []
        for chunk in self.subtopics:
            emb = self.get_embedding(chunk.text)
            self.embedded_chunks.append({
                "chapter": chunk.chapter,
                "title": chunk.title,
                "page_range": chunk.page_range,
                "text": chunk.text,
                "embedding": emb.tolist()
            })
        print("✅ Embedding complete.")
    
    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()
        print("🔁 Model unloaded and GPU memory cleared.")
    
    def save(self, filename="embedded_chunks.json"):
        with open(filename, "w") as f:
            json.dump(self.embedded_chunks, f, indent=2)
        print(f"✅ Saved embeddings to {filename}")
