from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import pickle
import os
from tqdm import tqdm
from utils.memory_utils import BatchProcessor, MemoryManager

class EmbeddingsManager:
    def __init__(self):
        self.device = MemoryManager.select_device()
        # Using a smaller multilingual model for efficiency
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=self.device)
        self.vector_size = 384  # Size for this model
        self.batch_processor = BatchProcessor(batch_size=32)
        self.index = None
        self.texts = []

    def create_embeddings(self, chunks):
        print("Creating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        
        def process_batch(batch):
            with torch.no_grad():
                return self.model.encode(batch, show_progress_bar=False)
        
        # Process in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_processor.batch_size)):
            batch = texts[i:i + self.batch_processor.batch_size]
            batch_embeddings = process_batch(batch)
            embeddings.extend(batch_embeddings)
            
            if i % (self.batch_processor.batch_size * 4) == 0:
                MemoryManager.optimize_memory()
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.vector_size)
        self.index.add(embeddings)
        self.texts = chunks
        
        print(f"Created embeddings for {len(texts)} chunks")

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
        
        # Save texts in batches to manage memory
        with open(os.path.join(directory, 'texts.pkl'), 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self, directory):
        print("Loading embeddings...")
        self.index = faiss.read_index(os.path.join(directory, 'index.faiss'))
        with open(os.path.join(directory, 'texts.pkl'), 'rb') as f:
            self.texts = pickle.load(f)

    def search(self, query, k=3):
        # Encode query
        query_vector = self.model.encode([query])
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):  # Safety check
                results.append({
                    'text': self.texts[idx]['text'],
                    'source': self.texts[idx]['source'],
                    'score': float(score)
                })
        
        return results