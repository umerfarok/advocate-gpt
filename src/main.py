import os
import torch
import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from src.data_processing.pdf_processor import PDFProcessor
from src.model.embeddings_manager import EmbeddingsManager
from src.utils.memory_utils import MemoryManager

def setup_system():
    print(f"Using device: {MemoryManager.select_device()}")
    print(f"Available RAM: {MemoryManager.get_available_memory():.2f}GB")
    if torch.cuda.is_available():
        print(f"Available GPU Memory: {MemoryManager.get_gpu_memory():.2f}GB")
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    embeddings_manager = EmbeddingsManager()
    
    # Process PDFs
    chunks = pdf_processor.process_directory('data/law_books')
    print(f"Processed {len(chunks)} text chunks")
    
    # Save processed chunks
    pdf_processor.save_processed_chunks(chunks, 'data/processed_chunks.json')
    
    # Create embeddings
    embeddings_manager.create_embeddings(chunks)
    
    # Save vector store
    embeddings_manager.save('vector_store')
    
    print("Setup complete!")

if __name__ == "__main__":
    setup_system()