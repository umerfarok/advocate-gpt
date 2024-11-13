import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from tqdm import tqdm
from utils.memory_utils import BatchProcessor
import nltk
from typing import List, Dict

class PDFProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for memory efficiency
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.batch_processor = BatchProcessor(batch_size=4)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.,;?!-]', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        # Basic sentence correction
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        
        return text
    
    def process_single_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Process a single PDF file"""
        try:
            reader = PdfReader(pdf_path)
            raw_text = ""
            
            # Extract text from PDF
            for page in reader.pages:
                raw_text += page.extract_text() + "\n"
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Add metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'text': chunk,
                    'source': os.path.basename(pdf_path),
                    'chunk_id': f"{os.path.basename(pdf_path)}_{i}",
                    'page_count': len(reader.pages)
                })
            
            return processed_chunks
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """Process all PDFs in a directory"""
        all_chunks = []
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(directory_path, pdf_file)
            chunks = self.process_single_pdf(pdf_path)
            all_chunks.extend(chunks)
            
            # Clean up memory periodically
            if len(all_chunks) % 1000 == 0:
                import gc
                gc.collect()
        
        return all_chunks

    def save_processed_chunks(self, chunks: List[Dict[str, str]], output_path: str):
        """Save processed chunks to file"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)