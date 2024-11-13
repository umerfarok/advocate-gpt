from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from src.utils.memory_utils import MemoryManager

class QASystem:
    def __init__(self):
        self.device = MemoryManager.select_device()
        # Using a smaller T5 model for efficiency
        self.model_name = "t5-base"
        
        print(f"Loading QA model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

    def generate_answer(self, question, context):
        # Limit context length to manage memory
        max_context_length = 1024
        if len(context) > max_context_length:
            context = context[:max_context_length]

        prompt = f"""
        Based on the following Pakistani law context, answer the question.
        If you're not sure or the context doesn't contain relevant information, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                min_length=50,
                temperature=0.7,
                num_return_sequences=1,
                do_sample=True,
                no_repeat_ngram_size=2
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up memory
        del inputs, outputs
        MemoryManager.optimize_memory()
        
        return answer