# src/llm_model.py - Fixed LLM with Better Error Handling and Debug

import torch
import os
import logging
from typing import List, Dict, Optional
import json
from datetime import datetime

# Disable transformers warnings that might interfere
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Import config with fallback
try:
    from src.config import config
except ImportError:
    class Config:
        model_name = "microsoft/DialoGPT-small"
        model_path = "./models/financial_model"
        max_length = 512
        fine_tune_epochs = 3
        learning_rate = 5e-5
        save_user_data = True
    config = Config()

class LightweightFinancialLLM:
    """Enhanced LLM with better error handling and debugging"""
    
    def __init__(self):
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.fallback_mode = False
        self.max_length = config.max_length
        
        print(f"🤖 Initializing {self.model_name} on {self.device}")
        print(f"🔧 Transformers available: {TRANSFORMERS_AVAILABLE}")
        
        # Try to load model immediately for faster responses
        try:
            self.load_model()
        except Exception as e:
            print(f"⚠️  Model loading failed during init: {e}")
            self.fallback_mode = True
    
    def load_model(self):
        """Load model with comprehensive error handling"""
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers library not available, using fallback responses")
            self.fallback_mode = True
            return
        
        try:
            print("🔄 Loading model...")
            
            # Try fine-tuned model first
            if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, "config.json")):
                print(f"📂 Loading fine-tuned model from {self.model_path}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,  # Force float32 for compatibility
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    print("✅ Loaded fine-tuned model successfully")
                except Exception as ft_error:
                    print(f"⚠️  Fine-tuned model failed: {ft_error}")
                    raise ft_error
            else:
                # Load base model
                print(f"📦 Loading base model: {self.model_name}")
                
                # Try different model loading strategies
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    print("✅ Loaded base model successfully")
                    
                except Exception as base_error:
                    print(f"❌ Base model loading failed: {base_error}")
                    # Try with minimal configuration
                    print("🔄 Trying minimal model configuration...")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        use_fast=False
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        low_cpu_mem_usage=True
                    )
                    print("✅ Loaded model with minimal configuration")
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    
                print(f"🔧 Set pad token: {self.tokenizer.pad_token}")
            
            # Move to device
            if not torch.cuda.is_available() or self.device == "cpu":
                self.model.to("cpu")
                print("📱 Model loaded on CPU")
            
            self.model_loaded = True
            print("✅ Model initialization complete")
            
        except Exception as e:
            print(f"❌ Critical model loading error: {str(e)}")
            print("🔄 Falling back to rule-based responses")
            self.fallback_mode = True
            self.model_loaded = False
    
    def generate_response(self, prompt: str, max_length: int = 100, debug: bool = True) -> str:
        """Generate response with debugging and fallback"""
        
        if debug:
            print(f"\n🔍 DEBUG - Input prompt: {prompt[:100]}...")
            print(f"🔍 DEBUG - Model loaded: {self.model_loaded}")
            print(f"🔍 DEBUG - Fallback mode: {self.fallback_mode}")
        
        # Use fallback if model not available
        if self.fallback_mode or not self.model_loaded:
            return self._generate_fallback_response(prompt)
        
        # Try loading model if not loaded
        if not self.model:
            try:
                self.load_model()
            except:
                return self._generate_fallback_response(prompt)
        
        try:
            # Format prompt for financial context
            formatted_prompt = f"User: {prompt}\nFinancial Assistant:"
            
            if debug:
                print(f"🔍 DEBUG - Formatted prompt: {formatted_prompt[:150]}...")
            
            # Tokenize
            inputs = self.tokenizer.encode(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=min(400, config.max_length - max_length)
            ).to(self.device)
            
            if debug:
                print(f"🔍 DEBUG - Input tokens shape: {inputs.shape}")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + max_length, config.max_length),
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                    attention_mask=torch.ones_like(inputs),
                    no_repeat_ngram_size=2  # Reduce repetition
                )
            
            if debug:
                print(f"🔍 DEBUG - Output tokens shape: {outputs.shape}")
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if debug:
                print(f"🔍 DEBUG - Full response: {full_response}")
            
            # Extract just the assistant's response
            if "Financial Assistant:" in full_response:
                response = full_response.split("Financial Assistant:")[-1].strip()
            elif "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
            else:
                # If no clear separator, take everything after the prompt
                response = full_response.replace(formatted_prompt, "").strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            if debug:
                print(f"🔍 DEBUG - Cleaned response: {response}")
            
            # Validate response
            if len(response.strip()) < 10:
                print("⚠️  Generated response too short, using fallback")
                return self._generate_fallback_response(prompt)
            
            return response
            
        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            if debug:
                import traceback
                print(f"🔍 DEBUG - Full traceback: {traceback.format_exc()}")
            
            return self._generate_fallback_response(prompt)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the model response"""
        
        # Remove common artifacts
        response = response.replace("User:", "").replace("Assistant:", "").strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        
        response = ' '.join(unique_lines)
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure minimum quality
        if len(response.strip()) < 20:
            return "I understand you're asking about financial topics. Let me provide some general guidance based on sound financial principles."
        
        return response.strip()
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate rule-based fallback response when model is unavailable"""
        
        prompt_lower = prompt.lower()
        
        # Financial knowledge base responses
        financial_responses = {
            "sip": "SIP (Systematic Investment Plan) is a disciplined way to invest in mutual funds. You invest a fixed amount regularly (monthly/quarterly) which helps with rupee cost averaging. Start with Rs.500-1000 monthly based on your income.",
            
            "diversification": "Diversification is key to reducing investment risk. Spread your investments across different asset classes (equity, debt, gold), market caps (large, mid, small), and sectors. A good starting allocation for young investors is 70% equity, 20% debt, 10% gold.",
            
            "tax": "For tax savings under Section 80C, consider ELSS mutual funds (3-year lock-in), PPF (15-year), or EPF. ELSS offers potential for highest returns. You can also use NPS for additional Rs.50,000 deduction under 80CCD.",
            
            "stock": "Before investing in individual stocks, analyze the company's fundamentals: P/E ratio, debt-to-equity ratio, revenue growth, and ROE. Compare these metrics with industry peers. Also consider technical factors like price trends and volume.",
            
            "mutual fund": "Mutual funds are professionally managed investment vehicles that pool money from many investors. They offer instant diversification and are suitable for beginners. Choose funds based on your risk tolerance and investment horizon.",
            
            "emergency fund": "Build an emergency fund covering 6-12 months of expenses before investing in risky assets. Keep this in liquid instruments like savings accounts or liquid funds for easy access during emergencies.",
            
            "portfolio": "A well-balanced portfolio should align with your risk tolerance, investment goals, and time horizon. Regular review and rebalancing help maintain your desired asset allocation. Don't put all money in one stock or sector."
        }
        
        # Check for keyword matches
        for keyword, response in financial_responses.items():
            if keyword in prompt_lower:
                return f"{response} Please remember this is general educational information and you should consult a qualified financial advisor for personalized advice."
        
        # Generic helpful response
        return "I understand you're seeking financial guidance. Based on general financial principles, it's important to have clear investment goals, maintain diversification, invest regularly through SIPs, and build an emergency fund. For specific advice tailored to your situation, please consult with a qualified financial advisor."
    
    def prepare_financial_dataset(self) -> Dataset:
        """Prepare training dataset with enhanced financial knowledge"""
        
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        # Enhanced financial Q&A dataset
        financial_data = [
            {
                "input": "What is SIP and how does it work?",
                "output": "SIP or Systematic Investment Plan allows you to invest a fixed amount regularly in mutual funds. It provides rupee cost averaging benefits and instills disciplined investing. You can start with as low as Rs.500 monthly and increase gradually."
            },
            {
                "input": "How should I diversify my investment portfolio?",
                "output": "Diversify across asset classes (70% equity, 20% debt, 10% gold for young investors), market capitalizations (large, mid, small cap), and sectors. Rebalance annually when allocation deviates by 5-10% from target."
            },
            {
                "input": "What are the best tax-saving investment options?",
                "output": "Under Section 80C: ELSS mutual funds (3-year lock-in, potential 10-15% returns), PPF (15-year lock-in, 7-8% tax-free returns), EPF contributions. Additional Rs.50,000 through NPS under Section 80CCD. ELSS offers shortest lock-in with market-linked returns."
            },
            {
                "input": "ETF vs Mutual Fund - which is better?",
                "output": "ETFs have lower expense ratios (0.1-0.5%) compared to mutual funds (1-2%) but require a demat account and real-time trading. Mutual funds offer SIP convenience and professional management. For beginners, start with mutual fund SIPs for systematic investing."
            },
            {
                "input": "How to analyze a stock before investing?",
                "output": "Analyze fundamentals: P/E ratio (compare with industry), debt-to-equity ratio (<0.5 preferred), ROE (>15% good), revenue growth consistency. Check technical indicators: price trends, support/resistance levels. Always compare with industry peers and analyze 3-5 year trends."
            },
            {
                "input": "Should I invest in stocks or mutual funds?",
                "output": "For beginners, mutual funds are better due to professional management and instant diversification. Direct stock investing requires research skills and time. Start with large-cap mutual funds through SIP, then gradually add mid-cap funds. Consider stocks only after gaining experience."
            },
            {
                "input": "What is an emergency fund and how much should I keep?",
                "output": "Emergency fund should cover 6-12 months of expenses in liquid investments like savings accounts or liquid funds. Build it gradually, starting with 1 month's expenses. Don't invest emergency funds in equity markets. Use only for genuine emergencies."
            },
            {
                "input": "How to start investing with a small amount?",
                "output": "Start with mutual fund SIPs from Rs.500 monthly. Choose large-cap funds initially for stability. Gradually increase amount as income grows. Build emergency fund first, then start long-term investments. Use apps like Zerodha Coin or Groww for easy investing."
            }
        ]
        
        # Load additional data if available
        data_file = "data/financial_dataset.json"
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    additional_data = json.load(f)
                    financial_data.extend(additional_data)
                    print(f"✅ Loaded {len(additional_data)} additional training examples")
            except Exception as e:
                print(f"⚠️  Could not load additional dataset: {e}")
        
        # Format for training
        formatted_data = []
        for item in financial_data:
            text = f"User: {item['input']}\nFinancial Assistant: {item['output']}<|endoftext|>"
            formatted_data.append({"text": text})
        
        print(f"📊 Prepared {len(formatted_data)} training examples")
        return Dataset.from_list(formatted_data)
    
    def fine_tune(self, additional_data):
        try:
            print("🚀 Starting model fine-tuning...")

            # --- Load dataset ---
            print("📚 Preparing dataset for fine-tuning...")
            dataset = Dataset.from_dict({"text": additional_data})
            print(f"✅ Loaded {len(additional_data)} additional training examples")

            # --- Tokenize dataset ---
            def tokenize_function(examples):
                encodings = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",   # ensures consistent shape
                    max_length=self.max_length
                )
                # set labels = input_ids for causal LM
                encodings["labels"] = encodings["input_ids"].copy()
                return encodings

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]  # 🔑 remove raw strings!
            )
            print(f"📊 Prepared {len(tokenized_dataset)} training examples")

            # --- Setup data collator ---
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # --- Initialize Trainer ---
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir="./results",
                    overwrite_output_dir=True,
                    num_train_epochs=3,
                    per_device_train_batch_size=2,
                    save_steps=10,
                    save_total_limit=2,
                    logging_dir="./logs",
                    logging_steps=5,
                ),
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # --- Start training ---
            print("🚀 Starting fine-tuning...")
            trainer.train()
            print("✅ Fine-tuning complete!")

        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            import traceback
            print("🔍 DEBUG - Full traceback:", traceback.format_exc())
    
    def test_model(self, test_queries: List[str] = None):
        """Test model with sample queries"""
        
        if not test_queries:
            test_queries = [
                "What is SIP?",
                "How to diversify portfolio?",
                "Best tax saving options?",
                "Should I invest in stocks or mutual funds?"
            ]
        
        print("\n🧪 Testing model responses...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i} ---")
            print(f"Query: {query}")
            
            response = self.generate_response(query, max_length=150, debug=True)
            
            print(f"Response: {response}")
            print("-" * 50)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information and status"""
        
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "fallback_mode": self.fallback_mode,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "last_updated": datetime.now().isoformat()
        }

# Global instance
llm = LightweightFinancialLLM()

def main():
    """CLI for testing the LLM"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial LLM CLI")
    parser.add_argument("--train", action="store_true", help="Fine-tune the model")
    parser.add_argument("--test", type=str, help="Test with a specific query")
    parser.add_argument("--info", action="store_true", help="Show model information")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.info:
        info = llm.get_model_info()
        print("🤖 Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    elif args.train:
        print("🚀 Starting model fine-tuning...")
        llm.fine_tune(additional_data="data/financial_dataset.json")
    
    elif args.test:
        print(f"🧪 Testing with query: {args.test}")
        response = llm.generate_response(args.test, debug=args.debug)
        print(f"📝 Response: {response}")
    
    else:
        # Interactive testing mode
        print("🤖 Financial LLM Interactive Testing")
        print("Type 'quit' to exit, 'info' for model info")
        
        while True:
            query = input("\n💬 Ask something: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'info':
                info = llm.get_model_info()
                for key, value in info.items():
                    print(f"  {key}: {value}")
            elif query:
                response = llm.generate_response(query, debug=args.debug)
                print(f"🤖 Response: {response}")

if __name__ == "__main__":
    main()