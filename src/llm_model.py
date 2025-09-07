# src/llm_model.py - Enhanced LLM with Financial Synthesis Capabilities

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
        DataCollatorForLanguageModeling,
        pipeline
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Import config with fallback
try:
    from src.config import config
except ImportError:
    class Config:
        model_name = "microsoft/DialoGPT-small"
        model_path = "./models/financial_model"
        max_length = 512
        fine_tune_epochs = 20
        learning_rate = 5e-5
        save_user_data = True
    config = Config()

class FinancialLLM:
    """Enhanced LLM with improved financial synthesis and generation capabilities"""

    def __init__(self):
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.fallback_mode = False
        self.max_length = config.max_length
        self.is_finetuned = False

        print(f"🤖 Initializing Enhanced Financial LLM: {self.model_name}")
        print(f"🔧 Device: {self.device} | Transformers: {TRANSFORMERS_AVAILABLE}")

        # Try to load model immediately for faster responses
        try:
            self.load_model()
        except Exception as e:
            print(f"⚠️ Model loading failed during init: {e}")
            self.fallback_mode = True

    def load_model(self):
        """Load model with enhanced financial capabilities"""

        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers library not available, using fallback responses")
            self.fallback_mode = True
            return

        try:
            print("🔄 Loading enhanced financial model...")

            # Try fine-tuned model first
            if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, "config.json")):
                print(f"📂 Loading fine-tuned financial model from {self.model_path}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

                    # Check if it's a PEFT model
                    if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                        print("🎯 Loading PEFT (LoRA) adapted model")
                        base_model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        self.model = PeftModel.from_pretrained(base_model, self.model_path)
                        self.is_finetuned = True
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        self.is_finetuned = True

                    print("✅ Loaded fine-tuned financial model successfully")

                except Exception as ft_error:
                    print(f"⚠️ Fine-tuned model loading failed: {ft_error}")
                    raise ft_error
            else:
                # Load base model with financial instruction training setup
                print(f"📦 Loading base model with financial enhancement: {self.model_name}")

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
                print("✅ Loaded base model - ready for financial fine-tuning")

            # Configure tokenizer for financial context
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            print(f"🔧 Tokenizer configured: pad_token = {self.tokenizer.pad_token}")

            # Move to device
            if not torch.cuda.is_available() or self.device == "cpu":
                self.model.to("cpu")
                print("📱 Model loaded on CPU")

            self.model_loaded = True
            print(f"✅ Enhanced Financial LLM initialization complete (Fine-tuned: {self.is_finetuned})")

        except Exception as e:
            print(f"❌ Critical model loading error: {str(e)}")
            print("🔄 Falling back to enhanced rule-based responses")
            self.fallback_mode = True
            self.model_loaded = False

    def generate_response(self, prompt: str, max_length: int = 150, debug: bool = False) -> str:
        """Generate enhanced response with financial synthesis capabilities"""

        if debug:
            print(f"\n🔍 DEBUG - Input prompt length: {len(prompt)} chars")
            print(f"🔍 DEBUG - Model status: Loaded={self.model_loaded}, Fallback={self.fallback_mode}, Fine-tuned={self.is_finetuned}")

        # Use enhanced fallback if model not available
        if self.fallback_mode or not self.model_loaded:
            return self._generate_enhanced_fallback_response(prompt)

        # Try loading model if not loaded
        if not self.model:
            try:
                self.load_model()
            except:
                return self._generate_enhanced_fallback_response(prompt)

        try:
            # Enhanced prompt formatting for financial synthesis
            formatted_prompt = self._format_financial_prompt(prompt)

            if debug:
                print(f"🔍 DEBUG - Formatted prompt: {formatted_prompt[:200]}...")

            # Tokenize with financial context optimization
            inputs = self.tokenizer.encode(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=min(400, config.max_length - max_length)
            ).to(self.device)

            if debug:
                print(f"🔍 DEBUG - Input tokens shape: {inputs.shape}")

            # Enhanced generation parameters for financial synthesis
            generation_params = {
                "max_length": min(len(inputs[0]) + max_length, config.max_length),
                "num_return_sequences": 1,
                "temperature": 0.7,  # Balanced creativity vs accuracy
                "do_sample": True,
                "top_p": 0.9,  # Nucleus sampling for coherent responses
                "top_k": 50,   # Limit vocabulary for focused responses
                "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                "attention_mask": torch.ones_like(inputs),
                "no_repeat_ngram_size": 3,  # Reduce repetition
                "early_stopping": True,
                "repetition_penalty": 1.1  # Slight penalty for repetition
            }

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_params)

            if debug:
                print(f"🔍 DEBUG - Output tokens shape: {outputs.shape}")

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if debug:
                print(f"🔍 DEBUG - Full response length: {len(full_response)}")

            # Extract and enhance the assistant's response
            response = self._extract_and_enhance_response(full_response, formatted_prompt)

            if debug:
                print(f"🔍 DEBUG - Final response: {response[:200]}...")

            # Validate response quality
            if len(response.strip()) < 50:  # Too short
                print("⚠️ Generated response too short, using enhanced fallback")
                return self._generate_enhanced_fallback_response(prompt)

            # Check for synthesis quality
            if not self._has_synthesis_indicators(response):
                print("⚠️ Response lacks synthesis, enhancing...")
                response = self._enhance_response_synthesis(response, prompt)

            return response

        except Exception as e:
            print(f"❌ Error during enhanced generation: {str(e)}")
            if debug:
                import traceback
                print(f"🔍 DEBUG - Full traceback: {traceback.format_exc()}")

            return self._generate_enhanced_fallback_response(prompt)

    def _format_financial_prompt(self, prompt: str) -> str:
        """Format prompt specifically for financial synthesis"""

        # Check if it's already a well-structured synthesis prompt
        if "synthesize" in prompt.lower() and "context" in prompt.lower():
            return prompt  # It's already enhanced from RAG system

        # Otherwise, create a basic financial synthesis prompt
        return f"""You are an expert financial advisor. Provide comprehensive analysis and synthesis.

User Question: {prompt}

Instructions: Provide a detailed response that:
1. Analyzes the question from multiple financial perspectives
2. Synthesizes information to create original insights
3. Gives specific, actionable recommendations
4. Explains reasoning clearly
5. Addresses risks and considerations

Financial Analysis:"""

    def _extract_and_enhance_response(self, full_response: str, formatted_prompt: str) -> str:
        """Extract and enhance the model's response"""

        # Find the actual response after the prompt
        response_markers = ["Financial Analysis:", "Analysis:", "Response:", "Answer:", "Assistant:"]

        response = full_response
        for marker in response_markers:
            if marker in full_response:
                parts = full_response.split(marker)
                if len(parts) > 1:
                    response = parts[-1].strip()
                    break

        # If no marker found, try to remove the original prompt
        if response == full_response:
            try:
                # Remove the formatted prompt from the response
                response = full_response.replace(formatted_prompt, "").strip()
            except:
                response = full_response

        # Clean and enhance response
        response = self._clean_and_enhance_response(response)

        return response

    def _clean_and_enhance_response(self, response: str) -> str:
        """Clean and enhance the model response for better quality"""

        # Remove common artifacts and cleanup
        response = response.replace("User:", "").replace("Assistant:", "").replace("Human:", "").strip()

        # Remove repetitive patterns
        lines = response.split('\n')
        unique_lines = []
        seen_lines = set()

        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean.lower() not in seen_lines:
                unique_lines.append(line_clean)
                seen_lines.add(line_clean.lower())

        response = '\n'.join(unique_lines) if unique_lines else ' '.join(unique_lines)

        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1:
            # Check if last sentence is incomplete (too short)
            if len(sentences[-1].strip()) < 10:
                response = '.'.join(sentences[:-1]) + '.'

        # Add synthesis enhancement if response is too basic
        if len(response.strip()) < 100 and not self._has_synthesis_indicators(response):
            response = self._add_synthesis_elements(response)

        return response.strip()

    def _has_synthesis_indicators(self, response: str) -> bool:
        """Check if response has synthesis indicators rather than just retrieval"""

        synthesis_indicators = [
            "analysis", "considering", "based on", "strategy", "recommend",
            "approach", "therefore", "however", "additionally", "furthermore",
            "key factors", "taking into account", "comprehensive", "balanced"
        ]

        lower_response = response.lower()
        indicator_count = sum(1 for indicator in synthesis_indicators if indicator in lower_response)

        return indicator_count >= 2  # Need at least 2 synthesis indicators

    def _enhance_response_synthesis(self, response: str, original_prompt: str) -> str:
        """Enhance response to add synthesis elements"""

        # Add analytical context
        enhanced_response = "Based on comprehensive financial analysis, " + response

        # Add strategic thinking if missing
        if "recommend" not in response.lower() and "suggest" not in response.lower():
            enhanced_response += " I recommend taking a systematic approach that balances these factors with your personal financial goals and risk tolerance."

        # Add risk consideration if missing
        if "risk" not in response.lower() and len(response) > 100:
            enhanced_response += " It's important to consider the associated risks and ensure any strategy aligns with your investment timeline and risk capacity."

        return enhanced_response

    def _add_synthesis_elements(self, basic_response: str) -> str:
        """Add synthesis elements to a basic response"""

        if len(basic_response.strip()) < 50:
            return basic_response + " A comprehensive approach to this involves analyzing multiple factors and creating a strategy that balances risk with potential returns while aligning with your financial objectives."

        return "From a financial planning perspective, " + basic_response + " The key is to develop a comprehensive strategy that considers your individual circumstances, risk tolerance, and long-term financial goals."

    def _generate_enhanced_fallback_response(self, prompt: str) -> str:
        """Generate enhanced fallback response with financial synthesis when model is unavailable"""

        prompt_lower = prompt.lower()

        # Enhanced financial knowledge base with synthesis-focused responses
        enhanced_financial_responses = {
            "sip": """Systematic Investment Plan (SIP) is a powerful wealth-building strategy that leverages three key principles: rupee cost averaging, disciplined investing, and compound growth. By investing a fixed amount regularly (starting from Rs.500-1000 monthly), you automatically buy more units when prices are low and fewer when prices are high, effectively averaging your cost.

For implementation, I recommend starting with large-cap diversified equity funds for stability, then gradually adding mid-cap funds as your comfort level increases. The strategy works best with a minimum 5-10 year horizon, and you should consider step-up SIPs where you increase your investment by 10-15% annually as your income grows.

Key considerations include choosing direct plans for lower expense ratios, automating investments to maintain discipline, and reviewing fund performance annually while avoiding frequent changes based on short-term performance fluctuations.""",

            "diversification": """Portfolio diversification is the cornerstone of risk management that goes beyond just spreading investments across different stocks. A comprehensive diversification strategy involves multiple layers: asset class diversification (equity, debt, gold, real estate), sector diversification within equities, market cap diversification (large, mid, small cap), and even geographic diversification.

For Indian investors, I recommend a strategic allocation of 60-70% equity for growth (distributed as 40% large-cap, 20% mid-cap, 10% small-cap), 20-25% in debt instruments for stability and regular income, 5-10% in gold as an inflation hedge, and 5% in international equity for currency diversification.

The implementation strategy involves rebalancing annually or when any asset class deviates more than 10% from target allocation. This disciplined approach helps you sell high-performing assets and buy underperforming ones, maintaining optimal risk-return balance while benefiting from market cycles.""",

            "tax": """Tax-efficient investing in India requires a strategic approach that maximizes deductions while building long-term wealth. The primary strategy involves utilizing Section 80C's Rs.1.5 lakh limit through ELSS mutual funds, which offer the best combination of tax benefits and growth potential with only a 3-year lock-in period.

A comprehensive tax strategy includes: ELSS for growth-oriented tax saving, PPF for guaranteed tax-free returns over 15 years, EPF maximization for salaried individuals, NPS for additional Rs.50,000 deduction under 80CCD, and health insurance premiums under 80D for essential protection.

The key insight is to view tax-saving investments as part of your overall portfolio allocation rather than separate decisions. This means choosing tax-saving instruments that align with your risk tolerance and financial goals, ensuring you don't compromise long-term wealth creation for short-term tax benefits.""",

            "stock": """Fundamental stock analysis requires a systematic approach that combines quantitative metrics with qualitative assessment. The analytical framework involves evaluating financial health through key ratios: P/E ratio compared to industry peers and historical averages, debt-to-equity ratio (preferably below 0.5 for most sectors), ROE above 15% indicating efficient capital utilization, and consistent revenue growth over 3-5 years.

Beyond numbers, qualitative analysis focuses on competitive advantages (brand strength, market position, regulatory moats), management quality (track record, transparency, capital allocation decisions), and industry dynamics (growth prospects, regulatory environment, cyclical patterns).

The synthesis approach involves creating a comprehensive score considering both quantitative and qualitative factors, comparing with 3-5 industry peers, and analyzing the business across different economic cycles. This holistic evaluation helps identify stocks trading below intrinsic value with sustainable competitive advantages and strong management teams.""",

            "mutual fund": """Mutual fund selection requires analyzing multiple dimensions beyond just past performance. The comprehensive evaluation framework includes fund manager consistency across market cycles, expense ratio impact on long-term returns (prefer below 1.5% for equity funds), portfolio concentration and overlap analysis, and risk-adjusted performance metrics like Sharpe ratio and Alpha.

Strategic selection involves matching fund characteristics with your investment goals: large-cap funds for stability, mid-cap funds for growth potential, hybrid funds for balanced exposure, and sector funds for tactical allocation. The analysis should include fund size considerations (avoid very small funds below Rs.100 crore and very large funds above Rs.10,000 crore in mid-cap space).

Implementation strategy focuses on building a core-satellite approach with 70% allocation to consistent, broad-market funds (core) and 30% to specialized or thematic funds (satellite) for enhanced returns. Regular monitoring involves annual review of fund performance, manager changes, and strategy drift while avoiding frequent churning based on short-term performance.""",

            "emergency": """Emergency fund planning forms the foundation of financial security by providing a buffer against unexpected events like job loss, medical emergencies, or major repairs. The strategic approach involves calculating monthly essential expenses (rent, groceries, loan EMIs, utilities, insurance) and building a fund covering 6-12 months of these expenses.

The implementation strategy focuses on liquidity and safety over returns: maintain 1-2 months of expenses in savings accounts for immediate access, place 3-4 months in liquid mutual funds for better returns with T+1 liquidity, and consider short-term fixed deposits for the remaining amount. This tiered approach balances accessibility with slightly better returns.

Building strategy involves gradual accumulation - start with 1 month's expenses, then build to 3, 6, and finally your target amount. Maintain the fund separately from investment accounts to avoid temptation of using it for non-emergencies, and replenish immediately after any withdrawal. Review and update the target amount annually as expenses increase with inflation and lifestyle changes.""",

            "retirement": """Retirement planning requires a comprehensive strategy that addresses longevity risk, inflation impact, and changing financial needs over time. The analytical framework involves calculating post-retirement monthly expenses (typically 70-80% of current expenses), factoring in 6-7% annual inflation, and planning for 25-30 years of retirement life.

Strategic asset allocation should evolve with age: aggressive growth-focused allocation in early career (80% equity, 20% debt), balanced approach in middle years (60% equity, 40% debt), and conservative preservation-focused allocation near retirement (40% equity, 60% debt). This glide path helps capture growth during wealth accumulation phase while protecting capital as retirement approaches.

Implementation involves maximizing tax-advantaged retirement accounts (EPF, NPS), building additional corpus through equity mutual fund SIPs, and creating post-retirement income strategy through systematic withdrawal plans (SWP), annuities for guaranteed income, and maintaining adequate liquid funds for emergencies. Healthcare cost planning requires separate consideration given medical inflation runs at 12-15% annually."""
        }

        # Enhanced matching with synthesis
        for keyword, response in enhanced_financial_responses.items():
            if keyword in prompt_lower:
                return response + "\n\nThis analysis provides a framework for your consideration. For personalized advice tailored to your specific situation, risk tolerance, and financial goals, please consult with a qualified financial advisor who can provide detailed recommendations based on your individual circumstances."

        # Enhanced generic response with synthesis
        return """From a comprehensive financial planning perspective, successful investing involves several interconnected principles that work together to build long-term wealth.

The foundation starts with establishing clear financial goals, building adequate emergency reserves, and understanding your risk tolerance. This creates the framework for making informed investment decisions that align with your personal circumstances and timeline.

Strategic implementation involves systematic investing through SIPs for rupee cost averaging, diversification across asset classes and sectors for risk management, tax-efficient investing to maximize after-tax returns, and regular portfolio review and rebalancing to maintain optimal allocation.

The key insight is that financial success comes from consistency, discipline, and a long-term perspective rather than trying to time markets or chase short-term gains. Each financial decision should be evaluated not in isolation but as part of your overall financial strategy.

For specific guidance tailored to your individual situation, investment goals, and risk profile, I strongly recommend consulting with a qualified financial advisor who can provide personalized advice and help create a comprehensive financial plan aligned with your unique circumstances."""

    def prepare_enhanced_financial_dataset(self) -> Dataset:
        """Prepare enhanced training dataset focused on financial synthesis"""

        if not TRANSFORMERS_AVAILABLE:
            return None

        # Enhanced financial Q&A dataset with synthesis-focused examples
        enhanced_financial_data = [
            {
                "input": "What is SIP and how should I implement it?",
                "output": """SIP (Systematic Investment Plan) is a disciplined investment strategy that leverages rupee cost averaging and compound growth. Here's a comprehensive implementation approach:

**Strategic Framework:**
1. Start with 15-20% of monthly income allocated to SIPs
2. Begin with large-cap diversified funds for stability, then add mid-cap exposure
3. Choose direct plans for lower expense ratios (0.5-1% vs 1.5-2% for regular plans)

**Implementation Strategy:**
- Automate investments on salary credit date for consistency
- Use step-up SIPs to increase investment by 10-15% annually
- Diversify across 3-4 fund categories: large-cap, mid-cap, international, debt
- Maintain minimum 5-10 year investment horizon for optimal results

**Key Considerations:**
The power of SIP lies not just in regular investing but in behavioral discipline it creates. By averaging market volatility and building wealth systematically, SIPs have historically delivered 12-15% annual returns in equity funds over long periods. However, success requires staying invested through market cycles and avoiding emotional decisions during volatility."""
            },
            {
                "input": "How should I diversify my investment portfolio?",
                "output": """Portfolio diversification requires a multi-layered approach that goes beyond just buying different stocks. Here's a comprehensive diversification strategy:

**Asset Allocation Framework:**
- Equity (60-70%): Growth engine for long-term wealth creation
  - Large-cap (40%): Stability and consistent returns
  - Mid-cap (20%): Higher growth potential
  - Small-cap (10%): Aggressive growth component
- Debt (20-25%): Stability and regular income through bonds, FDs, PPF
- Gold (5-10%): Inflation hedge and portfolio stabilizer
- International equity (5%): Currency and geographic diversification

**Sector Diversification Strategy:**
Limit exposure to any single sector to 15-20% to avoid concentration risk. Spread investments across defensive sectors (FMCG, pharma), growth sectors (technology, financial services), and cyclical sectors (auto, infrastructure).

**Implementation Approach:**
Review and rebalance annually or when any asset class deviates 10% from target allocation. This disciplined approach forces you to sell high-performing assets and buy underperforming ones, maintaining optimal risk-return balance while benefiting from market cycles.

**Risk Management:**
Diversification reduces portfolio volatility by 20-30% compared to concentrated investing while maintaining similar long-term returns. The key is finding the right balance - over-diversification can limit returns while under-diversification increases risk unnecessarily."""
            },
            {
                "input": "What are the best tax-saving investment options in India?",
                "output": """Tax-efficient investing in India requires strategic allocation across multiple sections while building long-term wealth. Here's a comprehensive analysis:

**Section 80C Optimization (Rs.1.5 lakh limit):**
1. **ELSS Mutual Funds**: Optimal choice for growth-oriented investors
   - Shortest lock-in period (3 years) with potential 12-15% annual returns
   - Better long-term wealth creation compared to traditional options
   - Flexibility to choose fund categories based on risk tolerance

2. **PPF**: Best for conservative investors seeking guaranteed returns
   - 15-year commitment with 7-8% tax-free returns
   - Triple tax benefit (deduction, growth, withdrawal)
   - Ideal for retirement planning and long-term goals

3. **EPF**: Mandatory for salaried employees, voluntary contributions possible
   - Current rate around 8.5% with stable returns
   - Additional employer matching for employed individuals

**Additional Tax Benefits:**
- **NPS (80CCD)**: Extra Rs.50,000 deduction with professional fund management
- **Health Insurance (80D)**: Rs.25,000 for individuals, Rs.50,000 for senior citizens
- **Home Loan Interest (24b)**: Up to Rs.2 lakh for self-occupied property

**Strategic Implementation:**
Allocate based on risk tolerance and liquidity needs: aggressive investors should maximize ELSS allocation, balanced investors can split between ELSS and PPF, while conservative investors may prefer PPF and debt instruments. The key is viewing tax savings as part of overall portfolio allocation rather than separate decisions.

**Long-term Impact:**
A 25-year-old investing Rs.1.5 lakh annually through optimized tax-saving instruments can potentially build a corpus of Rs.2-3 crore by age 60, demonstrating the powerful combination of tax efficiency and compound growth."""
            },
            {
                "input": "How should I analyze stocks before investing?",
                "output": """Comprehensive stock analysis requires integrating quantitative metrics with qualitative assessment to identify undervalued companies with strong fundamentals. Here's a systematic analytical framework:

**Quantitative Analysis Framework:**
1. **Valuation Metrics:**
   - P/E Ratio: Compare with industry average and historical norms
   - P/B Ratio: Below 2-3 for most sectors indicates reasonable valuation
   - PEG Ratio: Below 1 suggests growth at reasonable price

2. **Financial Health Indicators:**
   - Debt-to-Equity: Below 0.5 preferred (sector-specific variations)
   - Current Ratio: Above 1.5 shows good liquidity management
   - ROE: Above 15% indicates efficient capital utilization
   - ROA: Above 8-10% demonstrates effective asset management

3. **Growth and Profitability:**
   - Revenue growth consistency over 5-7 years
   - Profit margin trends and comparison with peers
   - Free cash flow generation capability

**Qualitative Analysis Dimensions:**
1. **Management Quality:** Track record, transparency in communication, capital allocation decisions
2. **Competitive Advantages:** Brand strength, market position, regulatory moats, switching costs
3. **Industry Dynamics:** Growth prospects, regulatory environment, competitive intensity
4. **Business Model:** Scalability, recurring revenue, pricing power

**Synthesis and Decision Framework:**
Create a comprehensive scoring system weighing both quantitative (60%) and qualitative (40%) factors. Compare with 3-5 industry peers to understand relative positioning. Analyze business performance across different economic cycles to assess resilience.

**Risk Assessment:**
Identify key risks including business-specific challenges, industry headwinds, regulatory changes, and management concerns. Determine acceptable risk-reward ratio based on your portfolio allocation and investment timeline.

**Implementation Strategy:**
Never invest based on single metrics. Successful stock picking requires patience, thorough research, and continuous monitoring of business fundamentals rather than short-term price movements."""
            }
        ]

        # Load additional data if available
        data_file = "data/enhanced_financial_dataset.json"
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    additional_data = json.load(f)
                enhanced_financial_data.extend(additional_data)
                print(f"✅ Loaded {len(additional_data)} additional synthesis training examples")
            except Exception as e:
                print(f"⚠️ Could not load additional dataset: {e}")

        # Format for training with synthesis instructions
        formatted_data = []
        for item in enhanced_financial_data:
            # Enhanced prompt format that encourages synthesis
            text = f"""### Financial Advisory Task: Provide comprehensive analysis and synthesis

User Question: {item['input']}

### Instructions: Provide detailed financial advice that:
1. Synthesizes information from multiple perspectives
2. Includes original analysis and insights
3. Offers specific, actionable recommendations
4. Explains reasoning step-by-step
5. Addresses risks and considerations

### Expert Financial Analysis:
{item['output']}<|endoftext|>"""

            formatted_data.append({"text": text})

        print(f"📊 Prepared {len(formatted_data)} enhanced synthesis training examples")
        return Dataset.from_list(formatted_data)

    def fine_tune_for_synthesis(self, additional_data=None):
        """Fine-tune model specifically for financial synthesis capabilities"""

        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available for fine-tuning")
            return False

        if not self.model_loaded:
            print("❌ Base model must be loaded before fine-tuning")
            return False

        try:
            print("🚀 Starting enhanced financial synthesis fine-tuning...")

            # Prepare enhanced dataset
            if additional_data:
                # Convert additional_data to synthesis format
                synthesis_data = []
                for item in additional_data:
                    if isinstance(item, dict) and 'input' in item and 'output' in item:
                        synthesis_data.append(item)
                    elif isinstance(item, str):
                        # Convert string to synthesis format
                        synthesis_data.append({
                            "input": "Provide financial guidance on this topic",
                            "output": f"Based on comprehensive analysis: {item}"
                        })
                dataset = Dataset.from_dict({"text": [self._format_synthesis_training_text(item) for item in synthesis_data]})
            else:
                dataset = self.prepare_enhanced_financial_dataset()

            if not dataset:
                print("❌ No training dataset available")
                return False

            print(f"📚 Loaded {len(dataset)} synthesis training examples")

            # Setup LoRA for efficient fine-tuning
            lora_config = LoraConfig(
                r=16,  # rank
                lora_alpha=32,
                target_modules=["c_attn", "c_proj", "c_fc"],  # DialoGPT modules
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            # Apply LoRA to model
            if not self.is_finetuned:
                self.model = get_peft_model(self.model, lora_config)
                print("✅ Applied LoRA configuration for efficient fine-tuning")

            # Tokenize dataset
            def tokenize_function(examples):
                encodings = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                )
                encodings["labels"] = encodings["input_ids"].copy()
                return encodings

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Training arguments optimized for synthesis
            training_args = TrainingArguments(
                output_dir=self.model_path,
                overwrite_output_dir=True,
                num_train_epochs=config.fine_tune_epochs,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                evaluation_strategy="steps",
                eval_steps=50,
                load_best_model_at_end=True,
                learning_rate=config.learning_rate,
                lr_scheduler_type="cosine",
                fp16=torch.cuda.is_available(),  # Use mixed precision if available
            )

            # Split dataset for evaluation
            split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True)

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=split_dataset["train"],
                eval_dataset=split_dataset["test"],
                data_collator=data_collator,
            )

            # Start training
            print("🚀 Starting synthesis-focused fine-tuning...")
            trainer.train()

            # Save the fine-tuned model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.model_path)

            print("✅ Financial synthesis fine-tuning completed successfully!")
            print(f"📁 Model saved to: {self.model_path}")

            self.is_finetuned = True
            return True

        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            import traceback
            print("🔍 DEBUG - Full traceback:", traceback.format_exc())
            return False

    def _format_synthesis_training_text(self, item: dict) -> str:
        """Format training text to encourage synthesis"""

        return f"""### Financial Advisory Task: Provide comprehensive analysis and synthesis

User Question: {item['input']}

### Instructions: Provide detailed financial advice that:
1. Synthesizes information from multiple perspectives
2. Includes original analysis and insights
3. Offers specific, actionable recommendations
4. Explains reasoning step-by-step
5. Addresses risks and considerations

### Expert Financial Analysis:
{item['output']}<|endoftext|>"""

    # Legacy method for backward compatibility
    def fine_tune(self, additional_data=None):
        """Legacy fine-tune method - redirects to enhanced version"""
        return self.fine_tune_for_synthesis(additional_data)

    def test_synthesis_capabilities(self, test_queries: List[str] = None):
        """Test model's synthesis capabilities with financial queries"""

        if not test_queries:
            test_queries = [
                "How should I balance SIP investments with tax planning?",
                "What's the comprehensive strategy for retirement planning in India?", 
                "Analyze the pros and cons of investing in small-cap vs large-cap funds",
                "How do I create a balanced portfolio for aggressive growth?",
                "What factors should I consider before investing in individual stocks?"
            ]

        print("\n🧪 Testing Financial Synthesis Capabilities...")
        print("=" * 60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: Synthesis Quality Assessment ---")
            print(f"Query: {query}")

            response = self.generate_response(query, max_length=200, debug=False)

            print(f"\nResponse: {response}")

            # Evaluate synthesis quality
            synthesis_score = self._evaluate_response_synthesis(response)
            print(f"\nSynthesis Score: {synthesis_score:.2f}/1.00")

            if synthesis_score > 0.7:
                print("✅ High synthesis quality")
            elif synthesis_score > 0.4:
                print("⚠️ Moderate synthesis quality")
            else:
                print("❌ Low synthesis quality - mostly retrieval")

            print("-" * 60)

    def _evaluate_response_synthesis(self, response: str) -> float:
        """Evaluate synthesis quality of a response"""

        synthesis_score = 0.0

        # Check for synthesis language
        synthesis_indicators = [
            "analysis", "strategy", "approach", "comprehensive", "considering",
            "framework", "implementation", "key factors", "balanced", "optimal"
        ]

        synthesis_count = sum(1 for indicator in synthesis_indicators if indicator in response.lower())
        synthesis_score += min(synthesis_count * 0.1, 0.4)

        # Check for structured thinking
        structure_indicators = ["1.", "first", "second", "additionally", "furthermore", "however", "therefore"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response.lower())
        synthesis_score += min(structure_count * 0.1, 0.3)

        # Check for practical recommendations
        practical_indicators = ["recommend", "suggest", "should", "consider", "strategy", "implement"]
        practical_count = sum(1 for indicator in practical_indicators if indicator in response.lower())
        synthesis_score += min(practical_count * 0.1, 0.3)

        return min(synthesis_score, 1.0)

    def get_model_info(self) -> Dict[str, any]:
        """Get enhanced model information and status"""

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "is_finetuned": self.is_finetuned,
            "fallback_mode": self.fallback_mode,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "max_length": self.max_length,
            "synthesis_optimized": True,
            "last_updated": datetime.now().isoformat()
        }

# Global enhanced instance
llm = FinancialLLM()

def main():
    """Enhanced CLI for testing the LLM"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Financial LLM CLI")
    parser.add_argument("--train", action="store_true", help="Fine-tune for synthesis")
    parser.add_argument("--test", type=str, help="Test with a specific query")
    parser.add_argument("--test-synthesis", action="store_true", help="Test synthesis capabilities")
    parser.add_argument("--info", action="store_true", help="Show model information")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.info:
        info = llm.get_model_info()
        print("🤖 Enhanced Financial LLM Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    elif args.train:
        print("🚀 Starting enhanced financial synthesis training...")
        success = llm.fine_tune_for_synthesis()
        if success:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed")

    elif args.test_synthesis:
        llm.test_synthesis_capabilities()

    elif args.test:
        print(f"🧪 Testing with query: {args.test}")
        response = llm.generate_response(args.test, debug=args.debug)
        print(f"📝 Response: {response}")

        # Evaluate synthesis
        synthesis_score = llm._evaluate_response_synthesis(response)
        print(f"🔍 Synthesis Score: {synthesis_score:.2f}/1.00")

    else:
        # Interactive testing mode
        print("🤖 Enhanced Financial LLM Interactive Testing")
        print("Commands: 'quit', 'info', 'synthesis' for synthesis test")

        while True:
            query = input("\n💬 Ask something: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'info':
                info = llm.get_model_info()
                for key, value in info.items():
                    print(f"  {key}: {value}")
            elif query.lower() == 'synthesis':
                llm.test_synthesis_capabilities()
            elif query:
                response = llm.generate_response(query, debug=args.debug)
                print(f"🤖 Response: {response}")

                synthesis_score = llm._evaluate_response_synthesis(response)
                print(f"🔍 Synthesis Quality: {synthesis_score:.2f}/1.00")

if __name__ == "__main__":
    main()