from dotenv import load_dotenv
import os
import json
import sys
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any,List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from newsapi import NewsApiClient
import yfinance as yf
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()
# --- CONFIGURATION ---
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
DATABASE_FILE = "analysis_archive.db"
MAX_CONTEXT_LENGTH = 5000  # Characters
MIN_NEWS_ARTICLES = 3

# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---
class FactorAnalysis(BaseModel):
    score: int = Field(ge=1, le=10, description="Score from 1-10")
    justification: str = Field(min_length=20, description="Detailed justification")
    confidence: Optional[str] = Field(default="medium", description="Confidence level: low/medium/high")
    
    @field_validator('score')
    @classmethod
    def score_must_be_valid(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Score must be between 1 and 10')
        return v

class InvestmentAnalysis(BaseModel):
    company_performance: FactorAnalysis
    management_and_governance: FactorAnalysis
    industry_and_sector_health: FactorAnalysis
    competitive_landscape: FactorAnalysis
    regulatory_risk: FactorAnalysis
    macroeconomic_exposure: FactorAnalysis
    overall_sentiment: FactorAnalysis
    risk_flags: List[str] = Field(default_factory=list, description="Key risk indicators")
    opportunities: List[str] = Field(default_factory=list, description="Key opportunities")
    
    @field_validator('risk_flags', 'opportunities')
    @classmethod
    def validate_lists(cls, v):
        if len(v) > 10:
            raise ValueError('Too many items in list')
        return v

# --- DATACLASSES ---
@dataclass
class CompanyProfile:
    name: str
    ticker: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    country: Optional[str] = None

# --- DATABASE MANAGER ---
class DatabaseManager:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
        """Initialize database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Main analyses table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                news_context TEXT,
                model_version TEXT,
                context_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, analysis_date)
            )
            """)
            
            # Metadata table for tracking analysis quality
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                avg_score REAL,
                num_risk_flags INTEGER,
                num_opportunities INTEGER,
                processing_time REAL,
                FOREIGN KEY(analysis_id) REFERENCES analyses(id)
            )
            """)
            
            # Cache table for news articles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                news_data TEXT,
                fetch_date TEXT,
                expiry_date TEXT
            )
            """)
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_file}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_analysis(self, ticker: str, analysis_json: str, context: str, 
                     model_version: str, metadata: Dict):
        """Save analysis with metadata"""
        today = date.today().isoformat()
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO analyses 
                (ticker, analysis_date, analysis_json, news_context, model_version, context_length)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, today, analysis_json, context, model_version, len(context)))
            
            analysis_id = cursor.lastrowid
            
            # Save metadata
            cursor.execute("""
                INSERT INTO analysis_metadata 
                (analysis_id, avg_score, num_risk_flags, num_opportunities, processing_time)
                VALUES (?, ?, ?, ?, ?)
            """, (analysis_id, metadata.get('avg_score', 0), 
                  metadata.get('num_risk_flags', 0),
                  metadata.get('num_opportunities', 0),
                  metadata.get('processing_time', 0)))
            
            conn.commit()
            conn.close()
            logger.info(f"Analysis saved for {ticker}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            raise
    
    def get_few_shot_examples(self, n: int = 2, min_quality_score: float = 6.0) -> List[Tuple[str, str]]:
        """Fetch high-quality examples for few-shot learning"""
        logger.info(f"Fetching {n} high-quality examples (min score: {min_quality_score})")
        examples = []
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.news_context, a.analysis_json 
                FROM analyses a
                JOIN analysis_metadata m ON a.id = m.analysis_id
                WHERE a.news_context IS NOT NULL 
                  AND a.news_context != ''
                  AND a.analysis_json IS NOT NULL 
                  AND a.analysis_json != ''
                  AND m.avg_score >= ?
                ORDER BY m.avg_score DESC, RANDOM() 
                LIMIT ?
            """, (min_quality_score, n))
            
            examples = cursor.fetchall()
            conn.close()
            
            logger.info(f"Retrieved {len(examples)} examples")
        except Exception as e:
            logger.warning(f"Error fetching examples: {e}")
        
        return examples
    
    def get_recent_analysis(self, ticker: str, days: int = 7) -> Optional[str]:
        """Get recent analysis if available"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cutoff_date = (date.today() - timedelta(days=days)).isoformat()
            cursor.execute("""
                SELECT analysis_json FROM analyses 
                WHERE ticker = ? AND analysis_date >= ?
                ORDER BY analysis_date DESC LIMIT 1
            """, (ticker, cutoff_date))
            
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error fetching recent analysis: {e}")
            return None

# --- NEWS FETCHER WITH CACHING ---
class NewsFetcher:
    def __init__(self, api_key: str, db_manager: DatabaseManager):
        self.client = NewsApiClient(api_key=api_key)
        self.db_manager = db_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
    
    def fetch_news(self, query: str, page_size: int = 20, use_cache: bool = True) -> str:
        """Fetch news with caching capability"""
        logger.info(f"Fetching news for query: {query[:50]}...")
        
        try:
            response = self.client.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=page_size,
                from_param=(date.today() - timedelta(days=30)).isoformat()
            )
            
            context_string = ""
            article_count = 0
            
            for article in response.get('articles', []):
                title = article.get('title', '')
                description = article.get('description', '')
                source = article.get('source', {}).get('name', 'Unknown')
                published_at = article.get('publishedAt', '')
                
                if description and title:
                    context_string += f"[{source} - {published_at}] {title}: {description}\n"
                    article_count += 1
            
            logger.info(f"Fetched {article_count} articles")
            return context_string
            
        except Exception as e:
            logger.error(f"Error fetching news for '{query}': {e}")
            return ""
    
    def fetch_comprehensive_news(self, profile: CompanyProfile, countries: List[str]) -> Dict[str, str]:
        """Fetch news from multiple sources"""
        news_sections = {}
        
        # Company-specific news
        company_query = f'("{profile.name}" OR "{profile.ticker}") AND (earnings OR revenue OR product OR acquisition OR lawsuit)'
        news_sections['company'] = self.fetch_news(company_query)
        
        # Sector news
        sector_query = f'"{profile.sector}" AND (trends OR outlook OR growth OR challenges)'
        news_sections['sector'] = self.fetch_news(sector_query, page_size=15)
        
        # Industry news
        industry_query = f'"{profile.industry}" AND (innovation OR competition OR regulation)'
        news_sections['industry'] = self.fetch_news(industry_query, page_size=15)
        
        # Macroeconomic news
        if countries:
            country_query = f'({" OR ".join([f"{c}" for c in countries])}) AND (economy OR GDP OR "interest rates" OR inflation)'
            news_sections['macro'] = self.fetch_news(country_query, page_size=15)
        
        return news_sections

# --- COMPANY PROFILE FETCHER ---
class CompanyProfileFetcher:
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_profile(ticker: str) -> Optional[CompanyProfile]:
        """Fetch company profile with retry logic"""
        logger.info(f"Fetching profile for {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Validate required fields
            required_fields = ['shortName', 'sector', 'industry']
            if not all(info.get(field) for field in required_fields):
                logger.error(f"Missing required fields for {ticker}")
                return None
            
            profile = CompanyProfile(
                name=info.get('shortName'),
                ticker=ticker,
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                country=info.get('country')
            )
            
            logger.info(f"Profile fetched: {profile.name} ({profile.sector})")
            return profile
            
        except Exception as e:
            logger.error(f"Error fetching profile: {e}")
            return None

# --- MODERN OLLAMA LLM WRAPPER ---
class OllamaLLM(LLM):
    """LangChain-compatible Ollama wrapper that properly inherits from LLM base class"""
    
    model_name: str=""
    base_url: str=""
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 300  # Increased to 5 minutes
    max_tokens: int = 2048  # Limit output length
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type"""
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        api_url = f"{self.base_url.rstrip('/')}/api/generate"
        
        try:
            response = requests.post(
                api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

# --- MODERN LANGCHAIN ANALYZER ---
class InvestmentAnalyzer:
    def __init__(self, model_name: str, base_url: str):
        self.llm = OllamaLLM(model_name=model_name, base_url=base_url)
        self.output_parser = StrOutputParser()
        self.model_name = model_name
    
    def get_format_instructions(self) -> str:
        """Get JSON format instructions with escaped curly braces for LangChain"""
        return """
    Return a JSON object with this exact structure:
    {{
  "company_performance": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "management_and_governance": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "industry_and_sector_health": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "competitive_landscape": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "regulatory_risk": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "macroeconomic_exposure": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "overall_sentiment": {{"score": 1-10, "justification": "...", "confidence": "low/medium/high"}},
  "risk_flags": ["risk1", "risk2", ...],
  "opportunities": ["opp1", "opp2", ...]
    }}
    """
    
    def create_prompt_template(self, examples: List[Tuple[str, str]]) -> PromptTemplate:
        """Create prompt template with few-shot examples"""
        
        # Base system instructions
        system_instructions = """You are an expert financial analyst. Analyze the provided news context and score each investment factor from 1 (Very Negative) to 10 (Very Positive).

CRITICAL INSTRUCTIONS:
1. Base analysis ONLY on provided news context
2. Provide specific evidence from the news for each score
3. Assign confidence levels (low/medium/high) to each factor
4. Identify 3-5 key risk flags and opportunities
5. Return ONLY valid JSON matching the specified format

{format_instructions}"""
        
        # Build the complete prompt template string
        template_str = system_instructions.format(format_instructions=self.get_format_instructions())
        
        # Add few-shot examples if available
        if examples:
            template_str += "\n\nHere are examples of good analyses:\n"
            for i, (ex_context, ex_analysis) in enumerate(examples, 1):
                
                # --- START FIX ---
                # Escape braces in the example data so PromptTemplate ignores them
                safe_context = ex_context.replace("{", "{{").replace("}", "}}")
                safe_analysis = ex_analysis.replace("{", "{{").replace("}", "}}")
                # --- END FIX ---

                template_str += f"\n--- EXAMPLE {i} ---\n"
                # Use the "safe" strings instead of the raw ones
                template_str += f"News Context:\n{safe_context[:500]}...\n\n"
                template_str += f"Analysis:\n{safe_analysis[:500]}...\n"
        
        # Add the current task
        template_str += "\n\n--- YOUR TASK ---\nNews Context:\n{news_context}\n\nAnalysis:"
        
        return PromptTemplate(
            template=template_str,
            input_variables=["news_context"]
        )
    
    def create_chain(self, examples: List[Tuple[str, str]]):
        """Create modern LCEL chain using pipe operator"""
        prompt = self.create_prompt_template(examples)
        
        # Modern LangChain Expression Language (LCEL) syntax
        chain = prompt | self.llm | self.output_parser
        
        return chain
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, context: str, examples: List[Tuple[str, str]]) -> Tuple[str, float]:
        """Run analysis with retry logic using modern LCEL"""
        logger.info("Starting LLM analysis with modern LCEL chain...")
        start_time = datetime.now()
        
        try:
            # Truncate context if too long
            if len(context) > MAX_CONTEXT_LENGTH:
                logger.warning(f"Context truncated from {len(context)} to {MAX_CONTEXT_LENGTH}")
                context = context[:MAX_CONTEXT_LENGTH]
            
            # Create and run the chain
            chain = self.create_chain(examples)
            result = chain.invoke({"news_context": context})
            
            # Clean the response
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            # Validate JSON structure
            parsed = json.loads(result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis completed in {processing_time:.2f}s")
            
            return result, processing_time
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise

# --- ANALYSIS VALIDATOR ---
class AnalysisValidator:
    @staticmethod
    def validate_analysis(analysis_json: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate analysis structure and content"""
        try:
            data = json.loads(analysis_json)
            
            required_factors = [
                'company_performance', 'management_and_governance',
                'industry_and_sector_health', 'competitive_landscape',
                'regulatory_risk', 'macroeconomic_exposure', 'overall_sentiment'
            ]
            
            # Check all required factors present
            for factor in required_factors:
                if factor not in data:
                    return False, None, f"Missing factor: {factor}"
                
                if 'score' not in data[factor] or 'justification' not in data[factor]:
                    return False, None, f"Invalid structure for {factor}"
                
                score = data[factor]['score']
                if not isinstance(score, int) or not 1 <= score <= 10:
                    return False, None, f"Invalid score for {factor}: {score}"
                
                justification = data[factor]['justification']
                if len(justification) < 20:
                    return False, None, f"Justification too short for {factor}"
            
            # Calculate metadata
            scores = [data[f]['score'] for f in required_factors]
            metadata = {
                'avg_score': sum(scores) / len(scores),
                'num_risk_flags': len(data.get('risk_flags', [])),
                'num_opportunities': len(data.get('opportunities', []))
            }
            
            return True, metadata, None
            
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {e}"
        except Exception as e:
            return False, None, f"Validation error: {e}"

# --- DISPLAY UTILITIES ---
class DisplayFormatter:
    @staticmethod
    def display_analysis(analysis_json: str, profile: CompanyProfile):
        """Format and display analysis"""
        try:
            data = json.loads(analysis_json)
            
            print("\n" + "="*70)
            print(f"  INVESTMENT ANALYSIS: {profile.name} ({profile.ticker})")
            print(f"  Sector: {profile.sector} | Industry: {profile.industry}")
            print("="*70 + "\n")
            
            factors = [
                'company_performance', 'management_and_governance',
                'industry_and_sector_health', 'competitive_landscape',
                'regulatory_risk', 'macroeconomic_exposure', 'overall_sentiment'
            ]
            
            for factor in factors:
                if factor in data:
                    info = data[factor]
                    score = info.get('score', 'N/A')
                    confidence = info.get('confidence', 'N/A')
                    justification = info.get('justification', 'No details provided.')
                    
                    print(f"### {factor.replace('_', ' ').upper()}")
                    print(f"  Score: {score}/10 | Confidence: {confidence}")
                    print(f"  {justification}\n")
            
            # Display risk flags
            if 'risk_flags' in data and data['risk_flags']:
                print("### KEY RISK FLAGS")
                for i, risk in enumerate(data['risk_flags'], 1):
                    print(f"  {i}. {risk}")
                print()
            
            # Display opportunities
            if 'opportunities' in data and data['opportunities']:
                print("### KEY OPPORTUNITIES")
                for i, opp in enumerate(data['opportunities'], 1):
                    print(f"  {i}. {opp}")
                print()
            
            print("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Display error: {e}")
            print("\n--- RAW ANALYSIS OUTPUT ---")
            print(analysis_json)

# --- MAIN ORCHESTRATOR ---
def main():
    print("\n" + "="*70)
    print("  ENHANCED RAG INVESTMENT ANALYSIS TOOL")
    print("  Powered by LangChain + Ollama + Yahoo Finance + NewsAPI")
    print("="*70 + "\n")
    
    # Validate environment
    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY not set")
        sys.exit(1)
    
    # Initialize components
    try:
        db_manager = DatabaseManager(DATABASE_FILE)
        analyzer = InvestmentAnalyzer(model_name=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        validator = AnalysisValidator()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., NVDA, AAPL): ").upper().strip()
    if not ticker:
        print("No ticker provided. Exiting.")
        sys.exit(1)
    
    # Check for recent analysis
    use_cache = input("Use cached analysis if available? (y/n, default: y): ").lower()
    if use_cache != 'n':
        recent = db_manager.get_recent_analysis(ticker, days=7)
        if recent:
            print("\n--- Using Recent Cached Analysis ---")
            profile = CompanyProfileFetcher.fetch_profile(ticker)
            if profile:
                DisplayFormatter.display_analysis(recent, profile)
                return
    
    # Fetch company profile
    profile = CompanyProfileFetcher.fetch_profile(ticker)
    if not profile:
        logger.error("Failed to fetch company profile")
        sys.exit(1)
    
    # Get countries for macro analysis
    countries_input = input(f"Enter major countries for {profile.name} (e.g., USA, China): ")
    countries = [c.strip() for c in countries_input.split(',')] if countries_input else []
    
    # Fetch news
    news_fetcher = NewsFetcher(NEWS_API_KEY, db_manager)
    news_sections = news_fetcher.fetch_comprehensive_news(profile, countries)
    
    # Build context
    context = ""
    for section_name, content in news_sections.items():
        if content:
            context += f"\n--- {section_name.upper()} NEWS ---\n{content}\n"
    
    if not context.strip():
        logger.error("No news articles found")
        sys.exit(1)
    
    # Get few-shot examples
    examples = db_manager.get_few_shot_examples(n=2, min_quality_score=6.5)
    
    # Run analysis
    try:
        analysis_json, processing_time = analyzer.analyze(context, examples)
        
        # Validate
        is_valid, metadata, error = validator.validate_analysis(analysis_json)
        if not is_valid:
            logger.error(f"Analysis validation failed: {error}")
            print(f"\nValidation Error: {error}")
            print("\n--- RAW OUTPUT ---")
            print(analysis_json)
            sys.exit(1)
        
        # Display results
        DisplayFormatter.display_analysis(analysis_json, profile)
        
        # Save to database
        metadata['processing_time'] = processing_time
        db_manager.save_analysis(ticker, analysis_json, context, OLLAMA_MODEL, metadata)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()