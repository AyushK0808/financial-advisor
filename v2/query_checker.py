import re
import spacy
import yfinance as yf
from thefuzz import process as fuzzy_process
import ollama  # Added for local Llama 3.2

try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Error: spaCy 'en_core_web_sm' model not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None

# --- Setup: Expanded Keyword Lists (Unchanged) ---
COMPARISON_KEYWORDS = {
    'vs', 'versus', 'or', 'compare', 'better', 'against', 'which is',
    'outperform', 'underperform'
}
FINANCIAL_KEYWORDS = {
    'stock', 'market', 'invest', 'shares', 'equity', 'portfolio', 'nasdaq', 'nyse', 
    'dow jones', 's&p 500', 'index fund', 'etf', 'ticker', 'economy', 'inflation', 
    'recession', 'gdp', 'interest rate', 'federal reserve', 'cpi', 'unemployment', 
    'monetary policy', 'fiscal policy', 'dividend', 'earnings', 'p/e ratio', 'eps', 
    'mutual fund', 'bond', 'asset', 'capital', 'yield', 'bull market', 'bear market', 
    'volatility', 'diversification', 'ipo', 'sec', '10-k', '10-q', 'balance sheet', 
    'income statement', 'cash flow', 'finance', 'financial', 'budget', 'debt', 
    'mortgage', 'loan', 'credit', 'insurance', 'retirement', '401k', 'ira',
}

# --- Setup: Company Database for Fuzzy Matching (Unchanged) ---
COMPANY_DATABASE = {
    "Google": "GOOGL", "Alphabet": "GOOGL", "Microsoft": "MSFT", "Apple": "AAPL",
    "Amazon": "AMZN", "Meta": "META", "Facebook": "META", "Tesla": "TSLA",
    "Nvidia": "NVDA", "Netflix": "NFLX", "JPMorgan Chase": "JPM", 
    "Johnson & Johnson": "JNJ", "Walmart": "WMT", "Procter & Gamble": "PG",
    "Home Depot": "HD", "Visa": "V", "Mastercard": "MA", "Exxon Mobil": "XOM",
    "Chevron": "CVX", "Coca-Cola": "KO", "PepsiCo": "PEP", "McDonald's": "MCD",
    "Disney": "DIS", "Intel": "INTC", "AMD": "AMD", "Salesforce": "CRM",
    "Adobe": "ADBE", "Oracle": "ORCL", "Cisco": "CSCO", "IBM": "IBM", "Nike": "NKE",
    "Starbucks": "SBUX", "Boeing": "BA", "Goldman Sachs": "GS", 
    "Morgan Stanley": "MS", "Ford": "F", "General Motors": "GM"
}


# --- Llama 3.2 Function (Using Ollama) ---
def call_llama_model(query):
    """
    Calls the Llama 3.2 model running locally via Ollama.
    """
    print("\n[--- CALLING OLLAMA (llama3.2) ---]")
    print(f"[--- Input Query: '{query}' ---]")
    
    try:
        # Assumes 'llama3.2' is the model name in Ollama
        # Use `ollama list` in your terminal to see available models
        response = ollama.chat(
            model='llama3.2',  # CHANGE THIS if your model name is different
            messages=[
                {'role': 'system', 'content': 'You are a financial advisor.'},
                {'role': 'user', 'content': query},
            ]
        )
        
        full_response = response['message']['content']
        print(f"Llama/finbert response: {full_response}")
        print("[--- END OF OLLAMA CALL ---]\n")
        return full_response

    except Exception as e:
        print(f"*** ERROR calling Ollama: {e} ***")
        print("Please ensure Ollama is running and the model 'llama3.2' is available.")
        print("[--- END OF OLLAMA CALL ---]\n")
        return None


# --- Placeholder for Stock Comparison Logic (Unchanged) ---
def get_stock_comparison(tickers):
    """
    MOCK function. Fetches real data from yfinance for comparison.
    """
    print(f"\n[--- SIMULATING STOCK COMPARISON ---]")
    print(f"[--- Tickers to compare: {tickers} ---]")
    
    if not tickers:
        print("No valid tickers to compare.")
        return

    try:
        data = {}
        for ticker in tickers:
            t = yf.Ticker(ticker)
            info = t.info
            data[ticker] = {
                "Name": info.get('longName', 'N/A'),
                "P/E Ratio": info.get('trailingPE', 'N/A'),
                "Market Cap": info.get('marketCap', 'N/A'),
            }
        
        print("Comparison Data Pulled:")
        for ticker, info in data.items():
            print(f"  {ticker} ({info['Name']}):")
            print(f"    Market Cap: {info['Market Cap']}")
            print(f"    P/E Ratio: {info['P/E Ratio']}")
            
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")

    print("[--- END OF STOCK COMPARISON ---]\n")
    return


# --- Main Processing Functions (DEBUGGED) ---

def clean_and_extract_companies(query):
    """
    Identifies and extracts company names/tickers using a multi-step process:
    1. Regex for Tickers (e.g., AAPL)
    2. NER (spaCy) to find ORGs and PROPNs (e.g., Google, Gogle, Intel)
    3. Fuzzy Matching to map these names to the database (e.g., Gogle -> Google -> GOOGL)
    4. yfinance validation
    """
    print("   Extracting entities...")
    
    # DEBUG FIX: Removed aggressive TextBlob typo correction.
    # It was causing 'Gogle' -> 'Sole'.
    
    found_tickers = set()

    # 1. Extract stock tickers (e.g., AAPL, MSFT)
    ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
    regex_tickers = ticker_pattern.findall(query)
    
    for ticker in regex_tickers:
        try:
            t = yf.Ticker(ticker)
            if t.info.get('marketCap', 0) > 0:
                print(f"   Validated Ticker (Regex): {ticker}")
                found_tickers.add(ticker)
            else:
                 print(f"   (Regex found '{ticker}', but yfinance validation failed.)")
        except Exception:
            pass # Ignore tickers that fail yfinance lookup

    # 2. Use spaCy NER to find organization/proper names
    if not nlp:
        print("   (Skipping NER, spaCy model not loaded)")
        return list(found_tickers)
        
    doc = nlp(query)
    
    # DEBUG FIX: Check all PROPer Nouns, not just ORGs.
    # This will find "Google", "Gogle", "Intel", etc.
    words_to_check = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            words_to_check.add(ent.text)
            
    for token in doc:
        if token.pos_ == "PROPN":
            words_to_check.add(token.text)
            
    print(f"   NER/PROPN words to check: {words_to_check}")

    # 3. Fuzzy Match all potential names
    for name in words_to_check:
        # DEBUG FIX: Check if fuzzy_process returns None (no match)
        result = fuzzy_process.extractOne(
            name, 
            COMPANY_DATABASE.keys(), 
            score_cutoff=85  # Needs to be a close match
        )
        
        if result:  # This check fixes the TypeError
            match, score = result
            print(f"   -> Fuzzy matched '{name}' to '{match}' (Score: {score})")
            ticker = COMPANY_DATABASE[match]
            found_tickers.add(ticker)
        else:
            print(f"   -> No strong match in database for '{name}'")

    return list(set(found_tickers)) # Remove duplicates

def process_query_robust(query):
    """
    Processes a user query and routes it using robust NLP tools.
    """
    print(f"Query: '{query}'")
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    # --- Scenario 1: Stock Comparison ---
    if query_words.intersection(COMPARISON_KEYWORDS) and len(query_words) > 2:
        print(f"Routing: Stock Comparison")
        
        companies = clean_and_extract_companies(query)
        
        if companies:
            print(f"Action: Identified tickers: {companies}")
            get_stock_comparison(companies)
        else:
            print("Action: This is a comparison query, but no companies were identified.")
        return

    # --- Scenario 2: General Financial Query ---
    if query_words.intersection(FINANCIAL_KEYWORDS):
        print(f"Routing: General Financial")
        print(f"Action: Calling Llama 3.2 / finbert...")
        call_llama_model(query)
        return

    # --- Scenario 3: Random Query ---
    print(f"Routing: Random")
    print(f"Action: I am a financial advisor and this is out of my scope")

if __name__ == "__main__":
    print("### Robust Script Examples (DEBUGGED) ###")

    print("\n--- Example 1: Stock Comparison (Tickers) ---")
    process_query_robust("What's better, AAPL vs MSFT?")

    print("\n--- Example 2: Stock Comparison (Names) ---")
    process_query_robust("Compare Google or Apple")

    print("\n--- Example 3: Stock Comparison (Fuzzy + Typo) ---")
    process_query_robust("Compare Gogle or Miccrosoft") # Note the typos

    print("\n--- Example 4: Stock Comparison (Mixed) ---")
    process_query_robust("Is NVDA a better buy than Intel?")

    print("\n--- Example 5: General Financial (Economy) ---")
    process_query_robust("What is the outlook for inflation in 2025?")

    print("\n--- Example 6: General Financial (Markets) ---")
    process_query_robust("How is the stock market doing?")

    print("\n--- Example 7: Random Query ---")
    process_query_robust("What's the weather like today?")
    # --- Example Usage for Robust Script (All Examples) ---
