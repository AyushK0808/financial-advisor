import requests
import yfinance as yf
import re
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

def clean_query(text: str) -> str:
    """Cleans the user query to improve search results."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    stopwords = {"what", "is", "stock", "price", "of", "tell", "me", "about", 
                 "share", "shares", "returns", "company"}
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words) if words else text

def find_ticker_from_text(text: str):
    """Finds the most likely stock ticker from a text query using Yahoo's search."""
    query = clean_query(text)
    if not query:
        print("❌ Please provide a company name to search for.")
        return None
        
    print(f"🔍 Searching for company name → '{query}'")
    
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        r = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or API error: {e}")
        return None
    except ValueError:
        print("❌ Could not parse API response.")
        return None
    
    if "quotes" not in r or len(r["quotes"]) == 0:
        print("❌ No results found.")
        return None
    
    # Find the first and most relevant "EQUITY" (stock)
    for item in r["quotes"]:
        if item.get("quoteType") == "EQUITY" and "symbol" in item:
            name = item.get('shortname', item.get('longname', 'Unknown'))
            print(f"✅ Found ticker: {item['symbol']} ({name})")
            return item["symbol"]
    
    print("❌ No stock ticker found in results.")
    return None

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index using EMA method."""
    if len(data) < period:
        return None
        
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty else 50

def calculate_macd(data):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """Calculate Sharpe Ratio (risk-adjusted return)."""
    excess_returns = returns - risk_free_rate / 252
    if len(returns) == 0 or excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.03):
    """Calculate Sortino Ratio (downside risk-adjusted return)."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_max_drawdown(data):
    """Calculate Maximum Drawdown."""
    cumulative = (1 + data.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100

def calculate_beta(stock_returns, market_returns):
    """Calculate Beta (systematic risk vs market)."""
    if len(stock_returns) == 0 or len(market_returns) == 0:
        return 1.0
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else 1.0

def analyze_fundamentals(ticker_obj):
    """Analyze fundamental metrics."""
    try:
        info = ticker_obj.info
        fundamentals = {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'roe': info.get('returnOnEquity'),
            'profit_margin': info.get('profitMargins'),
            'current_ratio': info.get('currentRatio'),
            'dividend_yield': info.get('dividendYield'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'short_name': info.get('shortName')
        }
        return fundamentals
    except:
        return {}

def score_fundamentals(fundamentals):
    """Score based on fundamental analysis with detailed criteria."""
    score = 0
    reasons = []
    
    # P/E Ratio (lower is generally better)
    pe = fundamentals.get('pe_ratio')
    if pe and pe > 0:
        if pe < 15:
            score += 20
            reasons.append(f"✓ Attractive P/E ratio: {pe:.2f} (undervalued)")
        elif pe < 25:
            score += 15
            reasons.append(f"• Fair P/E ratio: {pe:.2f} (reasonably priced)")
        elif pe < 40:
            score += 8
            reasons.append(f"• Moderate P/E ratio: {pe:.2f}")
        else:
            score += 3
            reasons.append(f"✗ High P/E ratio: {pe:.2f} (may be overvalued)")
    
    # PEG Ratio (< 1 is undervalued)
    peg = fundamentals.get('peg_ratio')
    if peg and peg > 0:
        if peg < 1:
            score += 20
            reasons.append(f"✓ Excellent PEG ratio: {peg:.2f} (growth undervalued)")
        elif peg < 2:
            score += 12
            reasons.append(f"• Fair PEG ratio: {peg:.2f} (fairly priced)")
        elif peg < 3:
            score += 6
            reasons.append(f"• Moderate PEG ratio: {peg:.2f}")
    
    # Price to Book (lower is better)
    pb = fundamentals.get('price_to_book')
    if pb and pb > 0:
        if pb < 1.5:
            score += 15
            reasons.append(f"✓ Low P/B ratio: {pb:.2f} (good value vs assets)")
        elif pb < 3:
            score += 10
            reasons.append(f"• Fair P/B ratio: {pb:.2f}")
        else:
            score += 5
    
    # Return on Equity (higher is better)
    roe = fundamentals.get('roe')
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:
            score += 20
            reasons.append(f"✓ Strong ROE: {roe_pct:.2f}% (efficient profit generation)")
        elif roe_pct > 15:
            score += 15
            reasons.append(f"✓ Good ROE: {roe_pct:.2f}%")
        elif roe_pct > 10:
            score += 10
            reasons.append(f"• Moderate ROE: {roe_pct:.2f}%")
        elif roe_pct > 0:
            score += 5
            reasons.append(f"• Positive ROE: {roe_pct:.2f}%")
        else:
            reasons.append(f"✗ Negative ROE: {roe_pct:.2f}% (losing shareholder value)")
    
    # Debt to Equity (lower is better)
    de = fundamentals.get('debt_to_equity')
    if de is not None and de >= 0:
        if de < 50:
            score += 15
            reasons.append(f"✓ Low debt: D/E = {de:.2f}% (low leverage/risk)")
        elif de < 100:
            score += 10
            reasons.append(f"• Moderate debt: D/E = {de:.2f}% (manageable)")
        elif de < 200:
            score += 5
            reasons.append(f"• Elevated debt: D/E = {de:.2f}%")
        else:
            score += 2
            reasons.append(f"✗ High debt: D/E = {de:.2f}% (high leverage/risk)")
    
    # Profit Margin
    pm = fundamentals.get('profit_margin')
    if pm:
        pm_pct = pm * 100
        if pm_pct > 20:
            score += 15
            reasons.append(f"✓ Excellent margins: {pm_pct:.2f}%")
        elif pm_pct > 10:
            score += 10
            reasons.append(f"✓ Good margins: {pm_pct:.2f}%")
        elif pm_pct > 5:
            score += 5
            reasons.append(f"• Fair margins: {pm_pct:.2f}%")
    
    # Current Ratio (liquidity)
    cr = fundamentals.get('current_ratio')
    if cr:
        if cr > 2:
            score += 10
            reasons.append(f"✓ Strong liquidity: CR = {cr:.2f}")
        elif cr > 1.5:
            score += 7
            reasons.append(f"• Good liquidity: CR = {cr:.2f}")
        elif cr > 1:
            score += 4
            reasons.append(f"• Adequate liquidity: CR = {cr:.2f}")
    
    # Dividend Yield
    dy = fundamentals.get('dividend_yield')
    if dy and dy > 0:
        dy_pct = dy * 100
        if dy_pct > 3:
            score += 10
            reasons.append(f"✓ Strong dividend yield: {dy_pct:.2f}%")
        elif dy_pct > 1:
            score += 6
            reasons.append(f"• Dividend yield: {dy_pct:.2f}%")
    else:
        reasons.append("• No dividend (common for growth stocks)")
    
    if not reasons:
        return -1, ["N/A (Fundamental data missing)"]
    
    return min(score, 100), reasons

def score_technical_analysis(data, rsi, macd, signal, ma50, ma200):
    """Score based on technical indicators with momentum analysis."""
    score = 0
    reasons = []
    current_price = data['Close'].iloc[-1]
    
    # RSI Analysis
    if rsi is not None:
        if rsi < 30:
            score += 25
            reasons.append(f"✓ RSI oversold: {rsi:.2f} (potential buy zone)")
        elif 30 <= rsi < 45:
            score += 20
            reasons.append(f"✓ RSI in buy zone: {rsi:.2f}")
        elif 45 <= rsi < 55:
            score += 15
            reasons.append(f"✓ RSI neutral: {rsi:.2f} (healthy)")
        elif 55 <= rsi < 70:
            score += 12
            reasons.append(f"• RSI slightly elevated: {rsi:.2f}")
        else:
            score += 5
            reasons.append(f"✗ RSI overbought: {rsi:.2f} (potential sell zone)")
    
    # MACD Analysis
    if macd > signal and macd > 0:
        score += 25
        reasons.append("✓ MACD bullish crossover (strong momentum)")
    elif macd > signal:
        score += 18
        reasons.append("✓ MACD positive momentum")
    elif macd < 0 and signal < 0:
        score += 5
        reasons.append("✗ MACD bearish trend")
    else:
        score += 10
        reasons.append("• MACD neutral/mixed")
    
    # Moving Average Analysis
    if current_price > ma50 > ma200:
        score += 30
        reasons.append("✓ Golden Cross - strong uptrend (Price > 50D MA > 200D MA)")
    elif current_price > ma50:
        score += 20
        reasons.append("✓ Price above 50-day MA (short-term uptrend)")
    elif ma50 < ma200 and current_price < ma200:
        score += 5
        reasons.append("✗ Death Cross - downtrend (50D MA < 200D MA)")
    
    # Price position relative to 200-day MA
    if current_price > ma200:
        score += 20
        reasons.append("✓ Above 200-day MA (long-term uptrend)")
    else:
        reasons.append("✗ Below 200-day MA (long-term downtrend)")
    
    return min(score, 100), reasons

def score_risk_metrics(sharpe, sortino, max_dd, volatility, beta, annual_return):
    """Score based on risk-adjusted metrics."""
    score = 0
    reasons = []
    
    # Sharpe Ratio (> 1 is good, > 2 is excellent)
    if sharpe > 2:
        score += 30
        reasons.append(f"✓ Excellent Sharpe ratio: {sharpe:.2f}")
    elif sharpe > 1:
        score += 25
        reasons.append(f"✓ Good Sharpe ratio: {sharpe:.2f}")
    elif sharpe > 0.5:
        score += 15
        reasons.append(f"• Moderate Sharpe ratio: {sharpe:.2f}")
    elif sharpe > 0:
        score += 8
        reasons.append(f"• Positive Sharpe ratio: {sharpe:.2f}")
    else:
        score += 2
        reasons.append(f"✗ Negative Sharpe ratio: {sharpe:.2f} (risk > reward)")
    
    # Sortino Ratio
    if sortino > 2:
        score += 25
        reasons.append(f"✓ Excellent Sortino ratio: {sortino:.2f}")
    elif sortino > 1:
        score += 20
        reasons.append(f"✓ Good downside protection: {sortino:.2f}")
    elif sortino > 0.5:
        score += 12
        reasons.append(f"• Moderate Sortino ratio: {sortino:.2f}")
    
    # Maximum Drawdown
    if max_dd > -10:
        score += 20
        reasons.append(f"✓ Low max drawdown: {max_dd:.2f}%")
    elif max_dd > -20:
        score += 15
        reasons.append(f"• Moderate max drawdown: {max_dd:.2f}%")
    elif max_dd > -30:
        score += 10
        reasons.append(f"• Significant drawdown: {max_dd:.2f}%")
    else:
        score += 5
        reasons.append(f"✗ High max drawdown: {max_dd:.2f}% (high risk)")
    
    # Volatility
    if volatility < 15:
        score += 15
        reasons.append(f"✓ Low volatility: {volatility:.2f}% (stable)")
    elif volatility < 25:
        score += 12
        reasons.append(f"• Moderate volatility: {volatility:.2f}%")
    elif volatility < 40:
        score += 8
        reasons.append(f"• Elevated volatility: {volatility:.2f}%")
    else:
        score += 3
        reasons.append(f"✗ High volatility: {volatility:.2f}% (risky)")
    
    # Beta (market correlation)
    if 0.8 < beta < 1.2:
        score += 10
        reasons.append(f"✓ Market-correlated beta: {beta:.2f}")
    elif beta < 0.8:
        score += 8
        reasons.append(f"• Low beta: {beta:.2f} (defensive)")
    elif beta < 1.5:
        score += 6
        reasons.append(f"• Moderate beta: {beta:.2f}")
    else:
        score += 3
        reasons.append(f"✗ High beta: {beta:.2f} (volatile)")
    
    return min(score, 100), reasons

def compute_comprehensive_stock_analysis(symbol: str):
    """Comprehensive stock analysis combining all metrics."""
    print(f"\n{'='*75}")
    print(f"📊 COMPREHENSIVE INVESTMENT ANALYSIS: {symbol}")
    print(f"{'='*75}\n")
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
        
        if data.empty or len(data) < 50:
            print("⚠️  Insufficient data for analysis")
            return
        
        # Get market data for Beta calculation
        try:
            market = yf.Ticker("^GSPC").history(period="1y")  # S&P 500
            market_returns = market['Close'].pct_change().dropna()
        except:
            market_returns = None
        
        # === CALCULATE ALL METRICS ===
        
        # Basic info
        fundamentals = analyze_fundamentals(ticker)
        company_name = fundamentals.get('short_name', symbol)
        sector = fundamentals.get('sector', 'N/A')
        industry = fundamentals.get('industry', 'N/A')
        
        # Returns and price
        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        yearly_return = ((current_price - start_price) / start_price) * 100
        
        # Daily returns and volatility
        daily_returns = data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Moving averages
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        ma200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else ma50
        
        # Technical indicators
        rsi = calculate_rsi(data['Close'])
        macd, signal = calculate_macd(data['Close'])
        
        # Risk metrics
        sharpe = calculate_sharpe_ratio(daily_returns)
        sortino = calculate_sortino_ratio(daily_returns)
        max_dd = calculate_max_drawdown(data['Close'])
        
        # Annual return
        annual_return = (1 + daily_returns.mean()) ** 252 - 1
        
        # Beta
        beta = 1.0
        if market_returns is not None and len(daily_returns) == len(market_returns):
            beta = calculate_beta(daily_returns.values, market_returns.values)
        
        # === SCORING ===
        
        fund_score, fund_reasons = score_fundamentals(fundamentals)
        tech_score, tech_reasons = score_technical_analysis(data, rsi, macd, signal, ma50, ma200)
        risk_score, risk_reasons = score_risk_metrics(sharpe, sortino, max_dd, volatility, beta, annual_return)
        
        # Weighted final score
        if fund_score == -1:
            # If fundamentals are missing, give more weight to technical and risk
            final_score = (tech_score * 0.55 + risk_score * 0.45)
            weights_used = "Technical: 55%, Risk: 45% (Fundamentals unavailable)"
        else:
            final_score = (fund_score * 0.40 + tech_score * 0.35 + risk_score * 0.25)
            weights_used = "Fundamentals: 40%, Technical: 35%, Risk: 25%"
        
        # === DISPLAY RESULTS ===
        
        print(f"🏢 COMPANY INFO")
        print(f"  Name: {company_name}")
        print(f"  Sector: {sector} | Industry: {industry}")
        if fundamentals.get('market_cap'):
            market_cap = fundamentals['market_cap']
            if market_cap > 1e12:
                print(f"  Market Cap: ${market_cap/1e12:.2f}T")
            elif market_cap > 1e9:
                print(f"  Market Cap: ${market_cap/1e9:.2f}B")
            else:
                print(f"  Market Cap: ${market_cap/1e6:.2f}M")
        print()
        
        print("📈 PRICE & RETURNS")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  1-Year Return: {yearly_return:+.2f}%")
        print(f"  Annualized Return: {annual_return*100:+.2f}%")
        print()
        
        print(f"📊 FUNDAMENTAL ANALYSIS (Score: {max(0, fund_score):.0f}/100)")
        if fund_score == -1:
            print("  ⚠️  Fundamental data unavailable")
        else:
            for reason in fund_reasons[:7]:
                print(f"  {reason}")
        print()
        
        print(f"📉 TECHNICAL ANALYSIS (Score: {tech_score:.0f}/100)")
        print(f"  RSI (14): {rsi:.2f}" if rsi else "  RSI: N/A")
        print(f"  MACD: {macd:.4f} | Signal: {signal:.4f}")
        print(f"  50-Day MA: ${ma50:.2f} | 200-Day MA: ${ma200:.2f}")
        for reason in tech_reasons[:6]:
            print(f"  {reason}")
        print()
        
        print(f"⚠️  RISK ANALYSIS (Score: {risk_score:.0f}/100)")
        print(f"  Volatility (Annual): {volatility:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Sortino Ratio: {sortino:.2f}")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Beta vs S&P 500: {beta:.2f}")
        for reason in risk_reasons[:6]:
            print(f"  {reason}")
        print()
        
        print(f"{'='*75}")
        print(f"⭐ FINAL INVESTMENT SCORE: {final_score:.1f}/100")
        print(f"   Weighting: {weights_used}")
        print(f"{'='*75}")
        
        # Investment recommendation
        print()
        if final_score >= 75:
            print("✅ RECOMMENDATION: STRONG BUY")
            print("   Excellent investment opportunity. Strong fundamentals, momentum, and")
            print("   risk-adjusted returns. Shows strength across multiple metrics.")
        elif final_score >= 60:
            print("👍 RECOMMENDATION: BUY")
            print("   Good investment candidate. Solid fundamentals or momentum with")
            print("   favorable risk characteristics.")
        elif final_score >= 45:
            print("⚡ RECOMMENDATION: HOLD/MODERATE")
            print("   Mixed signals. Consider your risk tolerance and investment timeline.")
            print("   May require further research or better entry timing.")
        elif final_score >= 30:
            print("⚠️  RECOMMENDATION: CAUTION")
            print("   Significant concerns identified. High risk or poor value metrics.")
            print("   Not recommended for conservative investors.")
        else:
            print("❌ RECOMMENDATION: AVOID")
            print("   Poor investment characteristics. Weak fundamentals, negative momentum,")
            print("   or unfavorable risk profile. Consider alternatives.")
        
        print(f"\n{'='*75}")
        print("⚠️  DISCLAIMER: This is not financial advice. Always do your own research")
        print("   and consult with a qualified financial advisor before investing.")
        print(f"{'='*75}\n")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        query = input("Enter company name or ticker: ")
        
        # Check if it's already a ticker (short uppercase string)
        if len(query) <= 5 and query.isupper():
            ticker = query
            print(f"✅ Using ticker: {ticker}")
        else:
            ticker = find_ticker_from_text(query)
        
        if ticker:
            compute_comprehensive_stock_analysis(ticker)
        else:
            print("\n❌ Could not find stock. Please try with the exact ticker symbol.")
            
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled. Exiting.")
        sys.exit(0)