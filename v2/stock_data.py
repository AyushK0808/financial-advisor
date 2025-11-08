import requests
import yfinance as yf
import re
import numpy as np

def clean_query(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    stopwords = {"what","is","stock","price","of","tell","me","about","share","shares","returns"}
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words) if words else text


def find_ticker_from_text(text: str):
    query = clean_query(text)
    print(f"🔍 Searching for company name → '{query}'")

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    response = requests.get(url, headers=headers)
    try:
        r = response.json()
    except:
        print("❌ Request error.")
        return None

    if "quotes" not in r or len(r["quotes"]) == 0:
        print("❌ No results.")
        return None

    for item in r["quotes"]:
        if item.get("quoteType") == "EQUITY":
            print(f"✅ Found ticker: {item['symbol']} ({item.get('shortname', 'Unknown')})")
            return item["symbol"]

    print("❌ No stock ticker found.")
    return None


def compute_stock_score(symbol: str):
    data = yf.Ticker(symbol).history(period="1y")
    if data.empty:
        print("⚠ No data found for:", symbol)
        return

    # 1-Year Return
    yearly_return = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100

    # Volatility (Standard deviation of daily returns)
    daily_returns = data["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100  # annualized %

    # Moving averages
    ma50 = data["Close"].rolling(50).mean().iloc[-1]
    ma200 = data["Close"].rolling(200).mean().iloc[-1]
    
    trend_score = 1 if ma50 > ma200 else -1

    # RSI (14-day)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))

    # --- Score Calculation System ---
    score = 0

    # Return score
    if yearly_return > 20: score += 30
    elif yearly_return > 5: score += 20
    elif yearly_return > 0: score += 10

    # Volatility score (lower is better)
    if volatility < 20: score += 30
    elif volatility < 35: score += 20
    elif volatility < 50: score += 10

    # Trend score
    score += 20 if trend_score == 1 else 0

    # RSI score (balanced RSI ~ good entry zone)
    if 40 < rsi < 60: score += 20
    elif 30 < rsi < 70: score += 10

    # Normalize max score = 100
    score = min(score, 100)

    print(f"\n📊 **Stock Analysis for {symbol}**")
    print(f"• 1-Year Return: {yearly_return:.2f}%")
    print(f"• Volatility (Risk): {volatility:.2f}%")
    print(f"• Trend (50D vs 200D): {'UP' if trend_score==1 else 'DOWN'}")
    print(f"• RSI (14): {rsi:.2f}")
    print(f"\n⭐ Stock Score: {score}/100")


if __name__ == "__main__":
    query = input("Ask: ")
    ticker = find_ticker_from_text(query)
    if ticker:
        compute_stock_score(ticker)
