import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import sqlite3
import json

def init_database():
    """Initialize SQLite database with investor profiles table."""
    conn = sqlite3.connect('investor_profiles.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS investor_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            portfolio JSON NOT NULL,
            purchase_prices JSON NOT NULL,
            holding_periods JSON NOT NULL,
            num_holdings INTEGER,
            num_sectors INTEGER,
            avg_volatility REAL,
            avg_beta REAL,
            investment_style TEXT,
            risk_mgmt_score REAL,
            diversification_score REAL,
            performance_score REAL,
            discipline_score REAL,
            timing_score REAL,
            overall_score REAL,
            recommendations TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def get_historical_purchase_price(ticker: str, holding_period_days: int) -> float:
    """
    Calculate the purchase price based on historical data.
    
    Args:
        ticker: Stock ticker symbol
        holding_period_days: Number of days the stock has been held
        
    Returns:
        Historical price from holding_period_days ago, or None if unavailable
    """
    try:
        # Calculate the purchase date
        purchase_date = datetime.now() - timedelta(days=holding_period_days)
        
        # Fetch historical data with buffer (extra days to account for weekends/holidays)
        buffer_days = int(holding_period_days * 1.2) + 10
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{buffer_days}d")
        
        if hist.empty:
            print(f"  ⚠️  No historical data available for {ticker}")
            return None
        
        # Find the closest trading day to the purchase date
        hist_dates = hist.index
        closest_date = min(hist_dates, key=lambda d: abs((d.date() - purchase_date.date()).days))
        
        purchase_price = hist.loc[closest_date, 'Close']
        
        print(f"  ✅ {ticker}: Purchase price ${purchase_price:.2f} from {closest_date.date()} ({holding_period_days} days ago)")
        return float(purchase_price)
        
    except Exception as e:
        print(f"  ❌ Error calculating purchase price for {ticker}: {e}")
        return None

def save_investor_profile(user_id, name, portfolio, purchase_prices, holding_periods, analysis_results):
    """Save or update investor profile in database."""
    conn = sqlite3.connect('investor_profiles.db')
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute('SELECT user_id FROM investor_profiles WHERE user_id = ?', (user_id,))
    exists = cursor.fetchone() is not None
    
    # Prepare data
    data = {
        'user_id': user_id,
        'name': name,
        'last_updated': datetime.now().isoformat(),
        'portfolio': json.dumps(portfolio),
        'purchase_prices': json.dumps(purchase_prices),
        'holding_periods': json.dumps(holding_periods),
        'num_holdings': analysis_results['num_holdings'],
        'num_sectors': analysis_results['num_sectors'],
        'avg_volatility': analysis_results['avg_volatility'],
        'avg_beta': analysis_results['avg_beta'],
        'investment_style': ', '.join(analysis_results['styles'][:3]),
        'risk_mgmt_score': analysis_results['risk_mgmt_score'],
        'diversification_score': analysis_results['div_score'],
        'performance_score': analysis_results['perf_score'],
        'discipline_score': analysis_results['discipline_score'],
        'timing_score': analysis_results['timing_score'],
        'overall_score': analysis_results['overall_score'],
        'recommendations': json.dumps(analysis_results['recommendations'])
    }
    
    if exists:
        # Update existing record
        cursor.execute('''
            UPDATE investor_profiles
            SET name = ?, last_updated = ?, portfolio = ?, purchase_prices = ?, holding_periods = ?,
                num_holdings = ?, num_sectors = ?, avg_volatility = ?, avg_beta = ?,
                investment_style = ?, risk_mgmt_score = ?, diversification_score = ?,
                performance_score = ?, discipline_score = ?, timing_score = ?,
                overall_score = ?, recommendations = ?
            WHERE user_id = ?
        ''', (
            data['name'], data['last_updated'], data['portfolio'], data['purchase_prices'], data['holding_periods'],
            data['num_holdings'], data['num_sectors'], data['avg_volatility'], data['avg_beta'],
            data['investment_style'], data['risk_mgmt_score'], data['diversification_score'],
            data['performance_score'], data['discipline_score'], data['timing_score'],
            data['overall_score'], data['recommendations'], data['user_id']
        ))
        print(f"\n✅ Updated existing profile for {name} (ID: {user_id})")
    else:
        # Insert new record
        cursor.execute('''
            INSERT INTO investor_profiles (
                user_id, name, last_updated, portfolio, purchase_prices, holding_periods,
                num_holdings, num_sectors, avg_volatility, avg_beta,
                investment_style, risk_mgmt_score, diversification_score,
                performance_score, discipline_score, timing_score,
                overall_score, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['user_id'], data['name'], data['last_updated'], data['portfolio'],
            data['purchase_prices'], data['holding_periods'], data['num_holdings'], data['num_sectors'],
            data['avg_volatility'], data['avg_beta'], data['investment_style'],
            data['risk_mgmt_score'], data['diversification_score'], data['performance_score'],
            data['discipline_score'], data['timing_score'], data['overall_score'],
            data['recommendations']
        ))
        print(f"\n✅ Created new profile for {name} (ID: {user_id})")
    
    conn.commit()
    conn.close()

def load_investor_profile(user_id):
    """Load investor profile from database."""
    conn = sqlite3.connect('investor_profiles.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM investor_profiles WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        columns = [desc[0] for desc in cursor.description]
        profile = dict(zip(columns, row))
        profile['portfolio'] = json.loads(profile['portfolio'])
        profile['purchase_prices'] = json.loads(profile['purchase_prices'])
        profile['holding_periods'] = json.loads(profile['holding_periods'])
        profile['recommendations'] = json.loads(profile['recommendations'])
        return profile
    return None

def list_all_investors():
    """List all investors in database."""
    conn = sqlite3.connect('investor_profiles.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, name, last_updated, overall_score, investment_style
        FROM investor_profiles
        ORDER BY overall_score DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    if rows:
        print("\n" + "="*75)
        print("📋 ALL INVESTOR PROFILES")
        print("="*75)
        print(f"{'ID':<15} {'Name':<20} {'Score':<10} {'Last Updated':<20}")
        print("-"*75)
        for row in rows:
            user_id, name, last_updated, score, style = row
            updated = datetime.fromisoformat(last_updated).strftime('%Y-%m-%d %H:%M')
            print(f"{user_id:<15} {name:<20} {score:>6.1f}/100  {updated:<20}")
        print("="*75)
    else:
        print("\n❌ No investor profiles found in database")

def get_portfolio_input():
    """Get portfolio holdings from user."""
    print("\n" + "="*75)
    print("📋 PORTFOLIO INPUT")
    print("="*75)
    print("Enter your stock holdings (ticker and percentage of portfolio)")
    print("Example: AAPL 30 (means Apple is 30% of your portfolio)")
    print("Type 'done' when finished\n")
    
    portfolio = {}
    total = 0
    
    while True:
        entry = input("Enter ticker and percentage (or 'done'): ").strip()
        if entry.lower() == 'done':
            break
        
        try:
            parts = entry.split()
            if len(parts) != 2:
                print("❌ Format: TICKER PERCENTAGE (e.g., AAPL 25)")
                continue
            
            ticker = parts[0].upper()
            percentage = float(parts[1])
            
            if percentage <= 0 or percentage > 100:
                print("❌ Percentage must be between 0 and 100")
                continue
            
            portfolio[ticker] = percentage
            total += percentage
            print(f"✅ Added {ticker}: {percentage}%")
            
        except ValueError:
            print("❌ Invalid input. Please enter ticker and percentage.")
    
    if total > 100:
        print(f"\n⚠️  Warning: Total allocation is {total}%, normalizing to 100%")
        portfolio = {k: (v/total)*100 for k, v in portfolio.items()}
    
    return portfolio

def analyze_investment_style(portfolio_data, holding_periods):
    """Analyze investor's style based on portfolio characteristics."""
    styles = []
    score_details = {}
    
    # Calculate portfolio metrics
    volatilities = []
    betas = []
    sectors = set()
    market_caps = []
    
    for ticker, data in portfolio_data.items():
        if data['returns'] is not None:
            volatilities.append(data['volatility'])
            betas.append(data['beta'])
            if data['sector']:
                sectors.add(data['sector'])
            if data['market_cap']:
                market_caps.append(data['market_cap'])
    
    avg_volatility = np.mean(volatilities) if volatilities else 0
    avg_beta = np.mean(betas) if betas else 1
    
    # Style classification
    style_score = 0
    
    # 1. Risk Tolerance
    if avg_volatility < 20:
        styles.append("Conservative")
        style_score += 20
    elif avg_volatility < 35:
        styles.append("Moderate")
        style_score += 15
    else:
        styles.append("Aggressive")
        style_score += 10
    
    score_details['risk_tolerance'] = {
        'score': style_score,
        'category': styles[0],
        'volatility': avg_volatility
    }
    
    # 2. Market Correlation
    correlation_score = 0
    if avg_beta < 0.8:
        styles.append("Defensive Investor")
        correlation_score = 15
    elif avg_beta > 1.3:
        styles.append("High-Beta Chaser")
        correlation_score = 8
    else:
        styles.append("Market-Aligned")
        correlation_score = 12
    
    score_details['market_correlation'] = {
        'score': correlation_score,
        'beta': avg_beta
    }
    
    # 3. Diversification
    diversification_score = 0
    num_holdings = len(portfolio_data)
    num_sectors = len(sectors)
    
    if num_holdings >= 15:
        styles.append("Well-Diversified")
        diversification_score = 25
    elif num_holdings >= 8:
        styles.append("Moderately Diversified")
        diversification_score = 18
    elif num_holdings >= 5:
        styles.append("Concentrated Portfolio")
        diversification_score = 12
    else:
        styles.append("Highly Concentrated")
        diversification_score = 5
    
    score_details['diversification'] = {
        'score': diversification_score,
        'holdings': num_holdings,
        'sectors': num_sectors
    }
    
    # 4. Market Cap Preference
    cap_score = 0
    if market_caps:
        avg_cap = np.mean(market_caps)
        if avg_cap > 200e9:
            styles.append("Large-Cap Focused")
            cap_score = 20
        elif avg_cap > 10e9:
            styles.append("Mid-Cap Investor")
            cap_score = 15
        else:
            styles.append("Small-Cap Hunter")
            cap_score = 10
    
    score_details['market_cap_focus'] = {
        'score': cap_score,
        'avg_cap': np.mean(market_caps) if market_caps else 0
    }
    
    # 5. Holding Period (Time Horizon)
    time_score = 0
    if holding_periods:
        avg_holding = np.mean(list(holding_periods.values()))
        if avg_holding > 730:  # 2+ years
            styles.append("Long-Term Investor")
            time_score = 25
        elif avg_holding > 365:  # 1+ year
            styles.append("Medium-Term Holder")
            time_score = 20
        elif avg_holding > 90:  # 3+ months
            styles.append("Swing Trader")
            time_score = 12
        else:
            styles.append("Short-Term Trader")
            time_score = 5
    
    score_details['time_horizon'] = {
        'score': time_score,
        'avg_days': np.mean(list(holding_periods.values())) if holding_periods else 0
    }
    
    return styles, score_details, style_score + correlation_score + diversification_score + cap_score + time_score

def calculate_portfolio_metrics(portfolio, period="1y"):
    """Calculate comprehensive portfolio metrics."""
    portfolio_data = {}
    weights = []
    returns_list = []
    
    # Get S&P 500 for beta calculation
    try:
        market = yf.Ticker("^GSPC").history(period=period)
        market_returns = market['Close'].pct_change().dropna()
    except:
        market_returns = None
    
    print("\n📊 Analyzing portfolio holdings...")
    
    for ticker, weight in portfolio.items():
        try:
            print(f"  Fetching {ticker}...", end=" ")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            info = stock.info
            
            if data.empty or len(data) < 20:
                print("❌ Insufficient data")
                continue
            
            # Calculate metrics
            daily_returns = data['Close'].pct_change().dropna()
            
            # Volatility
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Beta
            beta = 1.0
            if market_returns is not None and len(daily_returns) == len(market_returns):
                covariance = np.cov(daily_returns.values, market_returns.values)[0][1]
                market_variance = np.var(market_returns.values)
                beta = covariance / market_variance if market_variance != 0 else 1.0
            
            # Returns
            current_price = data['Close'].iloc[-1]
            start_price = data['Close'].iloc[0]
            period_return = ((current_price - start_price) / start_price) * 100
            
            # Store data
            portfolio_data[ticker] = {
                'weight': weight,
                'returns': daily_returns,
                'period_return': period_return,
                'volatility': volatility,
                'beta': beta,
                'sector': info.get('sector'),
                'market_cap': info.get('marketCap'),
                'current_price': current_price
            }
            
            weights.append(weight / 100)
            returns_list.append(daily_returns)
            
            print("✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return portfolio_data, weights, returns_list

def calculate_purchase_prices(portfolio, holding_periods):
    """Calculate historical purchase prices for all portfolio holdings."""
    print("\n💰 Calculating historical purchase prices...")
    purchase_prices = {}
    
    for ticker in portfolio.keys():
        if ticker in holding_periods:
            holding_days = holding_periods[ticker]
            purchase_price = get_historical_purchase_price(ticker, holding_days)
            if purchase_price:
                purchase_prices[ticker] = purchase_price
            else:
                print(f"  ⚠️  Could not determine purchase price for {ticker}")
        else:
            print(f"  ⚠️  No holding period specified for {ticker}")
    
    return purchase_prices

def score_investor_behavior(portfolio_data, weights, returns_list, score_details):
    """Score investor based on behavioral finance principles."""
    behavior_scores = {}
    
    # 1. Risk Management (0-100)
    risk_score = 0
    reasons = []
    
    avg_vol = score_details['risk_tolerance']['volatility']
    if avg_vol < 15:
        risk_score = 90
        reasons.append("✓ Excellent risk management - low volatility portfolio")
    elif avg_vol < 25:
        risk_score = 75
        reasons.append("✓ Good risk management - moderate volatility")
    elif avg_vol < 40:
        risk_score = 55
        reasons.append("• Moderate risk - higher volatility accepted")
    else:
        risk_score = 30
        reasons.append("✗ High risk exposure - volatile portfolio")
    
    behavior_scores['risk_management'] = {'score': risk_score, 'reasons': reasons}
    
    # 2. Diversification Quality (0-100)
    div_score = 0
    div_reasons = []
    
    holdings = score_details['diversification']['holdings']
    sectors = score_details['diversification']['sectors']
    
    if holdings >= 12 and sectors >= 5:
        div_score = 95
        div_reasons.append(f"✓ Excellent diversification - {holdings} stocks across {sectors} sectors")
    elif holdings >= 8 and sectors >= 4:
        div_score = 80
        div_reasons.append(f"✓ Good diversification - {holdings} stocks, {sectors} sectors")
    elif holdings >= 5:
        div_score = 60
        div_reasons.append(f"• Moderate diversification - {holdings} stocks")
    else:
        div_score = 30
        div_reasons.append(f"✗ Poor diversification - only {holdings} stocks (high concentration risk)")
    
    # Check for over-concentration
    max_weight = max([data['weight'] for data in portfolio_data.values()])
    if max_weight > 40:
        div_score -= 20
        div_reasons.append(f"✗ Over-concentrated: {max_weight:.1f}% in single position")
    elif max_weight > 25:
        div_score -= 10
        div_reasons.append(f"⚠️  High concentration: {max_weight:.1f}% in top position")
    
    behavior_scores['diversification'] = {'score': max(div_score, 0), 'reasons': div_reasons}
    
    # 3. Performance Discipline (0-100)
    perf_score = 0
    perf_reasons = []
    
    if returns_list and len(returns_list) > 0:
        # Calculate portfolio returns
        portfolio_returns = pd.concat(returns_list, axis=1).mean(axis=1)
        sharpe = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() != 0 else 0
        
        if sharpe > 1.5:
            perf_score = 95
            perf_reasons.append(f"✓ Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 1:
            perf_score = 80
            perf_reasons.append(f"✓ Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 0.5:
            perf_score = 65
            perf_reasons.append(f"• Fair risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 0:
            perf_score = 45
            perf_reasons.append(f"• Positive but weak returns (Sharpe: {sharpe:.2f})")
        else:
            perf_score = 20
            perf_reasons.append(f"✗ Negative risk-adjusted returns (Sharpe: {sharpe:.2f})")
    
    behavior_scores['performance'] = {'score': perf_score, 'reasons': perf_reasons}
    
    # 4. Market Timing Ability (0-100)
    timing_score = 60  # Neutral score (market timing is difficult)
    timing_reasons = []
    
    beta = score_details['market_correlation']['beta']
    if 0.85 < beta < 1.15:
        timing_score = 70
        timing_reasons.append(f"✓ Market-aligned strategy (Beta: {beta:.2f})")
    elif beta < 0.85:
        timing_score = 75
        timing_reasons.append(f"✓ Defensive positioning (Beta: {beta:.2f})")
    elif beta > 1.3:
        timing_score = 50
        timing_reasons.append(f"⚠️  Aggressive market exposure (Beta: {beta:.2f})")
    
    behavior_scores['market_timing'] = {'score': timing_score, 'reasons': timing_reasons}
    
    # 5. Investment Discipline (0-100)
    discipline_score = 0
    disc_reasons = []
    
    time_score = score_details['time_horizon']['score']
    avg_days = score_details['time_horizon']['avg_days']
    
    if avg_days > 730:
        discipline_score = 95
        disc_reasons.append(f"✓ Excellent patience - avg holding {avg_days:.0f} days")
    elif avg_days > 365:
        discipline_score = 85
        disc_reasons.append(f"✓ Good investment horizon - avg {avg_days:.0f} days")
    elif avg_days > 180:
        discipline_score = 65
        disc_reasons.append(f"• Medium-term approach - avg {avg_days:.0f} days")
    elif avg_days > 90:
        discipline_score = 45
        disc_reasons.append(f"⚠️  Short-term focus - avg {avg_days:.0f} days")
    else:
        discipline_score = 25
        disc_reasons.append(f"✗ Very short-term trading - avg {avg_days:.0f} days")
    
    behavior_scores['discipline'] = {'score': discipline_score, 'reasons': disc_reasons}
    
    return behavior_scores

def analyze_investor_profile(portfolio, holding_periods, user_id, name):
    """Main analysis function for investor profile."""
    print("\n" + "="*75)
    print("🔍 PERSONAL INVESTMENT PROFILE ANALYSIS")
    print("="*75)
    
    # Calculate purchase prices first
    purchase_prices = calculate_purchase_prices(portfolio, holding_periods)
    
    # Calculate portfolio metrics
    portfolio_data, weights, returns_list = calculate_portfolio_metrics(portfolio)
    
    if not portfolio_data:
        print("\n❌ Could not analyze portfolio. Please check ticker symbols.")
        return
    
    # Analyze investment style
    styles, score_details, style_score = analyze_investment_style(portfolio_data, holding_periods)
    
    # Score investor behavior
    behavior_scores = score_investor_behavior(portfolio_data, weights, returns_list, score_details)
    
    # Calculate final scores
    risk_mgmt_score = behavior_scores['risk_management']['score']
    div_score = behavior_scores['diversification']['score']
    perf_score = behavior_scores['performance']['score']
    timing_score = behavior_scores['market_timing']['score']
    discipline_score = behavior_scores['discipline']['score']
    
    # Weighted overall score
    overall_score = (
        risk_mgmt_score * 0.25 +
        div_score * 0.25 +
        perf_score * 0.20 +
        timing_score * 0.10 +
        discipline_score * 0.20
    )
    
    # Collect recommendations
    recommendations = []
    if risk_mgmt_score < 60:
        recommendations.append("Reduce portfolio volatility by adding more stable, blue-chip stocks")
    if div_score < 60:
        recommendations.append("Increase diversification - aim for 8-12+ stocks across 4+ sectors")
    if perf_score < 60:
        recommendations.append("Review your stock selection criteria - focus on quality companies")
    if discipline_score < 60:
        recommendations.append("Extend your holding periods - avoid excessive trading")
    if score_details['market_correlation']['beta'] > 1.4:
        recommendations.append("Consider adding defensive positions to reduce market sensitivity")
    
    # Prepare analysis results for database
    analysis_results = {
        'num_holdings': len(portfolio_data),
        'num_sectors': score_details['diversification']['sectors'],
        'avg_volatility': score_details['risk_tolerance']['volatility'],
        'avg_beta': score_details['market_correlation']['beta'],
        'styles': styles,
        'risk_mgmt_score': risk_mgmt_score,
        'div_score': div_score,
        'perf_score': perf_score,
        'timing_score': timing_score,
        'discipline_score': discipline_score,
        'overall_score': overall_score,
        'recommendations': recommendations
    }
    
    # Save to database with purchase prices
    save_investor_profile(user_id, name, portfolio, purchase_prices, holding_periods, analysis_results)
    
    # Display results
    print("\n" + "="*75)
    print("👤 INVESTOR PROFILE")
    print("="*75)
    print(f"Name: {name}")
    print(f"ID: {user_id}")
    print(f"Investment Style: {', '.join(styles[:3])}")
    print(f"Number of Holdings: {len(portfolio_data)}")
    print(f"Sectors Represented: {score_details['diversification']['sectors']}")
    print(f"Average Portfolio Volatility: {score_details['risk_tolerance']['volatility']:.2f}%")
    print(f"Average Beta: {score_details['market_correlation']['beta']:.2f}")
    
    # Display purchase prices
    print("\n" + "="*75)
    print("💰 PURCHASE PRICES (Historical)")
    print("="*75)
    for ticker, price in purchase_prices.items():
        current_price = portfolio_data.get(ticker, {}).get('current_price', 0)
        if current_price > 0:
            return_pct = ((current_price - price) / price) * 100
            print(f"{ticker}: ${price:.2f} → ${current_price:.2f} ({return_pct:+.2f}%)")
        else:
            print(f"{ticker}: ${price:.2f}")
    
    print("\n" + "="*75)
    print(f"📊 RISK MANAGEMENT SCORE: {risk_mgmt_score:.0f}/100")
    print("="*75)
    for reason in behavior_scores['risk_management']['reasons']:
        print(f"  {reason}")
    
    print("\n" + "="*75)
    print(f"🎯 DIVERSIFICATION SCORE: {div_score:.0f}/100")
    print("="*75)
    for reason in behavior_scores['diversification']['reasons']:
        print(f"  {reason}")
    
    print("\n" + "="*75)
    print(f"📈 PERFORMANCE SCORE: {perf_score:.0f}/100")
    print("="*75)
    for reason in behavior_scores['performance']['reasons']:
        print(f"  {reason}")
    
    print("\n" + "="*75)
    print(f"⏱️  INVESTMENT DISCIPLINE SCORE: {discipline_score:.0f}/100")
    print("="*75)
    for reason in behavior_scores['discipline']['reasons']:
        print(f"  {reason}")
    
    print("\n" + "="*75)
    print(f"⭐ OVERALL INVESTOR SCORE: {overall_score:.1f}/100")
    print("="*75)
    print("Weighting: Risk Management 25%, Diversification 25%, Performance 20%,")
    print("           Discipline 20%, Market Timing 10%")
    
    # Recommendations
    print("\n" + "="*75)
    print("💡 PERSONALIZED RECOMMENDATIONS")
    print("="*75)
    
    if overall_score >= 80:
        print("✅ EXCELLENT INVESTOR")
        print("   You demonstrate strong investment principles with good risk management,")
        print("   diversification, and discipline. Keep up the systematic approach!")
    elif overall_score >= 65:
        print("👍 GOOD INVESTOR")
        print("   Solid investment approach with room for optimization in some areas.")
    elif overall_score >= 50:
        print("⚡ DEVELOPING INVESTOR")
        print("   You're building good habits but could improve in several key areas.")
    elif overall_score >= 35:
        print("⚠️  NEEDS IMPROVEMENT")
        print("   Significant gaps in your investment approach. Focus on fundamentals.")
    else:
        print("❌ HIGH RISK APPROACH")
        print("   Your current strategy carries substantial risk. Consider reassessing")
        print("   your approach and consulting with a financial advisor.")
    
    # Specific recommendations
    if recommendations:
        print("\n📋 Action Items:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print("\n" + "="*75)
    print("⚠️  DISCLAIMER: This analysis is for educational purposes only and does")
    print("   not constitute financial advice. Consult a qualified financial advisor.")
    print("="*75 + "\n")

if __name__ == "__main__":
    try:
        # Initialize database
        init_database()
        
        print("="*75)
        print("🎯 PERSONAL INVESTMENT PROFILE ANALYZER")
        print("="*75)
        print("This tool analyzes YOUR investment behavior and provides personalized")
        print("recommendations based on how YOU invest in the stock market.")
        print("\nOptions:")
        print("  1. Create/Update your profile")
        print("  2. View existing profile")
        print("  3. List all profiles")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "3":
            list_all_investors()
            sys.exit(0)
        elif choice == "4":
            print("\nGoodbye!")
            sys.exit(0)
        elif choice == "2":
            user_id = input("\nEnter your User ID: ").strip()
            profile = load_investor_profile(user_id)
            if profile:
                print("\n" + "="*75)
                print(f"👤 SAVED PROFILE: {profile['name']}")
                print("="*75)
                print(f"Last Updated: {datetime.fromisoformat(profile['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Investment Style: {profile['investment_style']}")
                print(f"Holdings: {profile['num_holdings']} stocks across {profile['num_sectors']} sectors")
                print(f"Average Volatility: {profile['avg_volatility']:.2f}%")
                print(f"Average Beta: {profile['avg_beta']:.2f}")
                print(f"\n⭐ Overall Score: {profile['overall_score']:.1f}/100")
                print(f"   • Risk Management: {profile['risk_mgmt_score']:.0f}/100")
                print(f"   • Diversification: {profile['diversification_score']:.0f}/100")
                print(f"   • Performance: {profile['performance_score']:.0f}/100")
                print(f"   • Discipline: {profile['discipline_score']:.0f}/100")
                print("\n📋 Recommendations:")
                for rec in profile['recommendations']:
                    print(f"   • {rec}")
                print("="*75)
            else:
                print(f"\n❌ No profile found for User ID: {user_id}")
            sys.exit(0)
        elif choice != "1":
            print("\n❌ Invalid choice. Exiting.")
            sys.exit(0)
        
        # Get user identification
        print("\n" + "="*75)
        print("👤 USER IDENTIFICATION")
        print("="*75)
        user_id = input("Enter your User ID (e.g., email or username): ").strip()
        name = input("Enter your name: ").strip()
        
        if not user_id or not name:
            print("\n❌ User ID and name are required. Exiting.")
            sys.exit(0)
        
        # Check if profile exists
        existing = load_investor_profile(user_id)
        if existing:
            print(f"\n✅ Found existing profile for {existing['name']}")
            print(f"   Last updated: {datetime.fromisoformat(existing['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Previous score: {existing['overall_score']:.1f}/100")
            update = input("\nUpdate this profile? (yes/no): ").strip().lower()
            if update != 'yes':
                print("\nAnalysis cancelled. Exiting.")
                sys.exit(0)
        
        portfolio = get_portfolio_input()
        
        if not portfolio:
            print("\n❌ No portfolio entered. Exiting.")
            sys.exit(0)
        
        print(f"\n✅ Portfolio entered: {len(portfolio)} holdings")
        
        # Get holding periods
        print("\n" + "="*75)
        print("📅 HOLDING PERIOD INPUT")
        print("="*75)
        print("How long have you held each position? (in days)")
        print("If unsure, estimate or enter 365 for all\n")
        
        holding_periods = {}
        for ticker in portfolio.keys():
            while True:
                try:
                    days = input(f"Days held {ticker} (or press Enter for 365): ").strip()
                    if days == "":
                        holding_periods[ticker] = 365
                        break
                    holding_periods[ticker] = int(days)
                    break
                except ValueError:
                    print("❌ Please enter a number")
        
        analyze_investor_profile(portfolio, holding_periods, user_id, name)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled. Exiting.")
        sys.exit(0)