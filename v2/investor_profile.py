import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

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

def analyze_investor_profile(portfolio, holding_periods):
    """Main analysis function for investor profile."""
    print("\n" + "="*75)
    print("🔍 PERSONAL INVESTMENT PROFILE ANALYSIS")
    print("="*75)
    
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
    
    # Display results
    print("\n" + "="*75)
    print("👤 INVESTOR PROFILE")
    print("="*75)
    print(f"Investment Style: {', '.join(styles[:3])}")
    print(f"Number of Holdings: {len(portfolio_data)}")
    print(f"Sectors Represented: {score_details['diversification']['sectors']}")
    print(f"Average Portfolio Volatility: {score_details['risk_tolerance']['volatility']:.2f}%")
    print(f"Average Beta: {score_details['market_correlation']['beta']:.2f}")
    
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