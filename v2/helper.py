from stock_data import exec_stock_analysis
from stock_news import orchestrator
from investor_profile import load_investor_profile

import json
import requests
from datetime import datetime
from typing import Dict, Tuple, Optional, List

class PersonalizedStockRecommendation:
    """
    Analyzes stock suitability for an investor using Ollama LLM
    for intelligent reasoning instead of hardcoded rules.
    """
    
    def __init__(self, ollama_model: str = "llama3", ollama_base_url: str = "http://localhost:11434"):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.risk_tolerance_map = {
            'conservative': {'max_volatility': 20, 'min_sharpe': 0.8, 'max_beta': 0.9},
            'moderate': {'max_volatility': 30, 'min_sharpe': 0.5, 'max_beta': 1.2},
            'aggressive': {'max_volatility': 50, 'min_sharpe': 0.3, 'max_beta': 1.8}
        }
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for analysis."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=500
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def _infer_risk_tolerance(self, profile: Dict) -> str:
        """
        Infer risk tolerance from investor profile metrics.
        """
        avg_volatility = profile.get('avg_volatility', 25)
        avg_beta = profile.get('avg_beta', 1.0)
        risk_mgmt_score = profile.get('risk_mgmt_score', 50)
        
        # Conservative: low volatility, low beta, high risk management
        if avg_volatility < 20 and avg_beta < 1.0 and risk_mgmt_score > 60:
            return 'conservative'
        # Aggressive: high volatility, high beta, lower risk management focus
        elif avg_volatility > 35 or avg_beta > 1.3:
            return 'aggressive'
        # Default to moderate
        else:
            return 'moderate'
    
    def _infer_investment_horizon(self, profile: Dict) -> str:
        """
        Infer investment horizon from holding patterns and style.
        """
        investment_style = profile.get('investment_style', '')
        discipline_score = profile.get('discipline_score', 50)
        
        if 'long' in investment_style.lower() or 'value' in investment_style.lower():
            return 'long-term'
        elif 'momentum' in investment_style.lower() or 'growth' in investment_style.lower():
            return 'short-term'
        else:
            return 'medium-term'
    
    def _extract_preferences(self, profile: Dict, portfolio: Dict) -> Tuple[List[str], List[str]]:
        """
        Extract sector and industry preferences from portfolio.
        """
        # This would ideally query the actual holdings to get sectors/industries
        # For now, return general preferences based on investment style
        investment_style = profile.get('investment_style', '')
        
        # Default preferences based on style
        if 'growth' in investment_style.lower():
            sectors = ['Technology', 'Healthcare', 'Consumer Discretionary']
            industries = ['Software', 'Biotechnology', 'E-commerce']
        elif 'value' in investment_style.lower():
            sectors = ['Financial Services', 'Energy', 'Utilities']
            industries = ['Banks', 'Oil & Gas', 'Electric Utilities']
        else:
            sectors = ['Technology', 'Healthcare', 'Financial Services']
            industries = ['Software', 'Pharmaceuticals', 'Banks']
        
        return sectors, industries
    
    def analyze_stock_for_investor(
        self, 
        user_id: str, 
        ticker: str, 
        countries: list = None,
        use_cache: str = 'y'
    ) -> Dict:
        """
        Main function to analyze if investor should buy/sell/hold a stock.
        
        Args:
            user_id: Investor's user ID
            ticker: Stock ticker symbol
            countries: List of countries for macro analysis
            use_cache: Whether to use cached analysis ('y' or 'n')
            
        Returns:
            Dictionary with recommendation and detailed reasoning
        """
        # Load investor profile
        investor_profile = load_investor_profile(user_id)
        if not investor_profile:
            return {
                'status': 'error',
                'message': f'Investor profile not found for user_id: {user_id}'
            }
        
        # Parse portfolio data (it's stored as JSON string)
        try:
            portfolio = json.loads(investor_profile['portfolio']) if isinstance(investor_profile['portfolio'], str) else investor_profile['portfolio']
            purchase_prices = json.loads(investor_profile['purchase_prices']) if isinstance(investor_profile['purchase_prices'], str) else investor_profile['purchase_prices']
            holding_periods = json.loads(investor_profile['holding_periods']) if isinstance(investor_profile['holding_periods'], str) else investor_profile['holding_periods']
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to parse portfolio data: {e}'
            }
        
        print(f"Portfolio: {portfolio}")
        print(f"Purchase Prices: {purchase_prices}")
        
        # Infer investor preferences from profile
        risk_tolerance = self._infer_risk_tolerance(investor_profile)
        investment_horizon = self._infer_investment_horizon(investor_profile)
        sector_prefs, industry_prefs = self._extract_preferences(investor_profile, portfolio)
        
        # Get stock technical/fundamental analysis
        stock_analysis = exec_stock_analysis(ticker)
        if stock_analysis.get('status') == 'error':
            return {
                'status': 'error',
                'message': f"Stock analysis failed: {stock_analysis.get('message')}"
            }
        
        # Get news/macro analysis
        try:
            news_analysis = orchestrator(ticker, countries or [], use_cache)
        except Exception as e:
            print(f"News analysis failed: {e}")
            news_analysis = None
        
        # Check if investor currently holds this stock
        is_holding = ticker in portfolio
        current_position = {} 
        
        # If the investor holds the stock, populate the position details
        if is_holding:
            # Get the portfolio weight/percentage
            current_position['weight_pct'] = portfolio.get(ticker, 0)
            
            # Get purchase price from stored historical data
            purchase_price = purchase_prices.get(ticker)
            
            if purchase_price:
                current_position['purchase_price'] = purchase_price
            else:
                # Fallback: use current price with warning
                current_price = stock_analysis.get('price_and_return', {}).get('current_price', 0)
                current_position['purchase_price'] = current_price
                print(f"⚠️  Warning: No purchase price found for {ticker}, using current price")
            
            # Add holding period if available
            if ticker in holding_periods:
                current_position['holding_period'] = holding_periods[ticker]
            
            # Calculate shares based on assumed portfolio value
            # Assuming a $100k portfolio for share calculation
            assumed_portfolio_value = 100000
            position_value = assumed_portfolio_value * (current_position['weight_pct'] / 100)
            current_position['shares'] = int(position_value / purchase_price) if purchase_price > 0 else 0
        
        # Build enhanced profile for analysis
        enhanced_profile = {
            'user_id': investor_profile['user_id'],
            'name': investor_profile.get('name', 'Unknown'),
            'risk_tolerance': risk_tolerance,
            'investment_horizon': investment_horizon,
            'sector_preferences': sector_prefs,
            'industry_preferences': industry_prefs,
            'portfolio_metrics': {
                'num_holdings': investor_profile.get('num_holdings', 0),
                'num_sectors': investor_profile.get('num_sectors', 0),
                'avg_volatility': investor_profile.get('avg_volatility', 0),
                'avg_beta': investor_profile.get('avg_beta', 1.0),
                'investment_style': investor_profile.get('investment_style', 'N/A'),
            },
            'scores': {
                'risk_mgmt': investor_profile.get('risk_mgmt_score', 0),
                'diversification': investor_profile.get('diversification_score', 0),
                'performance': investor_profile.get('performance_score', 0),
                'discipline': investor_profile.get('discipline_score', 0),
                'timing': investor_profile.get('timing_score', 0),
                'overall': investor_profile.get('overall_score', 0)
            }
        }
        
        # Perform comprehensive analysis using Ollama
        if is_holding:
            recommendation = self._analyze_sell_or_hold_with_llm(
                enhanced_profile, 
                stock_analysis, 
                news_analysis,
                current_position,
                ticker
            )
        else:
            recommendation = self._analyze_buy_decision_with_llm(
                enhanced_profile, 
                stock_analysis, 
                news_analysis,
                ticker
            )
        
        return recommendation
    
    def _analyze_buy_decision_with_llm(
        self, 
        profile: Dict, 
        stock_analysis: Dict,
        news_analysis: Optional[Dict],
        ticker: str
    ) -> Dict:
        """Determine if investor should buy the stock using Ollama."""
        
        # Extract key metrics
        risk_tolerance = profile['risk_tolerance']
        investment_horizon = profile['investment_horizon']
        risk_limits = self.risk_tolerance_map[risk_tolerance]
        
        risk_metrics = stock_analysis.get('risk_metrics', {})
        price_info = stock_analysis.get('price_and_return', {})
        scores = stock_analysis.get('scores', {})
        info = stock_analysis.get('info', {})
        fundamentals = stock_analysis.get('fundamentals_data', {})
        technical = stock_analysis.get('technical_indicators', {})
        
        # Prepare context for LLM
        context = self._prepare_buy_context(
            profile, stock_analysis, news_analysis, 
            risk_limits, ticker
        )
        
        # Create prompt for Ollama
        prompt = f"""You are an expert investment advisor analyzing whether an investor should buy a stock.

INVESTOR PROFILE:
- Name: {profile['name']}
- Risk Tolerance: {risk_tolerance.upper()}
- Investment Horizon: {investment_horizon}
- Investment Style: {profile['portfolio_metrics']['investment_style']}
- Current Portfolio: {profile['portfolio_metrics']['num_holdings']} holdings across {profile['portfolio_metrics']['num_sectors']} sectors
- Portfolio Avg Volatility: {profile['portfolio_metrics']['avg_volatility']:.2f}%
- Portfolio Avg Beta: {profile['portfolio_metrics']['avg_beta']:.2f}
- Preferred Sectors: {', '.join(profile['sector_preferences'])}
- Risk Limits: Max Volatility {risk_limits['max_volatility']}%, Min Sharpe {risk_limits['min_sharpe']}, Max Beta {risk_limits['max_beta']}

INVESTOR PERFORMANCE SCORES:
- Risk Management: {profile['scores']['risk_mgmt']:.1f}/100
- Diversification: {profile['scores']['diversification']:.1f}/100
- Performance: {profile['scores']['performance']:.1f}/100
- Discipline: {profile['scores']['discipline']:.1f}/100
- Overall: {profile['scores']['overall']:.1f}/100

STOCK ANALYSIS FOR {ticker}:
{context}

Based on this comprehensive analysis, provide:

1. A recommendation score from 0-100 (where 0=strong avoid, 50=neutral, 100=strong buy)
2. Overall recommendation (STRONG BUY, BUY, CONSIDER, or DO NOT BUY)
3. Confidence level (High, Moderate, or Low)
4. 3-5 key reasons supporting purchase (be specific with numbers and context)
5. 3-5 key reasons against purchase (be specific with numbers and context)

Consider the investor's existing portfolio composition, risk profile, and how this stock would fit their strategy.

Format your response EXACTLY as JSON:
{{
    "score": <number 0-100>,
    "recommendation": "<STRONG BUY|BUY|CONSIDER|DO NOT BUY>",
    "confidence": "<High|Moderate|Low>",
    "action": "<brief action statement>",
    "reasons_for": ["reason 1", "reason 2", ...],
    "reasons_against": ["reason 1", "reason 2", ...]
}}

Respond ONLY with valid JSON, no other text."""

        # Get LLM analysis
        llm_response = self._call_ollama(prompt)
        
        # Parse LLM response
        try:
            analysis = self._parse_json_response(llm_response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {llm_response}")
            # Fallback to basic analysis
            return self._fallback_buy_analysis(profile, stock_analysis, ticker)
        
        # Calculate position size
        position_size = self._calculate_position_size(
            profile, risk_tolerance, analysis['score'], 
            price_info.get('current_price', 0)
        )
        
        return {
            'status': 'success',
            'ticker': ticker,
            'investor_profile': {
                'user_id': profile['user_id'],
                'name': profile['name'],
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'investment_style': profile['portfolio_metrics']['investment_style'],
                'overall_score': profile['scores']['overall']
            },
            'current_holding': None,
            'recommendation': analysis['recommendation'],
            'action': analysis['action'],
            'confidence': analysis['confidence'],
            'score': f"{analysis['score']}/100",
            'suggested_position_size': position_size,
            'reasons_for': analysis['reasons_for'],
            'reasons_against': analysis['reasons_against'],
            'stock_summary': {
                'current_price': price_info.get('current_price'),
                'volatility': risk_metrics.get('volatility_annualized_pct'),
                'beta': risk_metrics.get('beta'),
                'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
                'overall_score': scores.get('final_score'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            },
            'llm_analysis': True
        }
    
    def _analyze_sell_or_hold_with_llm(
        self, 
        profile: Dict, 
        stock_analysis: Dict,
        news_analysis: Optional[Dict],
        current_position: Dict,
        ticker: str
    ) -> Dict:
        """Determine if investor should sell or hold using Ollama."""
        
        # Extract metrics
        risk_tolerance = profile['risk_tolerance']
        price_info = stock_analysis.get('price_and_return', {})
        risk_metrics = stock_analysis.get('risk_metrics', {})
        scores = stock_analysis.get('scores', {})
        
        current_price = price_info.get('current_price', 0)
        purchase_price = current_position.get('purchase_price', 0)
        shares = current_position.get('shares', 0)
        
        # Calculate returns using actual purchase price
        position_return_pct = ((current_price - purchase_price) / purchase_price) * 100 if purchase_price > 0 else 0
        position_value = shares * current_price
        unrealized_gain = shares * (current_price - purchase_price)
        holding_period = current_position.get('holding_period', 0)
        
        # Prepare context
        context = self._prepare_hold_context(
            profile, stock_analysis, news_analysis, 
            current_position, position_return_pct, holding_period, ticker
        )
        
        # Create prompt
        prompt = f"""You are an expert investment advisor analyzing whether an investor should sell or hold an existing stock position.

INVESTOR PROFILE:
- Name: {profile['name']}
- Risk Tolerance: {risk_tolerance.upper()}
- Investment Horizon: {profile['investment_horizon']}
- Investment Style: {profile['portfolio_metrics']['investment_style']}
- Overall Score: {profile['scores']['overall']:.1f}/100

CURRENT POSITION IN {ticker}:
- Shares Held: {shares}
- Purchase Price: ${purchase_price:.2f}
- Current Price: ${current_price:.2f}
- Position Return: {position_return_pct:.2f}%
- Unrealized Gain/Loss: ${unrealized_gain:,.2f}
- Holding Period: {holding_period} days
- Position Value: ${position_value:,.2f}

CURRENT STOCK ANALYSIS:
{context}

Based on this analysis, provide:

1. A hold/sell score from 0-100 (where 0=strong sell, 50=neutral, 100=strong hold)
2. Overall recommendation (STRONG HOLD, HOLD, CONSIDER SELLING, or SELL)
3. Confidence level (High, Moderate, or Low)
4. 3-5 key reasons to continue holding (be specific with numbers)
5. 3-5 key reasons to sell (be specific with numbers)

Consider the investor's profit/loss position, holding period vs investment horizon, and how the stock's quality has evolved.

Format your response EXACTLY as JSON:
{{
    "score": <number 0-100>,
    "recommendation": "<STRONG HOLD|HOLD|CONSIDER SELLING|SELL>",
    "confidence": "<High|Moderate|Low>",
    "action": "<brief action statement>",
    "reasons_for_hold": ["reason 1", "reason 2", ...],
    "reasons_for_sell": ["reason 1", "reason 2", ...]
}}

Respond ONLY with valid JSON, no other text."""

        # Get LLM analysis
        llm_response = self._call_ollama(prompt)
        
        # Parse response
        try:
            analysis = self._parse_json_response(llm_response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {llm_response}")
            return self._fallback_hold_analysis(profile, stock_analysis, current_position, ticker)
        
        return {
            'status': 'success',
            'ticker': ticker,
            'investor_profile': {
                'user_id': profile['user_id'],
                'name': profile['name'],
                'risk_tolerance': risk_tolerance,
                'investment_horizon': profile['investment_horizon'],
                'investment_style': profile['portfolio_metrics']['investment_style']
            },
            'current_holding': {
                'shares': shares,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'position_value': position_value,
                'unrealized_gain_loss': unrealized_gain,
                'return_pct': position_return_pct,
                'holding_period_days': holding_period,
                'weight_pct': current_position.get('weight_pct', 0)
            },
            'recommendation': analysis['recommendation'],
            'action': analysis['action'],
            'confidence': analysis['confidence'],
            'score': f"{analysis['score']}/100",
            'reasons_for_hold': analysis.get('reasons_for_hold', []),
            'reasons_for_sell': analysis.get('reasons_for_sell', []),
            'stock_summary': {
                'current_price': current_price,
                'volatility': risk_metrics.get('volatility_annualized_pct'),
                'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
                'overall_score': scores.get('final_score'),
                'rsi': stock_analysis.get('technical_indicators', {}).get('rsi_14')
            },
            'llm_analysis': True
        }
    
    def _prepare_buy_context(self, profile, stock_analysis, news_analysis, risk_limits, ticker):
        """Prepare comprehensive context for buy decision."""
        
        risk_metrics = stock_analysis.get('risk_metrics', {})
        price_info = stock_analysis.get('price_and_return', {})
        scores = stock_analysis.get('scores', {})
        info = stock_analysis.get('info', {})
        fundamentals = stock_analysis.get('fundamentals_data', {})
        technical = stock_analysis.get('technical_indicators', {})
        
        context = f"""
Company: {info.get('company_name', ticker)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

PRICE & RETURNS:
- Current Price: ${price_info.get('current_price', 0):.2f}
- 1-Year Return: {price_info.get('1_year_return_pct', 0):.2f}%
- Annualized Return: {price_info.get('annualized_return_pct', 0):.2f}%

RISK METRICS:
- Volatility (Annualized): {risk_metrics.get('volatility_annualized_pct', 0):.2f}%
- Beta: {risk_metrics.get('beta', 1.0):.2f}
- Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}
- Max Drawdown: {risk_metrics.get('max_drawdown_pct', 0):.2f}%

TECHNICAL INDICATORS:
- RSI (14): {technical.get('rsi_14', 50):.2f}
- MACD: {technical.get('macd', 0):.2f}
- MACD Signal: {technical.get('macd_signal', 0):.2f}
- 50-Day MA: ${technical.get('ma_50', 0):.2f}
- 200-Day MA: ${technical.get('ma_200', 0):.2f}

FUNDAMENTAL METRICS:
- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
- P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}
- Market Cap: ${info.get('market_cap', 0):,.0f}

OVERALL SCORES:
- Fundamental Score: {scores.get('fundamental_score', -1):.1f}/100
- Technical Score: {scores.get('technical_score', 0):.1f}/100
- Risk Score: {scores.get('risk_score', 0):.1f}/100
- Final Score: {scores.get('final_score', 0):.1f}/100

STOCK RECOMMENDATION: {stock_analysis.get('recommendation', {}).get('rating', 'N/A')}
"""
        
        if news_analysis and isinstance(news_analysis, dict):
            # Handle SaveFormatter output structure
            if 'analysis' in news_analysis:
                analysis_content = news_analysis['analysis']
                sentiment = analysis_content.get('sentiment', 'neutral')
                context += f"\nNEWS SENTIMENT: {sentiment.upper()}"
                
                # Add key insights if available
                if 'key_insights' in analysis_content:
                    insights = analysis_content['key_insights']
                    context += f"\nKEY INSIGHTS: {insights}"
            else:
                sentiment = news_analysis.get('sentiment', 'neutral')
                context += f"\nNEWS SENTIMENT: {sentiment.upper()}"
            
        return context
    
    def _prepare_hold_context(self, profile, stock_analysis, news_analysis, 
                              current_position, position_return_pct, holding_period, ticker):
        """Prepare comprehensive context for hold/sell decision."""
        
        risk_metrics = stock_analysis.get('risk_metrics', {})
        price_info = stock_analysis.get('price_and_return', {})
        scores = stock_analysis.get('scores', {})
        technical = stock_analysis.get('technical_indicators', {})
        
        context = f"""
PERFORMANCE METRICS:
- Position Return: {position_return_pct:.2f}%
- Holding Period: {holding_period} days
- Investment Horizon: {profile['investment_horizon']}

CURRENT STOCK QUALITY:
- Overall Score: {scores.get('final_score', 0):.1f}/100
- Fundamental Score: {scores.get('fundamental_score', -1):.1f}/100
- Technical Score: {scores.get('technical_score', 0):.1f}/100
- Risk Score: {scores.get('risk_score', 0):.1f}/100

RISK PROFILE:
- Volatility: {risk_metrics.get('volatility_annualized_pct', 0):.2f}%
- Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {risk_metrics.get('max_drawdown_pct', 0):.2f}%
- Beta: {risk_metrics.get('beta', 1.0):.2f}

TECHNICAL SIGNALS:
- RSI: {technical.get('rsi_14', 50):.2f}
- Current Price: ${price_info.get('current_price', 0):.2f}
- 50-Day MA: ${technical.get('ma_50', 0):.2f}
- 200-Day MA: ${technical.get('ma_200', 0):.2f}
- Price vs MA50: {((price_info.get('current_price', 0) / max(technical.get('ma_50', 1), 1) - 1) * 100):.2f}%
- Price vs MA200: {((price_info.get('current_price', 0) / max(technical.get('ma_200', 1), 1) - 1) * 100):.2f}%

STOCK RECOMMENDATION: {stock_analysis.get('recommendation', {}).get('rating', 'N/A')}
"""
        
        if news_analysis and isinstance(news_analysis, dict):
            # Handle SaveFormatter output structure
            if 'analysis' in news_analysis:
                analysis_content = news_analysis['analysis']
                sentiment = analysis_content.get('sentiment', 'neutral')
                context += f"\nRECENT NEWS SENTIMENT: {sentiment.upper()}"
            else:
                sentiment = news_analysis.get('sentiment', 'neutral')
                context += f"\nRECENT NEWS SENTIMENT: {sentiment.upper()}"
            
        return context
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling various formats."""
        # Try to find JSON in the response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}')
        
        if start != -1 and end != -1:
            json_str = response[start:end+1]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")
    
    def _fallback_buy_analysis(self, profile, stock_analysis, ticker):
        """Fallback analysis if LLM fails."""
        scores = stock_analysis.get('scores', {})
        final_score = scores.get('final_score', 0)
        
        if final_score >= 70:
            recommendation = "BUY"
            score = 75
        elif final_score >= 50:
            recommendation = "CONSIDER"
            score = 55
        else:
            recommendation = "DO NOT BUY"
            score = 30
        
        return {
            'status': 'success',
            'ticker': ticker,
            'investor_profile': {
                'user_id': profile['user_id'],
                'name': profile.get('name', 'Unknown')
            },
            'recommendation': recommendation,
            'score': f"{score}/100",
            'confidence': "Low",
            'action': "Analysis based on fallback logic",
            'reasons_for': ["Based on overall stock score"],
            'reasons_against': ["LLM analysis unavailable"],
            'llm_analysis': False
        }
    
    def _fallback_hold_analysis(self, profile, stock_analysis, current_position, ticker):
        """Fallback hold/sell analysis if LLM fails."""
        return {
            'status': 'success',
            'ticker': ticker,
            'investor_profile': {
                'user_id': profile['user_id'],
                'name': profile.get('name', 'Unknown')
            },
            'recommendation': "HOLD",
            'score': "50/100",
            'confidence': "Low",
            'action': "Maintain position (fallback analysis)",
            'reasons_for_hold': ["Unable to perform detailed analysis"],
            'reasons_for_sell': ["Consider consulting financial advisor"],
            'llm_analysis': False
        }
    
    def _calculate_position_size(
        self, 
        profile: Dict, 
        risk_tolerance: str,
        recommendation_score: int,
        current_price: float
    ) -> Dict:
        """Calculate suggested position size based on risk tolerance."""
        
        if current_price <= 0:
            return {'shares': 0, 'investment_amount': 0, 'note': 'Invalid price'}
        
        # Base allocation percentages
        allocation_map = {
            'conservative': 0.03,
            'moderate': 0.05,
            'aggressive': 0.08
        }
        
        base_allocation = allocation_map.get(risk_tolerance, 0.05)
        
        # Adjust based on recommendation score
        if recommendation_score >= 70:
            multiplier = 1.5
        elif recommendation_score >= 55:
            multiplier = 1.0
        else:
            multiplier = 0.5
        
        adjusted_allocation = base_allocation * multiplier
        
        # Calculate suggested investment (assuming $100k portfolio)
        assumed_portfolio = 100000
        investment_amount = assumed_portfolio * adjusted_allocation
        suggested_shares = int(investment_amount / current_price)
        
        return {
            'suggested_shares': suggested_shares,
            'investment_amount': round(investment_amount, 2),
            'portfolio_allocation_pct': round(adjusted_allocation * 100, 2),
            'note': f'Based on {risk_tolerance} risk profile and recommendation score'
        }


def main():
    """Example usage"""
    
    # Initialize with your Ollama configuration
    recommender = PersonalizedStockRecommendation(
        ollama_model="llama3.2",  # or your preferred model
        ollama_base_url="http://localhost:11434"
    )
    
    # Example: Analyze if investor should buy AAPL
    user_id = "ayush0808"
    ticker = "AAPL"
    countries = ["United States", "China"]
    
    print("Analyzing stock recommendation using Ollama LLM...")
    print("This may take a moment...\n")
    
    result = recommender.analyze_stock_for_investor(
        user_id=user_id,
        ticker=ticker,
        countries=countries,
        use_cache='y'
    )
    
    # Pretty print the result
    print("\n" + "="*75)
    print("📊 PERSONALIZED STOCK RECOMMENDATION")
    print("="*75)
    
    if result.get('status') == 'error':
        print(f"\n❌ Error: {result.get('message')}")
        return
    
    # Investor Profile
    print(f"\n👤 INVESTOR: {result['investor_profile']['name']}")
    print(f"   Risk Tolerance: {result['investor_profile']['risk_tolerance'].upper()}")
    print(f"   Investment Horizon: {result['investor_profile']['investment_horizon']}")
    print(f"   Investment Style: {result['investor_profile']['investment_style']}")
    print(f"   Overall Score: {result['investor_profile']['overall_score']:.1f}/100")
    
    # Current Holding (if exists)
    if result.get('current_holding'):
        holding = result['current_holding']
        print(f"\n💼 CURRENT POSITION IN {ticker}:")
        print(f"   Shares: {holding['shares']}")
        print(f"   Purchase Price: ${holding['purchase_price']:.2f}")
        print(f"   Current Price: ${holding['current_price']:.2f}")
        print(f"   Position Value: ${holding['position_value']:,.2f}")
        print(f"   Return: {holding['return_pct']:+.2f}%")
        print(f"   Unrealized P/L: ${holding['unrealized_gain_loss']:+,.2f}")
        print(f"   Holding Period: {holding['holding_period_days']} days")
        print(f"   Portfolio Weight: {holding['weight_pct']:.2f}%")
    
    # Recommendation
    print(f"\n🎯 RECOMMENDATION: {result['recommendation']}")
    print(f"   Score: {result['score']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Action: {result['action']}")
    
    # Reasons
    if result.get('current_holding'):
        print("\n✅ REASONS TO HOLD:")
        for i, reason in enumerate(result.get('reasons_for_hold', []), 1):
            print(f"   {i}. {reason}")
        
        print("\n❌ REASONS TO SELL:")
        for i, reason in enumerate(result.get('reasons_for_sell', []), 1):
            print(f"   {i}. {reason}")
    else:
        print("\n✅ REASONS TO BUY:")
        for i, reason in enumerate(result.get('reasons_for', []), 1):
            print(f"   {i}. {reason}")
        
        print("\n❌ REASONS AGAINST:")
        for i, reason in enumerate(result.get('reasons_against', []), 1):
            print(f"   {i}. {reason}")
        
        # Position size suggestion
        if result.get('suggested_position_size'):
            pos = result['suggested_position_size']
            print(f"\n💰 SUGGESTED POSITION SIZE:")
            print(f"   Shares: {pos['suggested_shares']}")
            print(f"   Investment: ${pos['investment_amount']:,.2f}")
            print(f"   Portfolio Allocation: {pos['portfolio_allocation_pct']:.2f}%")
            print(f"   Note: {pos['note']}")
    
    # Stock Summary
    print(f"\n📈 STOCK SUMMARY ({ticker}):")
    summary = result['stock_summary']
    print(f"   Current Price: ${summary['current_price']:.2f}")
    print(f"   Volatility: {summary['volatility']:.2f}%")
    if summary.get('beta'):
        print(f"   Beta: {summary['beta']:.2f}")
    if summary.get('sharpe_ratio'):
        print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"   Overall Score: {summary['overall_score']:.1f}/100")
    if summary.get('sector'):
        print(f"   Sector: {summary['sector']}")
    if summary.get('rsi'):
        print(f"   RSI: {summary['rsi']:.2f}")
    
    print("\n" + "="*75)
    print(f"🤖 Analysis powered by: {'Ollama LLM' if result.get('llm_analysis') else 'Fallback Logic'}")
    print("="*75)


if __name__ == "__main__":
    main()