#!/usr/bin/env python3
"""
Integrated Investment Analysis System
Combines investor profiling, query processing, stock analysis, and personalized recommendations
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Import functions from existing modules
# Assuming all files are in the same directory
try:
    from investor_profile import (
        analyze_investor_profile,
        get_portfolio_input,
        load_investor_profile,
        list_all_investors,
        init_database as init_profile_db
    )
    from query_checker import (
        process_query_robust,
        clean_and_extract_companies,
        call_llama_model
    )
    from stock_data import (
        find_ticker_from_text,
        compute_comprehensive_stock_analysis
    )
    from stock_news import (
        CompanyProfileFetcher,
        NewsFetcher,
        InvestmentAnalyzer,
        DatabaseManager,
        DisplayFormatter
    )
    import ollama
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all required files are in the same directory:")
    print("  - investor_profile.py")
    print("  - query_checker.py")
    print("  - stock_data.py")
    print("  - stock_news.py")
    sys.exit(1)

# Constants
SEPARATOR = "=" * 80
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class IntegratedAnalysisSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.user_id = None
        self.user_name = None
        self.investor_profile = None
        self.analysis_cache = {}
        
        # Initialize databases
        print("🔧 Initializing system...")
        init_profile_db()
        if NEWS_API_KEY:
            self.db_manager = DatabaseManager("analysis_archive.db")
        else:
            print("⚠️  NEWS_API_KEY not set - news analysis will be limited")
            self.db_manager = None
        
        print("✅ System initialized\n")
    
    def display_menu(self):
        """Display main menu"""
        print(f"\n{SEPARATOR}")
        print("🎯 INTEGRATED INVESTMENT ANALYSIS SYSTEM")
        print(SEPARATOR)
        print("1. Login / Create Profile")
        print("2. View My Profile")
        print("3. Update My Portfolio")
        print("4. Ask Investment Question")
        print("5. Analyze Stocks")
        print("6. View All Profiles")
        print("7. Exit")
        print(SEPARATOR)
    
    def login_or_create_profile(self):
        """Handle user login or profile creation"""
        print(f"\n{SEPARATOR}")
        print("👤 USER LOGIN / REGISTRATION")
        print(SEPARATOR)
        
        user_id = input("Enter your User ID (email/username): ").strip()
        if not user_id:
            print("❌ User ID required")
            return False
        
        # Try to load existing profile
        profile = load_investor_profile(user_id)
        
        if profile:
            self.user_id = user_id
            self.user_name = profile['name']
            self.investor_profile = profile
            print(f"\n✅ Welcome back, {self.user_name}!")
            print(f"   Last updated: {datetime.fromisoformat(profile['last_updated']).strftime('%Y-%m-%d %H:%M')}")
            print(f"   Overall Score: {profile['overall_score']:.1f}/100")
            return True
        else:
            # Create new profile
            print(f"\n📝 No profile found for {user_id}. Let's create one!")
            name = input("Enter your name: ").strip()
            if not name:
                print("❌ Name required")
                return False
            
            self.user_id = user_id
            self.user_name = name
            
            # Get portfolio and analyze
            print("\n📊 Please enter your portfolio to create your investor profile:")
            portfolio = get_portfolio_input()
            
            if not portfolio:
                print("❌ Portfolio required to create profile")
                return False
            
            # Get holding periods
            print(f"\n{SEPARATOR}")
            print("📅 HOLDING PERIOD INPUT")
            print(SEPARATOR)
            holding_periods = {}
            for ticker in portfolio.keys():
                while True:
                    try:
                        days = input(f"Days held {ticker} (Enter for 365): ").strip()
                        holding_periods[ticker] = 365 if days == "" else int(days)
                        break
                    except ValueError:
                        print("❌ Please enter a number")
            
            # Analyze and create profile
            analyze_investor_profile(portfolio, holding_periods, user_id, name)
            
            # Load the newly created profile
            self.investor_profile = load_investor_profile(user_id)
            return True
    
    def view_profile(self):
        """Display current user profile"""
        if not self.investor_profile:
            print("\n❌ Please login first")
            return
        
        p = self.investor_profile
        print(f"\n{SEPARATOR}")
        print(f"👤 YOUR INVESTOR PROFILE")
        print(SEPARATOR)
        print(f"Name: {p['name']}")
        print(f"User ID: {p['user_id']}")
        print(f"Last Updated: {datetime.fromisoformat(p['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nInvestment Style: {p['investment_style']}")
        print(f"Holdings: {p['num_holdings']} stocks across {p['num_sectors']} sectors")
        print(f"Average Volatility: {p['avg_volatility']:.2f}%")
        print(f"Average Beta: {p['avg_beta']:.2f}")
        
        print(f"\n📊 SCORES:")
        print(f"   ⭐ Overall: {p['overall_score']:.1f}/100")
        print(f"   • Risk Management: {p['risk_mgmt_score']:.0f}/100")
        print(f"   • Diversification: {p['diversification_score']:.0f}/100")
        print(f"   • Performance: {p['performance_score']:.0f}/100")
        print(f"   • Discipline: {p['discipline_score']:.0f}/100")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in p['recommendations']:
            print(f"   • {rec}")
        print(SEPARATOR)
    
    def update_portfolio(self):
        """Update user portfolio"""
        if not self.user_id:
            print("\n❌ Please login first")
            return
        
        print(f"\n{SEPARATOR}")
        print("🔄 UPDATE PORTFOLIO")
        print(SEPARATOR)
        print("Enter your updated portfolio:")
        
        portfolio = get_portfolio_input()
        if not portfolio:
            print("❌ No portfolio entered")
            return
        
        # Get holding periods
        holding_periods = {}
        for ticker in portfolio.keys():
            while True:
                try:
                    days = input(f"Days held {ticker} (Enter for 365): ").strip()
                    holding_periods[ticker] = 365 if days == "" else int(days)
                    break
                except ValueError:
                    print("❌ Please enter a number")
        
        # Re-analyze
        analyze_investor_profile(portfolio, holding_periods, self.user_id, self.user_name)
        
        # Reload profile
        self.investor_profile = load_investor_profile(self.user_id)
        print("\n✅ Profile updated successfully!")
    
    def process_query(self):
        """Process investment query"""
        if not self.investor_profile:
            print("\n❌ Please login first to get personalized recommendations")
            return
        
        print(f"\n{SEPARATOR}")
        print("💬 ASK INVESTMENT QUESTION")
        print(SEPARATOR)
        print("Examples:")
        print("  - Compare AAPL vs MSFT")
        print("  - Is NVDA better than Intel?")
        print("  - What is the outlook for inflation?")
        print("  - Compare Google or Microsoft")
        print(SEPARATOR)
        
        query = input("\nYour question: ").strip()
        if not query:
            print("❌ No query entered")
            return
        
        print(f"\n🔍 Processing query: '{query}'")
        print(f"{SEPARATOR}")
        
        # Extract companies/tickers
        tickers = clean_and_extract_companies(query)
        
        if tickers and len(tickers) >= 2:
            print(f"\n✅ Detected comparison query with tickers: {tickers}")
            self.analyze_multiple_stocks(tickers, query)
        elif tickers and len(tickers) == 1:
            print(f"\n✅ Detected single stock query: {tickers[0]}")
            self.analyze_single_stock(tickers[0])
        else:
            # General financial query - use Llama
            print("\n📰 General financial query - consulting AI advisor...")
            response = call_llama_model(query)
            if response:
                print(f"\n{SEPARATOR}")
                print("🤖 AI ADVISOR RESPONSE")
                print(SEPARATOR)
                print(response)
                print(SEPARATOR)
    
    def analyze_single_stock(self, ticker: str):
        """Analyze a single stock"""
        print(f"\n{SEPARATOR}")
        print(f"📊 ANALYZING {ticker}")
        print(SEPARATOR)
        
        # Run comprehensive stock analysis
        compute_comprehensive_stock_analysis(ticker)
        
        # Run news analysis if available
        if NEWS_API_KEY and self.db_manager:
            self.run_news_analysis(ticker)
        
        # Generate personalized recommendation
        self.generate_personalized_recommendation(ticker)
    
    def analyze_multiple_stocks(self, tickers: List[str], original_query: str):
        """Analyze and compare multiple stocks"""
        if not self.investor_profile:
            print("\n❌ Please login first")
            return
        
        print(f"\n{SEPARATOR}")
        print(f"📊 COMPARATIVE ANALYSIS: {' vs '.join(tickers)}")
        print(SEPARATOR)
        
        # Analyze each stock
        stock_analyses = {}
        
        for ticker in tickers:
            print(f"\n{'─'*80}")
            print(f"Analyzing {ticker}...")
            print(f"{'─'*80}")
            
            try:
                # Store current stdout to capture analysis
                from io import StringIO
                import contextlib
                
                # Run stock data analysis
                f = StringIO()
                with contextlib.redirect_stdout(f):
                    compute_comprehensive_stock_analysis(ticker)
                stock_data_output = f.getvalue()
                
                # Run news analysis if available
                news_output = ""
                if NEWS_API_KEY and self.db_manager:
                    f = StringIO()
                    with contextlib.redirect_stdout(f):
                        self.run_news_analysis(ticker)
                    news_output = f.getvalue()
                
                stock_analyses[ticker] = {
                    'stock_data': stock_data_output,
                    'news_analysis': news_output
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {ticker}: {e}")
                stock_analyses[ticker] = {
                    'stock_data': f"Error: {e}",
                    'news_analysis': ""
                }
        
        # Generate comparative recommendation
        self.generate_comparative_recommendation(tickers, stock_analyses, original_query)
    
    def run_news_analysis(self, ticker: str):
        """Run news-based analysis for a stock"""
        try:
            print(f"\n📰 Fetching news analysis for {ticker}...")
            
            # Get company profile
            profile = CompanyProfileFetcher.fetch_profile(ticker)
            if not profile:
                print("❌ Could not fetch company profile")
                return
            
            # Check for cached analysis
            recent = self.db_manager.get_recent_analysis(ticker, days=7)
            if recent:
                print("✅ Using cached news analysis (< 7 days old)")
                DisplayFormatter.display_analysis(recent, profile)
                return
            
            # Fetch fresh news
            news_fetcher = NewsFetcher(NEWS_API_KEY, self.db_manager)
            news_sections = news_fetcher.fetch_comprehensive_news(profile, [])
            
            context = ""
            for section_name, content in news_sections.items():
                if content:
                    context += f"\n--- {section_name.upper()} NEWS ---\n{content}\n"
            
            if not context.strip():
                print("⚠️  No news articles found")
                return
            
            # Run analysis
            analyzer = InvestmentAnalyzer(OLLAMA_MODEL, OLLAMA_BASE_URL)
            examples = self.db_manager.get_few_shot_examples(n=2, min_quality_score=6.5)
            
            analysis_json, processing_time = analyzer.analyze(context, examples)
            
            # Display and save
            DisplayFormatter.display_analysis(analysis_json, profile)
            
            # Save to cache
            metadata = {
                'processing_time': processing_time,
                'avg_score': 0,
                'num_risk_flags': 0,
                'num_opportunities': 0
            }
            self.db_manager.save_analysis(ticker, analysis_json, context, OLLAMA_MODEL, metadata)
            
        except Exception as e:
            print(f"⚠️  News analysis error: {e}")
    
    def generate_personalized_recommendation(self, ticker: str):
        """Generate personalized recommendation based on investor profile"""
        if not self.investor_profile:
            return
        
        print(f"\n{SEPARATOR}")
        print(f"🎯 PERSONALIZED RECOMMENDATION FOR {ticker}")
        print(SEPARATOR)
        
        # Build context for Llama
        profile_summary = f"""
Investor Profile:
- Name: {self.investor_profile['name']}
- Investment Style: {self.investor_profile['investment_style']}
- Risk Management Score: {self.investor_profile['risk_mgmt_score']:.0f}/100
- Diversification Score: {self.investor_profile['diversification_score']:.0f}/100
- Performance Score: {self.investor_profile['performance_score']:.0f}/100
- Discipline Score: {self.investor_profile['discipline_score']:.0f}/100
- Overall Score: {self.investor_profile['overall_score']:.1f}/100
- Current Holdings: {self.investor_profile['num_holdings']} stocks
- Average Portfolio Volatility: {self.investor_profile['avg_volatility']:.2f}%
- Average Beta: {self.investor_profile['avg_beta']:.2f}

Current Recommendations:
{chr(10).join('- ' + rec for rec in self.investor_profile['recommendations'])}
"""
        
        prompt = f"""As a financial advisor, provide a personalized investment recommendation for {ticker} based on this investor's profile:

{profile_summary}

Consider:
1. Does this stock align with their risk tolerance?
2. Would it improve their diversification?
3. Is it appropriate for their investment style?
4. What allocation percentage would be suitable?
5. Any specific risks or opportunities for this investor?

Provide a concise, actionable recommendation in 3-4 paragraphs."""
        
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are a personalized financial advisor.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            recommendation = response['message']['content']
            print(recommendation)
            print(SEPARATOR)
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")
            print("Please ensure Ollama is running with llama3.2 model")
    
    def generate_comparative_recommendation(self, tickers: List[str], analyses: Dict, query: str):
        """Generate comparative recommendation for multiple stocks"""
        if not self.investor_profile:
            return
        
        print(f"\n{SEPARATOR}")
        print(f"🎯 PERSONALIZED COMPARATIVE RECOMMENDATION")
        print(SEPARATOR)
        
        profile_summary = f"""
Investor Profile:
- Investment Style: {self.investor_profile['investment_style']}
- Risk Management: {self.investor_profile['risk_mgmt_score']:.0f}/100
- Diversification: {self.investor_profile['diversification_score']:.0f}/100
- Overall Score: {self.investor_profile['overall_score']:.1f}/100
- Portfolio Volatility: {self.investor_profile['avg_volatility']:.2f}%
- Beta: {self.investor_profile['avg_beta']:.2f}
"""
        
        # Summarize analyses
        analyses_summary = ""
        for ticker, analysis in analyses.items():
            analyses_summary += f"\n--- {ticker} Analysis Summary ---\n"
            # Extract key points from the outputs
            if 'FINAL INVESTMENT SCORE' in analysis['stock_data']:
                lines = analysis['stock_data'].split('\n')
                for line in lines:
                    if 'SCORE' in line or 'RECOMMENDATION' in line:
                        analyses_summary += line + '\n'
        
        prompt = f"""Original Question: {query}

Stocks Being Compared: {', '.join(tickers)}

{profile_summary}

Analysis Results:
{analyses_summary}

As a financial advisor, provide a personalized comparative recommendation:
1. Which stock(s) best fit this investor's profile?
2. Why is it the better choice given their risk tolerance and style?
3. What percentage allocation would you recommend?
4. Any concerns or caveats?

Be specific and actionable. Limit to 4-5 paragraphs."""
        
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are a personalized financial advisor specializing in comparative stock analysis.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            recommendation = response['message']['content']
            print(recommendation)
            print(SEPARATOR)
            print("\n⚠️  DISCLAIMER: This is not financial advice. Consult a qualified advisor.")
            print(SEPARATOR)
            
        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")
    
    def run(self):
        """Main application loop"""
        print("\n" + SEPARATOR)
        print("🚀 WELCOME TO THE INTEGRATED INVESTMENT ANALYSIS SYSTEM")
        print(SEPARATOR)
        print("\nThis system combines:")
        print("  ✓ Personal investor profiling")
        print("  ✓ Intelligent query processing")
        print("  ✓ Comprehensive stock analysis")
        print("  ✓ News sentiment analysis")
        print("  ✓ Personalized AI recommendations")
        
        while True:
            try:
                self.display_menu()
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == '1':
                    self.login_or_create_profile()
                elif choice == '2':
                    self.view_profile()
                elif choice == '3':
                    self.update_portfolio()
                elif choice == '4':
                    self.process_query()
                elif choice == '5':
                    # Direct stock analysis
                    ticker = input("\nEnter ticker symbol: ").upper().strip()
                    if ticker:
                        self.analyze_single_stock(ticker)
                elif choice == '6':
                    list_all_investors()
                elif choice == '7':
                    print("\n👋 Thank you for using the Investment Analysis System!")
                    print("   Remember: Always do your own research and consult professionals.")
                    sys.exit(0)
                else:
                    print("\n❌ Invalid choice. Please select 1-7.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                input("\nPress Enter to continue...")


def main():
    """Entry point"""
    # Check environment
    if not os.environ.get("OLLAMA_MODEL"):
        print("⚠️  OLLAMA_MODEL not set, using default: llama3.2")
    
    # Initialize and run system
    system = IntegratedAnalysisSystem()
    system.run()


if __name__ == "__main__":
    main()