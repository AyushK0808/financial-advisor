import streamlit as st
import pandas as pd
import json
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import your modules
from helper import PersonalizedStockRecommendation
from stock_data import exec_stock_analysis, find_ticker_from_text
from stock_news import orchestrator
from investor_profile import (
    init_database, 
    load_investor_profile, 
    list_all_investors,
    analyze_investor_profile,
    calculate_portfolio_metrics
)

# Page configuration
st.set_page_config(
    page_title="Investment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .strong-buy {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .buy {
        background-color: #d1ecf1;
        border-color: #17a2b8;
    }
    .hold {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .sell {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# Initialize database
init_database()

# Sidebar navigation
with st.sidebar:
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Stock Analysis", "📰 News Analysis", "👤 Investor Profile", "🎯 Personalized Recommendation", "📊 Portfolio Dashboard"],
        key="navigation"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Model settings
    ollama_model = st.selectbox(
        "Ollama Model",
        ["llama3.2", "llama3", "mistral", "mixtral"],
        index=0
    )
    
    ollama_url = st.text_input(
        "Ollama URL",
        value="http://localhost:11434"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **Investment Analysis System**
    
    Comprehensive stock analysis powered by:
    - Technical Analysis
    - Fundamental Analysis
    - News Sentiment
    - Risk Metrics
    - AI-Powered Recommendations
    """)

# Main content area
st.markdown('<div class="main-header">📊 Investment Analysis System</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("Welcome to Your Investment Analysis Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📈 Stock Analysis
        - Real-time stock data
        - Technical indicators (RSI, MACD, MA)
        - Fundamental metrics
        - Risk analysis
        """)
        if st.button("Go to Stock Analysis", key="nav_stock"):
            st.session_state.current_page = "📈 Stock Analysis"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 📰 News Analysis
        - Multi-source news aggregation
        - AI-powered sentiment analysis
        - Sector & industry trends
        - Macroeconomic insights
        """)
        if st.button("Go to News Analysis", key="nav_news"):
            st.session_state.current_page = "📰 News Analysis"
            st.rerun()
    
    with col3:
        st.markdown("""
        ### 🎯 Personalized Recommendations
        - Profile-based analysis
        - Risk-aligned suggestions
        - Portfolio optimization
        - Buy/Hold/Sell recommendations
        """)
        if st.button("Go to Recommendations", key="nav_reco"):
            st.session_state.current_page = "🎯 Personalized Recommendation"
            st.rerun()
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analyses Today", "0", "0%")
    with col2:
        st.metric("Active Profiles", "0", "0")
    with col3:
        st.metric("Avg Score", "0.0", "0.0")
    with col4:
        st.metric("Success Rate", "0%", "0%")

# ==================== STOCK ANALYSIS PAGE ====================
elif page == "📈 Stock Analysis":
    st.header("📈 Comprehensive Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Enter Stock Ticker or Company Name",
            placeholder="e.g., AAPL, Tesla, Microsoft"
        )
    
    with col2:
        search_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
    
    if search_button and ticker_input:
        with st.spinner("Analyzing stock... This may take a moment..."):
            # Find ticker if company name provided
            if len(ticker_input) > 5 or not ticker_input.isupper():
                ticker = find_ticker_from_text(ticker_input)
                if not ticker:
                    st.error("Could not find stock ticker. Please try exact ticker symbol.")
                    st.stop()
            else:
                ticker = ticker_input.upper()
            
            # Perform analysis
            result = exec_stock_analysis(ticker)
            
            if result.get('status') == 'error':
                st.error(f"Analysis failed: {result.get('message')}")
                st.stop()
            
            # Display results
            st.success(f"Analysis completed for **{result['info']['company_name']}** ({ticker})")
            
            # Company Info
            st.subheader("🏢 Company Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sector", result['info']['sector'])
            with col2:
                st.metric("Industry", result['info']['industry'])
            with col3:
                market_cap = result['info'].get('market_cap', 0)
                if market_cap > 1e12:
                    cap_display = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    cap_display = f"${market_cap/1e9:.2f}B"
                else:
                    cap_display = f"${market_cap/1e6:.2f}M"
                st.metric("Market Cap", cap_display)
            with col4:
                st.metric("Current Price", f"${result['price_and_return']['current_price']:.2f}")
            
            st.markdown("---")
            
            # Price & Returns
            st.subheader("💰 Price & Returns")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${result['price_and_return']['current_price']:.2f}"
                )
            with col2:
                yearly_return = result['price_and_return']['1_year_return_pct']
                st.metric(
                    "1-Year Return",
                    f"{yearly_return:.2f}%",
                    delta=f"{yearly_return:.2f}%"
                )
            with col3:
                annual_return = result['price_and_return']['annualized_return_pct']
                st.metric(
                    "Annualized Return",
                    f"{annual_return:.2f}%",
                    delta=f"{annual_return:.2f}%"
                )
            
            st.markdown("---")
            
            # Scores Display
            st.subheader("📊 Investment Scores")
            
            scores = result['scores']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fund_score = scores['fundamental_score'] if scores['fundamental_score'] != -1 else 0
                st.metric("Fundamental", f"{fund_score:.0f}/100")
            with col2:
                st.metric("Technical", f"{scores['technical_score']:.0f}/100")
            with col3:
                st.metric("Risk", f"{scores['risk_score']:.0f}/100")
            with col4:
                st.metric("Overall", f"{scores['final_score']:.1f}/100")
            
            # Score visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Fundamental', 'Technical', 'Risk', 'Overall'],
                    y=[fund_score, scores['technical_score'], scores['risk_score'], scores['final_score']],
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                )
            ])
            fig.update_layout(
                title="Score Breakdown",
                yaxis_range=[0, 100],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Technical Indicators
            st.subheader("📉 Technical Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                tech = result['technical_indicators']
                st.write("**Moving Averages**")
                st.write(f"- 50-Day MA: ${tech['ma_50']:.2f}")
                st.write(f"- 200-Day MA: ${tech['ma_200']:.2f}")
                
                st.write("\n**Momentum Indicators**")
                st.write(f"- RSI (14): {tech['rsi_14']:.2f}")
                st.write(f"- MACD: {tech['macd']:.4f}")
                st.write(f"- MACD Signal: {tech['macd_signal']:.4f}")
            
            with col2:
                risk = result['risk_metrics']
                st.write("**Risk Metrics**")
                st.write(f"- Volatility: {risk['volatility_annualized_pct']:.2f}%")
                st.write(f"- Beta: {risk['beta']:.2f}")
                st.write(f"- Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
                st.write(f"- Sortino Ratio: {risk['sortino_ratio']:.2f}")
                st.write(f"- Max Drawdown: {risk['max_drawdown_pct']:.2f}%")
            
            st.markdown("---")
            
            # Recommendation
            st.subheader("🎯 Investment Recommendation")
            
            rec = result['recommendation']
            rec_class = {
                'STRONG BUY': 'strong-buy',
                'BUY': 'buy',
                'HOLD/MODERATE': 'hold',
                'CAUTION': 'sell',
                'AVOID': 'sell'
            }.get(rec['rating'], 'hold')
            
            st.markdown(f"""
            <div class="recommendation-box {rec_class}">
                <h3>{rec['rating']}</h3>
                <p>{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed reasons
            with st.expander("📋 Detailed Analysis"):
                if scores['fundamental_score'] != -1:
                    st.write("**Fundamental Analysis:**")
                    for reason in scores['fundamental_reasons']:
                        st.write(f"- {reason}")
                
                st.write("\n**Technical Analysis:**")
                for reason in scores['technical_reasons']:
                    st.write(f"- {reason}")
                
                st.write("\n**Risk Analysis:**")
                for reason in scores['risk_reasons']:
                    st.write(f"- {reason}")

# ==================== NEWS ANALYSIS PAGE ====================
elif page == "📰 News Analysis":
    st.header("📰 News & Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Enter Stock Ticker",
            placeholder="e.g., NVDA, TSLA"
        )
    
    with col2:
        use_cache = st.checkbox("Use Cached Analysis (if available)", value=True)
    
    countries_input = st.text_input(
        "Enter Major Countries (comma-separated)",
        placeholder="e.g., United States, China, India"
    )
    
    analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    if analyze_button and ticker_input:
        with st.spinner("Fetching and analyzing news... This may take 1-2 minutes..."):
            ticker = ticker_input.upper()
            countries = [c.strip() for c in countries_input.split(',')] if countries_input else []
            cache_option = 'y' if use_cache else 'n'
            
            try:
                result = orchestrator(ticker, countries, cache_option)
                
                if result and 'profile' in result:
                    profile = result['profile']
                    analysis = result['analysis']
                    
                    st.success(f"Analysis completed for **{profile['name']}** ({profile['ticker']})")
                    
                    # Company Info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Company", profile['name'])
                    with col2:
                        st.metric("Sector", profile['sector'])
                    with col3:
                        st.metric("Industry", profile['industry'])
                    
                    st.markdown("---")
                    
                    # Factor Analysis
                    st.subheader("📊 Investment Factor Analysis")
                    
                    factors = {
                        'Company Performance': analysis.get('company_performance', {}),
                        'Management & Governance': analysis.get('management_and_governance', {}),
                        'Industry & Sector Health': analysis.get('industry_and_sector_health', {}),
                        'Competitive Landscape': analysis.get('competitive_landscape', {}),
                        'Regulatory Risk': analysis.get('regulatory_risk', {}),
                        'Macroeconomic Exposure': analysis.get('macroeconomic_exposure', {}),
                        'Overall Sentiment': analysis.get('overall_sentiment', {})
                    }
                    
                    # Create score visualization
                    factor_names = list(factors.keys())
                    scores = [factors[f].get('score', 0) for f in factor_names]
                    confidences = [factors[f].get('confidence', 'medium') for f in factor_names]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=factor_names,
                            y=scores,
                            text=scores,
                            textposition='auto',
                            marker_color=['#28a745' if s >= 7 else '#ffc107' if s >= 5 else '#dc3545' for s in scores]
                        )
                    ])
                    fig.update_layout(
                        title="Factor Scores (1-10 scale)",
                        yaxis_range=[0, 10],
                        height=400
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed factor analysis
                    st.subheader("📋 Detailed Factor Analysis")
                    
                    for factor_name, factor_data in factors.items():
                        with st.expander(f"**{factor_name}** - Score: {factor_data.get('score', 'N/A')}/10"):
                            st.write(f"**Confidence:** {factor_data.get('confidence', 'N/A')}")
                            st.write(f"**Analysis:** {factor_data.get('justification', 'No details available')}")
                    
                    st.markdown("---")
                    
                    # Risk Flags and Opportunities
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚠️ Key Risk Flags")
                        risks = analysis.get('risk_flags', [])
                        if risks:
                            for i, risk in enumerate(risks, 1):
                                st.warning(f"{i}. {risk}")
                        else:
                            st.info("No significant risk flags identified")
                    
                    with col2:
                        st.subheader("✅ Key Opportunities")
                        opps = analysis.get('opportunities', [])
                        if opps:
                            for i, opp in enumerate(opps, 1):
                                st.success(f"{i}. {opp}")
                        else:
                            st.info("No significant opportunities identified")
                
                else:
                    st.error("Analysis failed or returned invalid data")
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

# ==================== INVESTOR PROFILE PAGE ====================
elif page == "👤 Investor Profile":
    st.header("👤 Investor Profile Management")
    
    tab1, tab2, tab3 = st.tabs(["Create/Update Profile", "View Profile", "All Profiles"])
    
    with tab1:
        st.subheader("Create or Update Your Investor Profile")
        
        user_id = st.text_input("User ID", placeholder="e.g., email or username")
        name = st.text_input("Name", placeholder="Your full name")
        
        st.markdown("### Portfolio Holdings")
        st.write("Enter your stock holdings and their percentage allocation:")
        
        # Dynamic portfolio input
        if 'portfolio_items' not in st.session_state:
            st.session_state.portfolio_items = [{'ticker': '', 'allocation': 0, 'days_held': 365}]
        
        portfolio = {}
        holding_periods = {}
        
        for i, item in enumerate(st.session_state.portfolio_items):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                ticker = st.text_input(f"Ticker {i+1}", value=item['ticker'], key=f"ticker_{i}")
            with col2:
                allocation = st.number_input(f"Allocation % {i+1}", min_value=0.0, max_value=100.0, 
                                            value=float(item['allocation']), key=f"alloc_{i}")
            with col3:
                days = st.number_input(f"Days Held {i+1}", min_value=1, value=item['days_held'], key=f"days_{i}")
            with col4:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.portfolio_items.pop(i)
                    st.rerun()
            
            if ticker and allocation > 0:
                portfolio[ticker.upper()] = allocation
                holding_periods[ticker.upper()] = days
        
        if st.button("➕ Add Another Holding"):
            st.session_state.portfolio_items.append({'ticker': '', 'allocation': 0, 'days_held': 365})
            st.rerun()
        
        total_allocation = sum(portfolio.values())
        if total_allocation > 0:
            st.info(f"Total Allocation: {total_allocation:.1f}%")
            if total_allocation > 100:
                st.warning("Total allocation exceeds 100%. Will be normalized.")
        
        if st.button("📊 Analyze Profile", type="primary", use_container_width=True):
            if not user_id or not name or not portfolio:
                st.error("Please fill in all required fields and add at least one holding")
            else:
                with st.spinner("Analyzing your investment profile... This may take a moment..."):
                    try:
                        # Normalize portfolio if needed
                        if total_allocation > 100:
                            portfolio = {k: (v/total_allocation)*100 for k, v in portfolio.items()}
                        
                        analyze_investor_profile(portfolio, holding_periods, user_id, name)
                        st.success("Profile analysis completed and saved!")
                        
                        # Load and display the profile
                        profile = load_investor_profile(user_id)
                        if profile:
                            st.session_state.user_profile = profile
                            st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    
    with tab2:
        st.subheader("View Existing Profile")
        
        user_id_view = st.text_input("Enter User ID to view", key="view_user_id")
        
        if st.button("Load Profile"):
            profile = load_investor_profile(user_id_view)
            if profile:
                st.session_state.user_profile = profile
            else:
                st.error(f"No profile found for User ID: {user_id_view}")
        
        if st.session_state.user_profile:
            profile = st.session_state.user_profile
            
            st.success(f"Profile loaded: **{profile['name']}**")
            
            # Profile summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{profile['overall_score']:.1f}/100")
            with col2:
                st.metric("Holdings", profile['num_holdings'])
            with col3:
                st.metric("Sectors", profile['num_sectors'])
            with col4:
                st.metric("Avg Volatility", f"{profile['avg_volatility']:.1f}%")
            
            st.markdown("---")
            
            # Investment style
            st.subheader("Investment Style")
            st.write(f"**Style:** {profile['investment_style']}")
            st.write(f"**Average Beta:** {profile['avg_beta']:.2f}")
            
            # Scores breakdown
            st.subheader("Performance Scores")
            
            scores_data = {
                'Category': ['Risk Management', 'Diversification', 'Performance', 'Discipline', 'Timing'],
                'Score': [
                    profile['risk_mgmt_score'],
                    profile['diversification_score'],
                    profile['performance_score'],
                    profile['discipline_score'],
                    profile['timing_score']
                ]
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=scores_data['Category'],
                    y=scores_data['Score'],
                    text=[f"{s:.0f}" for s in scores_data['Score']],
                    textposition='auto',
                    marker_color='#1f77b4'
                )
            ])
            fig.update_layout(
                title="Investor Scores Breakdown",
                yaxis_range=[0, 100],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Personalized Recommendations")
            recs = profile['recommendations']
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.info(f"{i}. {rec}")
            else:
                st.success("Great job! No major recommendations at this time.")
            
            # Portfolio details
            st.subheader("📊 Portfolio Holdings")
            portfolio_dict = profile['portfolio']
            holding_periods_dict = profile['holding_periods']
            
            portfolio_df = pd.DataFrame({
                'Ticker': list(portfolio_dict.keys()),
                'Allocation (%)': list(portfolio_dict.values()),
                'Days Held': [holding_periods_dict.get(t, 0) for t in portfolio_dict.keys()]
            })
            st.dataframe(portfolio_df, use_container_width=True)
    
    with tab3:
        st.subheader("All Investor Profiles")
        
        if st.button("🔄 Refresh List"):
            st.rerun()
        
        # This would need to be implemented in investor_profile.py
        st.info("List of all investors in the database")
        # list_all_investors() function would be called here

# ==================== PERSONALIZED RECOMMENDATION PAGE ====================
elif page == "🎯 Personalized Recommendation":
    st.header("🎯 Personalized Stock Recommendation")
    
    st.write("Get AI-powered buy/sell/hold recommendations based on your investor profile")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        user_id = st.text_input("Your User ID", placeholder="e.g., your email or username")
    
    with col2:
        ticker = st.text_input("Stock Ticker", placeholder="e.g., AAPL, TSLA")
    
    countries_input = st.text_input(
        "Major Countries (optional, comma-separated)",
        placeholder="e.g., United States, China"
    )
    
    use_cache = st.checkbox("Use cached news analysis", value=True)
    
    if st.button("🎯 Get Recommendation", type="primary", use_container_width=True):
        if not user_id or not ticker:
            st.error("Please provide both User ID and Stock Ticker")
        else:
            with st.spinner("Analyzing... This may take 1-2 minutes..."):
                try:
                    countries = [c.strip() for c in countries_input.split(',')] if countries_input else []
                    cache_option = 'y' if use_cache else 'n'
                    
                    # Initialize recommender
                    recommender = PersonalizedStockRecommendation(
                        ollama_model=ollama_model,
                        ollama_base_url=ollama_url
                    )
                    
                    # Get recommendation
                    result = recommender.analyze_stock_for_investor(
                        user_id=user_id,
                        ticker=ticker.upper(),
                        countries=countries,
                        use_cache=cache_option
                    )
                    
                    if result.get('status') == 'error':
                        st.error(f"Error: {result.get('message')}")
                    else:
                        st.success("Recommendation generated successfully!")
                        
                        # Investor Profile Summary
                        st.subheader("👤 Your Profile")
                        profile = result['investor_profile']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Name", profile['name'])
                        with col2:
                            st.metric("Risk Tolerance", profile['risk_tolerance'].title())
                        with col3:
                            st.metric("Horizon", profile['investment_horizon'].replace('-', ' ').title())
                        with col4:
                            st.metric("Overall Score", f"{profile['overall_score']:.0f}/100")
                        
                        st.markdown("---")
                        
                        # Recommendation
                        st.subheader("🎯 Recommendation")
                        
                        rec = result['recommendation']
                        rec_class = {
                            'STRONG BUY': 'strong-buy',
                            'BUY': 'buy',
                            'CONSIDER': 'hold',
                            'HOLD': 'hold',
                            'STRONG HOLD': 'buy',
                            'CONSIDER SELLING': 'sell',
                            'SELL': 'sell',
                            'DO NOT BUY': 'sell'
                        }.get(rec, 'hold')
                        
                        st.markdown(f"""
                        <div class="recommendation-box {rec_class}">
                            <h2>{rec}</h2>
                            <p><strong>Action:</strong> {result['action']}</p>
                            <p><strong>Confidence:</strong> {result['confidence']}</p>
                            <p><strong>Score:</strong> {result['score']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Stock Summary
                        st.subheader("📊 Stock Summary")
                        stock_sum = result['stock_summary']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${stock_sum.get('current_price', 0):.2f}")
                        with col2:
                            st.metric("Volatility", f"{stock_sum.get('volatility', 0):.1f}%")
                        with col3:
                            st.metric("Beta", f"{stock_sum.get('beta', 0):.2f}")
                        with col4:
                            st.metric("Sharpe Ratio", f"{stock_sum.get('sharpe_ratio', 0):.2f}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sector", stock_sum.get('sector', 'N/A'))
                        with col2:
                            st.metric("Overall Score", f"{stock_sum.get('overall_score', 0):.1f}/100")
                        
                        st.markdown("---")
                        
                        # Detailed Reasons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'reasons_for' in result:
                                st.subheader("✅ Reasons For")
                                for reason in result['reasons_for']:
                                    st.success(f"• {reason}")
                            elif 'reasons_for_hold' in result:
                                st.subheader("✅ Reasons to Hold")
                                for reason in result['reasons_for_hold']:
                                    st.success(f"• {reason}")
                        
                        with col2:
                            if 'reasons_against' in result:
                                st.subheader("⚠️ Reasons Against")
                                for reason in result['reasons_against']:
                                    st.warning(f"• {reason}")
                            elif 'reasons_for_sell' in result:
                                st.subheader("⚠️ Reasons to Sell")
                                for reason in result['reasons_for_sell']:
                                    st.warning(f"• {reason}")
                        
                        # Current Holding Info (if applicable)
                        if result.get('current_holding'):
                            st.markdown("---")
                            st.subheader("💼 Your Current Position")
                            
                            holding = result['current_holding']
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Shares", f"{holding['shares']}")
                            with col2:
                                st.metric("Purchase Price", f"${holding['purchase_price']:.2f}")
                            with col3:
                                return_pct = holding['return_pct']
                                st.metric("Return", f"{return_pct:.2f}%", delta=f"{return_pct:.2f}%")
                            with col4:
                                gain_loss = holding['unrealized_gain_loss']
                                st.metric("Gain/Loss", f"${gain_loss:,.2f}", 
                                        delta=f"${gain_loss:,.2f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Position Value", f"${holding['position_value']:,.2f}")
                            with col2:
                                st.metric("Days Held", f"{holding['holding_period_days']}")
                        
                        # Position Size Suggestion (for new purchases)
                        if result.get('suggested_position_size'):
                            st.markdown("---")
                            st.subheader("💡 Suggested Position Size")
                            
                            pos_size = result['suggested_position_size']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Suggested Shares", pos_size['suggested_shares'])
                            with col2:
                                st.metric("Investment Amount", f"${pos_size['investment_amount']:,.2f}")
                            with col3:
                                st.metric("Portfolio %", f"{pos_size['portfolio_allocation_pct']:.2f}%")
                            
                            st.info(pos_size['note'])
                        
                        # LLM Analysis indicator
                        if result.get('llm_analysis'):
                            st.success("✨ Analysis powered by LLM reasoning")
                        else:
                            st.info("ℹ️ Analysis based on fallback logic (LLM unavailable)")
                
                except Exception as e:
                    st.error(f"Recommendation failed: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# ==================== PORTFOLIO DASHBOARD ====================
elif page == "📊 Portfolio Dashboard":
    st.header("📊 Portfolio Dashboard")
    
    user_id = st.text_input("Enter User ID", placeholder="your-user-id")
    
    if st.button("Load Dashboard"):
        profile = load_investor_profile(user_id)
        
        if not profile:
            st.error(f"No profile found for User ID: {user_id}")
        else:
            st.success(f"Dashboard loaded for **{profile['name']}**")
            
            # Overview metrics
            st.subheader("📈 Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                score = profile['overall_score']
                st.metric("Overall Score", f"{score:.1f}/100", 
                         delta=f"{'Good' if score >= 65 else 'Needs Work'}")
            with col2:
                st.metric("Holdings", profile['num_holdings'])
            with col3:
                st.metric("Sectors", profile['num_sectors'])
            with col4:
                st.metric("Avg Volatility", f"{profile['avg_volatility']:.1f}%")
            with col5:
                st.metric("Avg Beta", f"{profile['avg_beta']:.2f}")
            
            st.markdown("---")
            
            # Investment style and strategy
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Investment Profile")
                st.write(f"**Style:** {profile['investment_style']}")
                st.write(f"**Last Updated:** {profile['last_updated'][:10]}")
                
                # Risk profile gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=profile['risk_mgmt_score'],
                    title={'text': "Risk Management"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Score Breakdown")
                
                # Radar chart for scores
                categories = ['Risk Mgmt', 'Diversification', 'Performance', 
                             'Discipline', 'Timing']
                scores = [
                    profile['risk_mgmt_score'],
                    profile['diversification_score'],
                    profile['performance_score'],
                    profile['discipline_score'],
                    profile['timing_score']
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name='Your Scores'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Portfolio composition
            st.subheader("💼 Portfolio Composition")
            
            portfolio_dict = profile['portfolio']
            if portfolio_dict:
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=list(portfolio_dict.keys()),
                    values=list(portfolio_dict.values()),
                    hole=.3
                )])
                fig.update_layout(
                    title="Portfolio Allocation",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table view
                holding_periods = profile['holding_periods']
                df = pd.DataFrame({
                    'Ticker': list(portfolio_dict.keys()),
                    'Allocation (%)': [f"{v:.1f}" for v in portfolio_dict.values()],
                    'Days Held': [holding_periods.get(t, 0) for t in portfolio_dict.keys()]
                })
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No portfolio holdings found")
            
            st.markdown("---")
            
            # Recommendations
            st.subheader("💡 Action Items")
            recs = profile['recommendations']
            
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.warning(f"**{i}.** {rec}")
            else:
                st.success("✅ No urgent action items. Your portfolio looks good!")
            
            st.markdown("---")
            
            # Performance insights
            st.subheader("📈 Performance Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = profile['risk_mgmt_score']
                if score >= 80:
                    st.success(f"**Risk Management:** Excellent ({score:.0f}/100)")
                elif score >= 60:
                    st.info(f"**Risk Management:** Good ({score:.0f}/100)")
                else:
                    st.warning(f"**Risk Management:** Needs Improvement ({score:.0f}/100)")
            
            with col2:
                score = profile['diversification_score']
                if score >= 80:
                    st.success(f"**Diversification:** Excellent ({score:.0f}/100)")
                elif score >= 60:
                    st.info(f"**Diversification:** Good ({score:.0f}/100)")
                else:
                    st.warning(f"**Diversification:** Needs Improvement ({score:.0f}/100)")
            
            with col3:
                score = profile['discipline_score']
                if score >= 80:
                    st.success(f"**Discipline:** Excellent ({score:.0f}/100)")
                elif score >= 60:
                    st.info(f"**Discipline:** Good ({score:.0f}/100)")
                else:
                    st.warning(f"**Discipline:** Needs Improvement ({score:.0f}/100)")
            
            # Quick actions
            st.markdown("---")
            st.subheader("⚡ Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Update Profile", use_container_width=True):
                    st.session_state.current_page = "👤 Investor Profile"
                    st.rerun()
            
            with col2:
                if st.button("🎯 Get Recommendation", use_container_width=True):
                    st.session_state.current_page = "🎯 Personalized Recommendation"
                    st.rerun()
            
            with col3:
                if st.button("📈 Analyze Stock", use_container_width=True):
                    st.session_state.current_page = "📈 Stock Analysis"
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational and informational purposes only. 
    Not financial advice. Always consult with a qualified financial advisor before making investment decisions.</p>
    <p>Powered by OpenAI, Ollama, Yahoo Finance & NewsAPI</p>
</div>
""", unsafe_allow_html=True)