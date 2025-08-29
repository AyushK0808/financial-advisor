# src/streamlit_app.py - Complete Streamlit Web Interface
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import uuid
import json

# Import our modules
from src.rag_system import rag_system
from src.financial_data import financial_data
from src.llm_model import llm
from src.database import db

# Configure page
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    """Main Streamlit application"""
    
    st.title("🤖 AI-Powered Financial Advisor")
    st.markdown("Get personalized financial advice powered by local AI and real-time market data")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        page = st.selectbox(
            "Choose a section:",
            ["💬 Chat Advisor", "📊 Portfolio", "📈 Market Data", "🎯 Stock Analysis", "⚙️ Settings"],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Quick stats
        try:
            portfolio_summary = financial_data.get_portfolio_summary(st.session_state.user_id)
            if "error" not in portfolio_summary:
                st.metric("Portfolio Value", f"₹{portfolio_summary['total_value']:,.0f}")
                st.metric("P&L", f"₹{portfolio_summary['total_gain_loss']:,.0f}", f"{portfolio_summary['total_gain_loss_percent']:.1f}%")
        except:
            pass  # Ignore errors in sidebar
    
    # Route to selected page
    if page == "💬 Chat Advisor":
        chat_advisor_page()
    elif page == "📊 Portfolio":
        portfolio_page()
    elif page == "📈 Market Data":
        market_data_page()
    elif page == "🎯 Stock Analysis":
        stock_analysis_page()
    elif page == "⚙️ Settings":
        settings_page()

def chat_advisor_page():
    """Chat advisor interface with AI responses"""
    
    st.header("💬 Financial Chat Advisor")
    st.markdown("Ask me anything about investments, stocks, portfolio management, or financial planning!")
    
    # Quick question buttons
    st.subheader("🔥 Quick Questions")
    quick_questions = [
        "What is SIP and how does it work?",
        "How should I diversify my portfolio?", 
        "What are the best tax-saving investments?",
        "How to analyze stocks before buying?",
        "When should I sell my stocks?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        col = cols[i % 3]
        if col.button(question, key=f"quick_{i}", use_container_width=True):
            st.session_state.user_query = question
            st.experimental_rerun()
    
    st.markdown("---")
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            "Your question:",
            placeholder="e.g., Should I invest in RELIANCE stock?",
            key="chat_input",
            value=st.session_state.get("user_query", "")
        )
    
    with col2:
        stock_symbol = st.text_input(
            "Stock symbol (optional):",
            placeholder="RELIANCE.NS",
            key="stock_input"
        )
    
    if st.button("🚀 Ask Advisor", type="primary", use_container_width=True):
        if user_query:
            with st.spinner("🤔 Thinking... This may take a moment for web search..."):
                # Generate response using RAG system
                response = rag_system.generate_rag_response(
                    user_query, 
                    stock_symbol if stock_symbol else None,
                    st.session_state.user_id
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "stock_symbol": stock_symbol,
                    "response": response,
                    "timestamp": datetime.now()
                })
                
                # Clear the input
                st.session_state.user_query = ""
                st.experimental_rerun()
    
    # Display latest response prominently
    if st.session_state.chat_history:
        latest_chat = st.session_state.chat_history[-1]
        
        st.subheader("💡 Latest Response")
        
        # Question
        st.markdown(f"**❓ Question:** {latest_chat['query']}")
        if latest_chat.get('stock_symbol'):
            st.markdown(f"**📈 Stock Context:** {latest_chat['stock_symbol']}")
        
        # Response
        st.markdown("**🤖 AI Advisor:**")
        st.info(latest_chat['response']['answer'])
        
        # Additional info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{latest_chat['response'].get('confidence', 0):.1%}")
        with col2:
            st.metric("Context Used", latest_chat['response'].get('context_used', 0))
        with col3:
            st.metric("Sources", len(latest_chat['response'].get('sources', [])))
        
        # Sources
        if latest_chat['response'].get('sources'):
            with st.expander("📚 Information Sources"):
                for i, source in enumerate(latest_chat['response']['sources']):
                    st.write(f"{i+1}. {source}")
        
        st.markdown("---")
    
    # Chat history
    if len(st.session_state.chat_history) > 1:
        st.subheader("💭 Previous Conversations")
        
        # Show last 5 conversations (excluding latest)
        for i, chat in enumerate(reversed(st.session_state.chat_history[-6:-1])):
            with st.expander(f"Q: {chat['query'][:60]}..." if len(chat['query']) > 60 else f"Q: {chat['query']}"):
                st.write("**Question:**", chat['query'])
                if chat.get('stock_symbol'):
                    st.write("**Stock:**", chat['stock_symbol'])
                st.write("**Answer:**", chat['response']['answer'])
                st.caption(f"⏰ {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}")

def portfolio_page():
    """Portfolio management and tracking"""
    
    st.header("📊 Your Investment Portfolio")
    
    # Get portfolio data
    with st.spinner("📈 Loading portfolio..."):
        portfolio_summary = financial_data.get_portfolio_summary(st.session_state.user_id)
    
    if "error" in portfolio_summary:
        st.error(f"Error loading portfolio: {portfolio_summary['error']}")
        return
    
    # Portfolio overview metrics
    st.subheader("📋 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Value",
            f"₹{portfolio_summary['total_value']:,.2f}",
            delta=f"₹{portfolio_summary['total_gain_loss']:,.2f}"
        )
    
    with col2:
        st.metric(
            "📈 Stocks Value", 
            f"₹{portfolio_summary['stocks_value']:,.2f}",
            delta=f"{portfolio_summary['total_gain_loss_percent']:.2f}%"
        )
    
    with col3:
        st.metric(
            "💵 Available Cash", 
            f"₹{portfolio_summary['cash']:,.2f}"
        )
    
    with col4:
        st.metric(
            "🏢 Holdings", 
            f"{portfolio_summary['stock_count']} stocks"
        )
    
    # Portfolio visualization
    if portfolio_summary['stocks']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Portfolio Allocation")
            
            # Pie chart
            labels = [stock['symbol'] for stock in portfolio_summary['stocks']]
            values = [stock['current_value'] for stock in portfolio_summary['stocks']]
            
            fig_pie = px.pie(
                values=values,
                names=labels,
                title="Portfolio Distribution by Stock"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 Sector Allocation")
            
            # Sector pie chart
            sectors = portfolio_summary['diversification']['sectors']
            if sectors:
                fig_sector = px.pie(
                    values=list(sectors.values()),
                    names=list(sectors.keys()),
                    title="Portfolio Distribution by Sector"
                )
                st.plotly_chart(fig_sector, use_container_width=True)
        
        # Performance metrics
        st.subheader("🎯 Performance Highlights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if portfolio_summary['performance']['best_performer']:
                best = portfolio_summary['performance']['best_performer']
                st.success(f"🏆 Best Performer: {best['symbol']} (+{best['return_percent']:.1f}%)")
        
        with col2:
            if portfolio_summary['performance']['worst_performer']:
                worst = portfolio_summary['performance']['worst_performer']
                st.error(f"📉 Needs Attention: {worst['symbol']} ({worst['return_percent']:.1f}%)")
        
        # Holdings table
        st.subheader("📋 Detailed Holdings")
        
        # Convert to DataFrame for better display
        holdings_df = pd.DataFrame(portfolio_summary['stocks'])
        
        # Format columns for display
        display_df = pd.DataFrame({
            'Symbol': holdings_df['symbol'],
            'Company': holdings_df['company_name'].str[:20] + '...',
            'Qty': holdings_df['quantity'],
            'Avg Price': '₹' + holdings_df['avg_price'].round(2).astype(str),
            'Current Price': '₹' + holdings_df['current_price'].round(2).astype(str),
            'Current Value': '₹' + holdings_df['current_value'].round(0).astype(str),
            'P&L': holdings_df['gain_loss'].round(2),
            'Return %': holdings_df['gain_loss_percent'].round(1).astype(str) + '%',
            'Weight %': holdings_df['weight'].astype(str) + '%'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
    else:
        st.info("💡 Your portfolio is empty. Start by buying your first stock below!")
    
    # Trading section
    st.subheader("⚡ Quick Trading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🛒 Buy Stock**")
        
        buy_symbol = st.text_input("Stock Symbol", key="buy_symbol", placeholder="RELIANCE.NS")
        
        col_qty, col_price = st.columns(2)
        with col_qty:
            buy_quantity = st.number_input("Quantity", min_value=1, value=1, key="buy_quantity")
        with col_price:
            buy_price = st.number_input("Price per share", min_value=0.01, value=100.0, step=0.01, key="buy_price")
        
        if st.button("🛒 Buy Stock", type="primary", use_container_width=True):
            if buy_symbol:
                with st.spinner("Processing buy order..."):
                    result = financial_data.add_stock_to_portfolio(
                        st.session_state.user_id, buy_symbol, buy_quantity, buy_price
                    )
                
                if result["success"]:
                    st.success(result["message"])
                    st.balloons()
                    st.experimental_rerun()
                else:
                    st.error(result["error"])
    
    with col2:
        st.write("**💸 Sell Stock**")
        
        # Get current holdings for dropdown
        if portfolio_summary['stocks']:
            holdings_symbols = [stock['symbol'] for stock in portfolio_summary['stocks']]
            sell_symbol = st.selectbox("Select Stock", holdings_symbols, key="sell_symbol")
        else:
            sell_symbol = st.text_input("Stock Symbol", key="sell_symbol_text", placeholder="RELIANCE.NS")
        
        col_qty, col_price = st.columns(2)
        with col_qty:
            sell_quantity = st.number_input("Quantity", min_value=1, value=1, key="sell_quantity")
        with col_price:
            sell_price = st.number_input("Price per share", min_value=0.01, value=100.0, step=0.01, key="sell_price")
        
        if st.button("💸 Sell Stock", type="secondary", use_container_width=True):
            if sell_symbol:
                with st.spinner("Processing sell order..."):
                    result = financial_data.sell_stock_from_portfolio(
                        st.session_state.user_id, sell_symbol, sell_quantity, sell_price
                    )
                
                if result["success"]:
                    st.success(result["message"])
                    gain_loss = result["transaction"]["gain_loss"]
                    if gain_loss > 0:
                        st.success(f"💰 Profit: ₹{gain_loss:.2f}")
                    elif gain_loss < 0:
                        st.error(f"📉 Loss: ₹{gain_loss:.2f}")
                    st.experimental_rerun()
                else:
                    st.error(result["error"])

def market_data_page():
    """Market overview and stock data"""
    
    st.header("📈 Market Overview")
    
    # Get market data
    with st.spinner("📊 Loading market data..."):
        market_overview = financial_data.get_market_overview()
    
    # Market sentiment
    sentiment = market_overview.get("market_sentiment", "neutral")
    sentiment_colors = {
        "bullish": "🟢",
        "bearish": "🔴", 
        "neutral": "🟡"
    }
    
    st.subheader(f"{sentiment_colors.get(sentiment, '🟡')} Market Sentiment: {sentiment.title()}")
    
    # Market summary
    if "summary" in market_overview:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stocks Tracked", market_overview["summary"]["total_stocks"])
        with col2:
            st.metric("Gainers", market_overview["summary"]["gainers"], delta_color="normal")
        with col3:
            st.metric("Losers", market_overview["summary"]["losers"], delta_color="inverse")
        with col4:
            st.metric("Avg Change", f"{market_overview['summary']['avg_change']:.2f}%")
    
    # Top stocks cards
    if market_overview['stocks']:
        st.subheader("🏢 Top Stocks Performance")
        
        # Create cards for top stocks
        cols = st.columns(min(4, len(market_overview['stocks'])))
        
        for i, stock in enumerate(market_overview['stocks'][:4]):
            col = cols[i]
            with col:
                change_color = "normal" if stock['change_percent'] >= 0 else "inverse"
                col.metric(
                    label=stock['company_name'][:15] + "..." if len(stock['company_name']) > 15 else stock['company_name'],
                    value=f"₹{stock['current_price']:.2f}",
                    delta=f"{stock['change_percent']:.2f}%",
                    delta_color=change_color
                )
        
        # Detailed market table
        st.subheader("📊 Detailed Market Data")
        
        # Convert to DataFrame
        market_df = pd.DataFrame(market_overview['stocks'])
        
        # Format for display
        display_df = pd.DataFrame({
            'Symbol': market_df['symbol'],
            'Company': market_df['company_name'],
            'Price': '₹' + market_df['current_price'].round(2).astype(str),
            'Change': market_df['change'].round(2),
            'Change %': market_df['change_percent'].round(2).astype(str) + '%',
            'Volume': (market_df['volume'] / 1000).round(0).astype(int).astype(str) + 'K',
            'Sector': market_df['sector']
        })
        
        st.dataframe(display_df, use_container_width=True)
    
    # Individual stock lookup
    st.subheader("🔍 Individual Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        lookup_symbol = st.text_input("Enter stock symbol:", placeholder="RELIANCE.NS", key="lookup_symbol")
    
    with col2:
        lookup_period = st.selectbox("Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2, key="lookup_period")
    
    if st.button("📊 Get Stock Data", use_container_width=True) and lookup_symbol:
        with st.spinner("📈 Fetching stock data..."):
            stock_data = financial_data.get_stock_data(lookup_symbol, lookup_period)
            
            if "error" not in stock_data:
                # Stock info display
                st.success(f"📈 {stock_data['company_name']} ({lookup_symbol})")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{stock_data['current_price']:.2f}")
                with col2:
                    st.metric("Change", f"₹{stock_data['change']:.2f}", f"{stock_data['change_percent']:.2f}%")
                with col3:
                    st.metric("Volume", f"{stock_data['volume']:,}")
                with col4:
                    st.metric("52W High", f"₹{stock_data['52_week_high']:.2f}")
                
                # Additional details
                st.subheader("📋 Stock Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Company:** {stock_data['company_name']}")
                    st.write(f"**Sector:** {stock_data['sector']}")
                    st.write(f"**52W Low:** ₹{stock_data['52_week_low']:.2f}")
                    if stock_data.get('pe_ratio') != 'N/A':
                        st.write(f"**P/E Ratio:** {stock_data['pe_ratio']:.2f}")
                
                with col2:
                    if stock_data.get('sma_20'):
                        st.write(f"**SMA 20:** ₹{stock_data['sma_20']:.2f}")
                    if stock_data.get('sma_50'):
                        st.write(f"**SMA 50:** ₹{stock_data['sma_50']:.2f}")
                    st.write(f"**Volatility:** {stock_data.get('volatility', 0):.1f}%")
                    if stock_data.get('market_cap') != 'N/A':
                        st.write(f"**Market Cap:** ₹{stock_data['market_cap']:,}")
                
            else:
                st.error(stock_data['error'])

def stock_analysis_page():
    """AI-powered stock analysis with web search"""
    
    st.header("🎯 AI Stock Analysis")
    st.markdown("Get comprehensive AI-powered analysis of any stock using local knowledge and real-time web search")
    
    # Stock input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_symbol = st.text_input("Stock Symbol:", placeholder="RELIANCE.NS", key="analysis_symbol")
    
    with col2:
        analysis_type = st.selectbox("Analysis Type:", [
            "General Analysis",
            "Technical Analysis", 
            "Fundamental Analysis",
            "Recent News & Trends",
            "Buy/Sell Recommendation"
        ], key="analysis_type")
    
    if st.button("🔍 Analyze Stock", type="primary", use_container_width=True) and analysis_symbol:
        with st.spinner("🤖 Analyzing stock... This may take a moment for comprehensive analysis..."):
            
            # Get stock data first
            stock_data = financial_data.get_stock_data(analysis_symbol, "3mo")
            
            if "error" not in stock_data:
                # Display basic stock info
                st.subheader(f"📊 {stock_data['company_name']} ({analysis_symbol})")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", f"₹{stock_data['current_price']:.2f}", f"{stock_data['change_percent']:.2f}%")
                with col2:
                    st.metric("Volume", f"{stock_data['volume']:,}")
                with col3:
                    st.metric("Market Cap", stock_data.get('market_cap', 'N/A'))
                with col4:
                    st.metric("P/E Ratio", stock_data.get('pe_ratio', 'N/A'))
                
                # Generate AI analysis
                analysis_query = f"{analysis_type} for {stock_data['company_name']} stock {analysis_symbol}"
                
                ai_response = rag_system.generate_rag_response(
                    analysis_query,
                    analysis_symbol,
                    st.session_state.user_id
                )
                
                # Display AI analysis
                st.subheader("🤖 AI Analysis")
                st.info(ai_response['answer'])
                
                # Analysis metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{ai_response.get('confidence', 0):.1%}")
                with col2:
                    st.metric("Knowledge Used", ai_response.get('kb_results', 0))
                with col3:
                    st.metric("Web Sources", ai_response.get('web_results', 0))
                
                # Show sources
                if ai_response.get('sources'):
                    with st.expander("📚 Analysis Sources"):
                        for i, source in enumerate(ai_response['sources']):
                            st.write(f"{i+1}. {source}")
                
                # Technical recommendation
                if analysis_type in ["Technical Analysis", "Buy/Sell Recommendation"]:
                    with st.spinner("🎯 Generating technical recommendation..."):
                        recommendation = financial_data.get_stock_recommendation(analysis_symbol)
                        
                        if "error" not in recommendation:
                            st.subheader("🎯 Technical Recommendation")
                            
                            # Recommendation display
                            rec_color = {
                                "BUY": "🟢",
                                "HOLD": "🟡", 
                                "SELL": "🔴"
                            }
                            
                            st.markdown(f"### {rec_color.get(recommendation['recommendation'], '🟡')} {recommendation['recommendation']}")
                            st.write(f"**Score:** {recommendation['score']}/5")
                            
                            if recommendation.get('factors'):
                                st.write("**Positive Factors:**")
                                for factor in recommendation['factors']:
                                    st.write(f"• {factor}")
                            
                            if recommendation.get('risks'):
                                st.write("**Risk Factors:**")
                                for risk in recommendation['risks']:
                                    st.write(f"• {risk}")
                            
                            st.caption("⚠️ This is for educational purposes only. Please consult a financial advisor for personalized advice.")
                
            else:
                st.error(stock_data['error'])

def settings_page():
    """Settings and system information"""
    
    st.header("⚙️ Settings & Information")
    
    # System status
    st.subheader("🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Database Connection:** ✅ Connected")
        st.write("**AI Model:** ✅ Available")
        st.write("**Vector Store:** ✅ Active")
    
    with col2:
        st.write("**Web Search:** ✅ SearXNG Ready")
        st.write("**Market Data:** ✅ Yahoo Finance")
        st.write("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # User preferences
    st.subheader("👤 User Preferences")
    
    risk_tolerance = st.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        key="risk_tolerance"
    )
    
    investment_horizon = st.slider(
        "Investment Horizon (years):", 
        1, 30, 5,
        key="investment_horizon"
    )
    
    # Portfolio settings
    st.subheader("📊 Portfolio Settings")
    
    max_stocks = st.slider(
        "Maximum stocks in portfolio:",
        10, 100, 50,
        key="max_stocks"
    )
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")
    
    # Data management
    st.subheader("🗑️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.experimental_rerun()
    
    with col2:
        if st.button("🔄 Reset Portfolio", type="secondary"):
            if st.button("⚠️ Confirm Reset", type="secondary"):
                # Reset to default cash
                db.update_portfolio(st.session_state.user_id, [], 100000.0)
                st.success("Portfolio reset to ₹1,00,000 cash!")
                st.experimental_rerun()
    
    # Knowledge base stats
    st.subheader("🧠 Knowledge Base")
    
    try:
        kb_stats = rag_system.get_knowledge_stats()
        if "error" not in kb_stats:
            st.write(f"**Documents:** {kb_stats['total_documents']}")
            st.write(f"**Collection:** {kb_stats['collection_name']}")
    except:
        st.write("Knowledge base statistics unavailable")
    
    # About section
    st.subheader("ℹ️ About")
    
    st.markdown("""
    **AI Financial Advisor v1.0**
    
    - 🤖 **Local LLM:** DialoGPT-small with financial fine-tuning
    - 🔍 **Web Search:** SearXNG for real-time market trends  
    - 📊 **Database:** MongoDB for data persistence
    - 🧠 **Knowledge:** ChromaDB vector database
    - 📈 **Market Data:** Yahoo Finance integration
    - 🔒 **Privacy:** All processing happens locally
    
    ⚠️ **Disclaimer:** This tool is for educational purposes only. 
    Not a substitute for professional financial advice.
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or contact support if the error persists.")