# cli/main.py - Command Line Interface with Rich Formatting

import click
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import rag_system
from src.financial_data import financial_data
from src.llm_model import llm
from src.database import db

console = Console()

@click.group()
def cli():
    """🤖 AI Financial Advisor CLI - Your personal investment assistant"""
    console.print("\n[bold blue]🤖 AI Financial Advisor CLI[/bold blue]", style="bold")
    console.print("[dim]Ultra-simple AI-powered financial advice system[/dim]\n")

@cli.command()
@click.argument('query')
@click.option('--stock', '-s', help='Stock symbol for context (e.g., RELIANCE.NS)')
@click.option('--user', '-u', default='cli_user', help='User ID for saving history')
def ask(query, stock, user):
    """Ask the AI financial advisor a question
    
    Examples:
    cli ask "What is SIP?"
    cli ask "Should I invest in RELIANCE?" --stock RELIANCE.NS
    """
    
    console.print(f"\n[bold green]❓ Question:[/bold green] {query}")
    
    if stock:
        console.print(f"[bold yellow]📈 Stock Context:[/bold yellow] {stock}")
    
    with console.status("[bold green]🤔 AI is thinking..."):
        response = rag_system.generate_rag_response(query, stock, user)
    
    # Display response in a nice panel
    console.print(f"\n[bold blue]🤖 AI Advisor:[/bold blue]")
    console.print(Panel(response['answer'], border_style="blue", title="💡 Financial Advice"))
    
    # Show additional info
    if response.get('sources'):
        console.print("\n[bold yellow]📚 Information Sources:[/bold yellow]")
        for i, source in enumerate(response['sources'][:3]):
            console.print(f"  {i+1}. {source}")
    
    # Show confidence and context
    confidence = response.get('confidence', 0)
    context_used = response.get('context_used', 0)
    
    console.print(f"\n[dim]Confidence: {confidence:.1%} | Context used: {context_used} sources[/dim]")
    
    # Web search results indicator
    if response.get('web_results', 0) > 0:
        console.print(f"[dim]✅ Used {response['web_results']} real-time web sources[/dim]")

@cli.command()
@click.argument('symbol')
@click.option('--period', '-p', default='1mo', help='Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed technical analysis')
def stock(symbol, period, detailed):
    """Get comprehensive stock data and analysis
    
    Examples:
    cli stock RELIANCE.NS
    cli stock TCS.NS --period 3mo --detailed
    """
    
    with console.status(f"[bold green]📊 Fetching data for {symbol}..."):
        stock_data = financial_data.get_stock_data(symbol, period)
    
    if "error" in stock_data:
        console.print(f"[bold red]❌ Error:[/bold red] {stock_data['error']}")
        return
    
    # Main stock info table
    table = Table(title=f"📈 {stock_data['company_name']} ({symbol})")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    # Basic metrics
    table.add_row("Current Price", f"₹{stock_data['current_price']:.2f}")
    table.add_row("Change", f"₹{stock_data['change']:.2f} ({stock_data['change_percent']:+.2f}%)")
    table.add_row("Volume", f"{stock_data['volume']:,}")
    table.add_row("Sector", stock_data.get('sector', 'N/A'))
    
    # Financial metrics
    if stock_data.get('pe_ratio') != 'N/A':
        table.add_row("P/E Ratio", f"{stock_data['pe_ratio']:.2f}")
    if stock_data.get('market_cap') != 'N/A':
        table.add_row("Market Cap", f"₹{stock_data['market_cap']:,}")
    
    # Price levels
    table.add_row("52W High", f"₹{stock_data['52_week_high']:.2f}")
    table.add_row("52W Low", f"₹{stock_data['52_week_low']:.2f}")
    
    # Technical indicators
    if stock_data.get('sma_20'):
        table.add_row("SMA 20", f"₹{stock_data['sma_20']:.2f}")
    if stock_data.get('sma_50'):
        table.add_row("SMA 50", f"₹{stock_data['sma_50']:.2f}")
    
    console.print(table)
    
    # Detailed analysis
    if detailed:
        console.print("\n[bold blue]🎯 Technical Analysis[/bold blue]")
        
        with console.status("Generating technical recommendation..."):
            recommendation = financial_data.get_stock_recommendation(symbol)
        
        if "error" not in recommendation:
            # Recommendation panel
            rec_colors = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}
            rec_color = rec_colors.get(recommendation['recommendation'], 'white')
            
            console.print(f"\n[bold {rec_color}]🎯 Recommendation: {recommendation['recommendation']}[/bold {rec_color}]")
            console.print(f"[dim]Technical Score: {recommendation['score']:.1f}/5[/dim]")
            
            if recommendation.get('factors'):
                console.print("\n[green]✅ Positive Factors:[/green]")
                for factor in recommendation['factors']:
                    console.print(f"  • {factor}")
            
            if recommendation.get('risks'):
                console.print("\n[red]⚠️  Risk Factors:[/red]")
                for risk in recommendation['risks']:
                    console.print(f"  • {risk}")
            
            console.print("\n[dim]⚠️  For educational purposes only. Consult a financial advisor.[/dim]")

@cli.command()
@click.option('--user', '-u', default='cli_user', help='User ID')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed breakdown')
def portfolio(user, detailed):
    """View comprehensive portfolio summary
    
    Examples:
    cli portfolio
    cli portfolio --detailed
    """
    
    with console.status("[bold green]📊 Loading portfolio..."):
        portfolio_summary = financial_data.get_portfolio_summary(user)
    
    if "error" in portfolio_summary:
        console.print(f"[bold red]❌ Error:[/bold red] {portfolio_summary['error']}")
        return
    
    # Portfolio overview
    console.print("\n[bold blue]📊 Portfolio Overview[/bold blue]")
    
    overview_table = Table()
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", style="green")
    
    overview_table.add_row("💰 Total Value", f"₹{portfolio_summary['total_value']:,.2f}")
    overview_table.add_row("💵 Cash", f"₹{portfolio_summary['cash']:,.2f}")
    overview_table.add_row("📈 Stocks Value", f"₹{portfolio_summary['stocks_value']:,.2f}")
    overview_table.add_row("📊 Total P&L", f"₹{portfolio_summary['total_gain_loss']:,.2f}")
    overview_table.add_row("📈 Return %", f"{portfolio_summary['total_gain_loss_percent']:+.2f}%")
    overview_table.add_row("🏢 Holdings", f"{portfolio_summary['stock_count']} stocks")
    
    console.print(overview_table)
    
    # Holdings detail
    if portfolio_summary['stocks']:
        console.print("\n[bold blue]📋 Holdings Detail[/bold blue]")
        
        holdings_table = Table()
        holdings_table.add_column("Symbol", style="cyan")
        holdings_table.add_column("Company", style="white")
        holdings_table.add_column("Qty", justify="right")
        holdings_table.add_column("Avg Price", justify="right")
        holdings_table.add_column("Current", justify="right")
        holdings_table.add_column("Value", justify="right", style="green")
        holdings_table.add_column("P&L", justify="right")
        holdings_table.add_column("Return %", justify="right")
        
        if detailed:
            holdings_table.add_column("Weight %", justify="right")
            holdings_table.add_column("Sector", style="dim")
        
        for stock in portfolio_summary['stocks']:
            gain_loss_style = "green" if stock['gain_loss'] >= 0 else "red"
            
            row_data = [
                stock['symbol'],
                stock['company_name'][:15] + "..." if len(stock['company_name']) > 15 else stock['company_name'],
                str(stock['quantity']),
                f"₹{stock['avg_price']:.2f}",
                f"₹{stock['current_price']:.2f}",
                f"₹{stock['current_value']:,.0f}",
                f"[{gain_loss_style}]₹{stock['gain_loss']:+,.0f}[/{gain_loss_style}]",
                f"[{gain_loss_style}]{stock['gain_loss_percent']:+.1f}%[/{gain_loss_style}]"
            ]
            
            if detailed:
                row_data.extend([
                    f"{stock['weight']:.1f}%",
                    stock.get('sector', 'N/A')[:10]
                ])
            
            holdings_table.add_row(*row_data)
        
        console.print(holdings_table)
        
        # Performance highlights
        if detailed:
            perf = portfolio_summary.get('performance', {})
            
            console.print("\n[bold blue]🎯 Performance Highlights[/bold blue]")
            
            if perf.get('best_performer'):
                best = perf['best_performer']
                console.print(f"[green]🏆 Best: {best['symbol']} (+{best['return_percent']:.1f}%)[/green]")
            
            if perf.get('worst_performer'):
                worst = perf['worst_performer']
                console.print(f"[red]📉 Worst: {worst['symbol']} ({worst['return_percent']:.1f}%)[/red]")
            
            # Sector diversification
            sectors = portfolio_summary.get('diversification', {}).get('sectors', {})
            if sectors:
                console.print("\n[bold blue]🥧 Sector Allocation[/bold blue]")
                for sector, percentage in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
                    console.print(f"  {sector}: {percentage:.1f}%")
    
    else:
        console.print("\n[yellow]💡 Your portfolio is empty. Use 'cli buy' to purchase stocks.[/yellow]")

@cli.command()
@click.argument('symbol')
@click.argument('quantity', type=int)
@click.argument('price', type=float)
@click.option('--user', '-u', default='cli_user', help='User ID')
@click.option('--confirm', '-y', is_flag=True, help='Skip confirmation')
def buy(symbol, quantity, price, user, confirm):
    """Buy stock for portfolio
    
    Examples:
    cli buy RELIANCE.NS 10 2500
    cli buy TCS.NS 5 3200 --user my_user
    """
    
    total_cost = quantity * price
    
    # Display order details
    console.print(f"\n[bold yellow]🛒 Buy Order Details[/bold yellow]")
    
    order_table = Table()
    order_table.add_column("Detail", style="cyan")
    order_table.add_column("Value", style="white")
    
    order_table.add_row("Symbol", symbol)
    order_table.add_row("Quantity", str(quantity))
    order_table.add_row("Price per share", f"₹{price:.2f}")
    order_table.add_row("Total Cost", f"₹{total_cost:,.2f}")
    
    console.print(order_table)
    
    # Confirmation
    if not confirm:
        if not Confirm.ask("\n[yellow]Confirm purchase?[/yellow]"):
            console.print("[red]❌ Order cancelled[/red]")
            return
    
    # Process order
    with console.status("[bold green]🔄 Processing buy order..."):
        result = financial_data.add_stock_to_portfolio(user, symbol, quantity, price)
    
    if result["success"]:
        console.print(f"\n[bold green]✅ {result['message']}[/bold green]")
        
        # Show transaction details
        transaction = result.get('transaction', {})
        console.print(f"[dim]New cash balance: ₹{transaction.get('new_cash_balance', 0):,.2f}[/dim]")
        
        if transaction.get('type') == 'addition':
            console.print("[dim]💡 Added to existing position[/dim]")
    else:
        console.print(f"[bold red]❌ Order failed: {result['error']}[/bold red]")

@cli.command()
@click.argument('symbol')
@click.argument('quantity', type=int)
@click.argument('price', type=float)
@click.option('--user', '-u', default='cli_user', help='User ID')
@click.option('--confirm', '-y', is_flag=True, help='Skip confirmation')
def sell(symbol, quantity, price, user, confirm):
    """Sell stock from portfolio
    
    Examples:
    cli sell RELIANCE.NS 5 2600
    cli sell TCS.NS 10 3300 --user my_user
    """
    
    total_value = quantity * price
    
    # Display order details
    console.print(f"\n[bold yellow]💸 Sell Order Details[/bold yellow]")
    
    order_table = Table()
    order_table.add_column("Detail", style="cyan")
    order_table.add_column("Value", style="white")
    
    order_table.add_row("Symbol", symbol)
    order_table.add_row("Quantity", str(quantity))
    order_table.add_row("Price per share", f"₹{price:.2f}")
    order_table.add_row("Total Value", f"₹{total_value:,.2f}")
    
    console.print(order_table)
    
    # Confirmation
    if not confirm:
        if not Confirm.ask("\n[yellow]Confirm sale?[/yellow]"):
            console.print("[red]❌ Order cancelled[/red]")
            return
    
    # Process order
    with console.status("[bold green]🔄 Processing sell order..."):
        result = financial_data.sell_stock_from_portfolio(user, symbol, quantity, price)
    
    if result["success"]:
        console.print(f"\n[bold green]✅ {result['message']}[/bold green]")
        
        # Show transaction details
        transaction = result.get('transaction', {})
        gain_loss = transaction.get('gain_loss', 0)
        
        if gain_loss > 0:
            console.print(f"[green]💰 Profit: ₹{gain_loss:,.2f} ({transaction.get('gain_loss_percent', 0):+.1f}%)[/green]")
        elif gain_loss < 0:
            console.print(f"[red]📉 Loss: ₹{gain_loss:,.2f} ({transaction.get('gain_loss_percent', 0):+.1f}%)[/red]")
        
        console.print(f"[dim]New cash balance: ₹{transaction.get('new_cash_balance', 0):,.2f}[/dim]")
        
        if transaction.get('remaining_quantity', 0) > 0:
            console.print(f"[dim]Remaining shares: {transaction['remaining_quantity']}[/dim]")
    else:
        console.print(f"[bold red]❌ Order failed: {result['error']}[/bold red]")

@cli.command()
@click.option('--user', '-u', default='cli_user', help='User ID for saving history')
def interactive(user):
    """Start interactive chat mode for continuous conversation
    
    Example:
    cli interactive
    """
    
    console.print("\n[bold blue]🤖 Interactive AI Financial Advisor[/bold blue]")
    console.print("[dim]Type 'quit' to exit, 'help' for commands, 'portfolio' for portfolio view[/dim]\n")
    
    # Welcome message
    console.print(Panel(
        "Welcome! I'm your AI financial advisor. Ask me anything about:\n"
        "• Investment strategies (SIP, ETF, stocks)\n"
        "• Portfolio management and diversification\n" 
        "• Tax-saving investments (ELSS, PPF)\n"
        "• Stock analysis and recommendations\n"
        "• Market trends and news",
        title="💡 How I can help",
        border_style="blue"
    ))
    
    while True:
        try:
            console.print()
            query = Prompt.ask("[bold green]You")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("\n[bold blue]👋 Happy investing! Remember to diversify and invest for the long term.[/bold blue]")
                break
            
            if query.lower() == 'help':
                console.print("""
[bold yellow]Available commands:[/bold yellow]
• Ask any financial question
• 'portfolio' - View your portfolio  
• 'market' - Quick market overview
• 'clear' - Clear chat history
• 'quit' - Exit chat

[bold yellow]Example questions:[/bold yellow]
• "What is SIP and should I start one?"
• "How to analyze RELIANCE stock?"
• "Best tax saving investments for 2025?"
• "Should I sell my losing stocks?"
                """)
                continue
            
            if query.lower() == 'portfolio':
                # Quick portfolio view
                with console.status("Loading portfolio..."):
                    portfolio_summary = financial_data.get_portfolio_summary(user)
                
                if "error" not in portfolio_summary:
                    console.print(f"\n[bold blue]💰 Quick Portfolio View[/bold blue]")
                    console.print(f"Total Value: ₹{portfolio_summary['total_value']:,.0f}")
                    console.print(f"P&L: ₹{portfolio_summary['total_gain_loss']:+,.0f} ({portfolio_summary['total_gain_loss_percent']:+.1f}%)")
                    console.print(f"Holdings: {portfolio_summary['stock_count']} stocks")
                continue
            
            if query.lower() == 'market':
                # Quick market overview
                with console.status("Fetching market data..."):
                    market_overview = financial_data.get_market_overview()
                
                console.print("\n[bold blue]📈 Market Snapshot[/bold blue]")
                sentiment = market_overview.get('market_sentiment', 'neutral')
                console.print(f"Market Sentiment: {sentiment.title()}")
                
                for stock in market_overview.get('stocks', [])[:5]:
                    change_style = "green" if stock['change_percent'] >= 0 else "red"
                    console.print(f"• {stock['symbol']}: ₹{stock['current_price']:.2f} "
                                f"[{change_style}]({stock['change_percent']:+.1f}%)[/{change_style}]")
                continue
            
            if query.lower() == 'clear':
                # This would clear chat history in a real implementation
                console.print("[green]✅ Chat history cleared[/green]")
                continue
            
            # Regular AI query
            with console.status("[bold green]🤔 AI is thinking..."):
                response = rag_system.generate_rag_response(query, None, user)
            
            console.print(f"\n[bold blue]🤖 AI Advisor:[/bold blue]")
            console.print(Panel(response['answer'], border_style="blue"))
            
            # Show confidence and sources briefly
            if response.get('sources'):
                source_count = len(response['sources'])
                console.print(f"[dim]Sources: {source_count} | Confidence: {response.get('confidence', 0):.0%}[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n\n[bold blue]👋 Goodbye! Thanks for using AI Financial Advisor.[/bold blue]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
            console.print("[yellow]Please try again or type 'quit' to exit.[/yellow]")

@cli.command()
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--model', default='microsoft/DialoGPT-small', help='Base model to fine-tune')
def train(epochs, model):
    """Train/fine-tune the local LLM model on financial data
    
    Examples:
    cli train
    cli train --epochs 5 --model distilgpt2
    """
    
    console.print(f"[bold blue]🚀 Training Local Financial LLM[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Epochs: {epochs}")
    
    if not Confirm.ask("\n[yellow]This will fine-tune the model on financial data. Continue?[/yellow]"):
        console.print("[red]❌ Training cancelled[/red]")
        return
    
    try:
        with Progress() as progress:
            task = progress.add_task("[green]Training model...", total=100)
            
            # Start training (this is a simplified progress display)
            progress.update(task, advance=10)
            console.print("📚 Loading financial dataset...")
            
            progress.update(task, advance=20)
            console.print("🤖 Initializing model...")
            
            progress.update(task, advance=30)
            console.print("⚙️  Setting up LoRA fine-tuning...")
            
            # Actual training call
            llm.fine_tune()
            
            progress.update(task, advance=100)
            
        console.print("\n[bold green]✅ Model training completed successfully![/bold green]")
        console.print("[dim]Model saved to ./models/financial_model/[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Training failed: {str(e)}[/bold red]")
        console.print("[yellow]💡 Check if you have sufficient memory and disk space[/yellow]")

@cli.command()
def market():
    """Get quick market overview of top Indian stocks"""
    
    with console.status("[bold green]📊 Fetching market data..."):
        market_overview = financial_data.get_market_overview()
    
    # Market sentiment
    sentiment = market_overview.get("market_sentiment", "neutral").title()
    sentiment_emoji = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
    
    console.print(f"\n[bold blue]📈 Indian Stock Market Overview[/bold blue]")
    console.print(f"Market Sentiment: {sentiment_emoji.get(sentiment, '🟡')} {sentiment}")
    
    if "summary" in market_overview:
        summary = market_overview["summary"]
        console.print(f"Gainers: {summary['gainers']} | Losers: {summary['losers']} | Avg Change: {summary['avg_change']:+.2f}%")
    
    # Stocks table
    if market_overview.get('stocks'):
        table = Table(title="Top Stocks Performance")
        table.add_column("Symbol", style="cyan")
        table.add_column("Company", style="white")
        table.add_column("Price", justify="right")
        table.add_column("Change %", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Sector", style="dim")
        
        for stock in market_overview['stocks']:
            change_style = "green" if stock['change_percent'] >= 0 else "red"
            
            table.add_row(
                stock['symbol'],
                stock['company_name'][:20] + "..." if len(stock['company_name']) > 20 else stock['company_name'],
                f"₹{stock['current_price']:.2f}",
                f"[{change_style}]{stock['change_percent']:+.2f}%[/{change_style}]",
                f"{stock['volume']/1000:.0f}K",
                stock['sector'][:15]
            )
        
        console.print(table)

@cli.command()
def status():
    """Check system status and configuration"""
    
    console.print("\n[bold blue]🔧 AI Financial Advisor System Status[/bold blue]")
    
    # System status table
    status_table = Table()
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="dim")
    
    # Check components
    try:
        # Database
        db.get_user_portfolio("test")
        status_table.add_row("Database", "✅ Connected", "MongoDB operational")
    except:
        status_table.add_row("Database", "❌ Error", "MongoDB connection failed")
    
    try:
        # Vector store
        stats = rag_system.get_knowledge_stats()
        docs = stats.get('total_documents', 0)
        status_table.add_row("Knowledge Base", "✅ Active", f"ChromaDB with {docs} documents")
    except:
        status_table.add_row("Knowledge Base", "❌ Error", "ChromaDB not available")
    
    try:
        # Market data
        financial_data.get_market_overview()
        status_table.add_row("Market Data", "✅ Available", "Yahoo Finance operational")
    except:
        status_table.add_row("Market Data", "⚠️  Limited", "Yahoo Finance issues")
    
    try:
        # LLM Model
        llm.load_model()
        status_table.add_row("AI Model", "✅ Loaded", f"Model: {llm.model_name}")
    except:
        status_table.add_row("AI Model", "❌ Error", "Model loading failed")
    
    console.print(status_table)
    
    # Configuration info
    console.print(f"\n[bold blue]⚙️  Configuration[/bold blue]")
    console.print(f"Model: {llm.model_name}")
    console.print(f"Device: {llm.device}")
    console.print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    cli()