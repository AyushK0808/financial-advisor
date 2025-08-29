# src/financial_data.py - Financial Data Processing with Yahoo Finance

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.config import config
from src.database import db

class FinancialDataProcessor:
    """Handle financial data fetching and portfolio management"""
    
    def __init__(self):
        self.default_stocks = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
            "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS"
        ]
    
    def get_stock_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Get comprehensive stock data from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for symbol {symbol}"}
            
            # Get current info
            info = ticker.info
            
            # Calculate current metrics
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_price
            change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
            
            # Technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else None
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
            
            # Volatility (standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0  # Annualized
            
            # Volume analysis
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price levels
            week_52_high = hist['High'].max()
            week_52_low = hist['Low'].min()
            current_vs_52w_high = (current_price / week_52_high) * 100 if week_52_high > 0 else 100
            
            # Build comprehensive stock data
            stock_data = {
                "symbol": symbol,
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                
                # Price information
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "prev_close": round(prev_price, 2),
                
                # Volume information
                "volume": int(current_volume),
                "avg_volume": int(avg_volume),
                "volume_ratio": round(volume_ratio, 2),
                
                # Technical indicators
                "sma_20": round(sma_20, 2) if sma_20 else None,
                "sma_50": round(sma_50, 2) if sma_50 else None,
                "volatility": round(volatility, 2),
                
                # Price levels
                "52_week_high": round(week_52_high, 2),
                "52_week_low": round(week_52_low, 2),
                "current_vs_52w_high": round(current_vs_52w_high, 1),
                
                # Fundamental data
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "pb_ratio": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else "N/A",
                "roe": info.get("returnOnEquity", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                
                # Additional info
                "currency": info.get("currency", "INR"),
                "exchange": info.get("exchange", "NSI"),
                "last_updated": datetime.now().isoformat()
            }
            
            # Save to database
            db.save_market_data(symbol, stock_data)
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}
    
    def get_stock_history(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical stock data for charts"""
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No historical data found for {symbol}"}
            
            # Convert to dict format for frontend
            history_data = {
                "symbol": symbol,
                "period": period,
                "dates": hist.index.strftime('%Y-%m-%d').tolist(),
                "open": hist['Open'].round(2).tolist(),
                "high": hist['High'].round(2).tolist(),
                "low": hist['Low'].round(2).tolist(),
                "close": hist['Close'].round(2).tolist(),
                "volume": hist['Volume'].tolist(),
                "last_updated": datetime.now().isoformat()
            }
            
            return history_data
            
        except Exception as e:
            print(f"Error getting history for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio summary with current values"""
        
        try:
            portfolio = db.get_user_portfolio(user_id)
            
            if not portfolio.get("stocks"):
                return {
                    "total_value": portfolio.get("cash", 100000.0),
                    "cash": portfolio.get("cash", 100000.0),
                    "stocks_value": 0,
                    "stocks": [],
                    "total_gain_loss": 0,
                    "total_gain_loss_percent": 0,
                    "diversification": {"sectors": {}, "stocks": 0},
                    "performance": {"best_performer": None, "worst_performer": None},
                    "last_updated": datetime.now().isoformat()
                }
            
            stocks_data = []
            total_stocks_value = 0
            total_invested = 0
            sector_allocation = {}
            
            # Process each stock
            for stock in portfolio["stocks"]:
                symbol = stock["symbol"]
                quantity = stock["quantity"]
                avg_price = stock["avg_price"]
                
                # Get current market data
                current_data = self.get_stock_data(symbol, period="1d")
                
                if "error" not in current_data:
                    current_price = current_data["current_price"]
                    current_value = quantity * current_price
                    invested_value = quantity * avg_price
                    gain_loss = current_value - invested_value
                    gain_loss_percent = (gain_loss / invested_value) * 100 if invested_value > 0 else 0
                    
                    stock_summary = {
                        "symbol": symbol,
                        "company_name": current_data.get("company_name", symbol),
                        "sector": current_data.get("sector", "Unknown"),
                        "quantity": quantity,
                        "avg_price": round(avg_price, 2),
                        "current_price": current_price,
                        "invested_value": round(invested_value, 2),
                        "current_value": round(current_value, 2),
                        "gain_loss": round(gain_loss, 2),
                        "gain_loss_percent": round(gain_loss_percent, 2),
                        "change_percent": current_data.get("change_percent", 0),
                        "weight": 0  # Will calculate after total
                    }
                    
                    stocks_data.append(stock_summary)
                    total_stocks_value += current_value
                    total_invested += invested_value
                    
                    # Track sector allocation
                    sector = current_data.get("sector", "Unknown")
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + current_value
            
            # Calculate portfolio weights
            for stock in stocks_data:
                stock["weight"] = round((stock["current_value"] / total_stocks_value) * 100, 1) if total_stocks_value > 0 else 0
            
            # Calculate total metrics
            total_gain_loss = total_stocks_value - total_invested
            total_gain_loss_percent = (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0
            
            # Find best and worst performers
            best_performer = max(stocks_data, key=lambda x: x["gain_loss_percent"]) if stocks_data else None
            worst_performer = min(stocks_data, key=lambda x: x["gain_loss_percent"]) if stocks_data else None
            
            # Sector allocation percentages
            sector_percentages = {}
            for sector, value in sector_allocation.items():
                sector_percentages[sector] = round((value / total_stocks_value) * 100, 1) if total_stocks_value > 0 else 0
            
            summary = {
                "total_value": round(portfolio.get("cash", 0) + total_stocks_value, 2),
                "cash": round(portfolio.get("cash", 0), 2),
                "stocks_value": round(total_stocks_value, 2),
                "total_invested": round(total_invested, 2),
                "total_gain_loss": round(total_gain_loss, 2),
                "total_gain_loss_percent": round(total_gain_loss_percent, 2),
                "stocks": stocks_data,
                "stock_count": len(stocks_data),
                
                "diversification": {
                    "sectors": sector_percentages,
                    "stocks": len(stocks_data),
                    "top_holding_weight": max([s["weight"] for s in stocks_data]) if stocks_data else 0
                },
                
                "performance": {
                    "best_performer": {
                        "symbol": best_performer["symbol"],
                        "return_percent": best_performer["gain_loss_percent"]
                    } if best_performer else None,
                    "worst_performer": {
                        "symbol": worst_performer["symbol"], 
                        "return_percent": worst_performer["gain_loss_percent"]
                    } if worst_performer else None
                },
                
                "last_updated": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting portfolio summary: {str(e)}")
            return {"error": str(e)}
    
    def add_stock_to_portfolio(self, user_id: str, symbol: str, quantity: int, price: float) -> Dict[str, Any]:
        """Add stock to user portfolio with detailed validation"""
        
        try:
            # Validate stock symbol
            stock_data = self.get_stock_data(symbol, period="1d")
            if "error" in stock_data:
                return {"success": False, "error": f"Invalid stock symbol: {symbol}"}
            
            portfolio = db.get_user_portfolio(user_id)
            stocks = portfolio.get("stocks", [])
            cash = portfolio.get("cash", 0)
            
            total_cost = quantity * price
            
            # Check sufficient funds
            if cash < total_cost:
                return {
                    "success": False, 
                    "error": f"Insufficient funds. Available: ₹{cash:.2f}, Required: ₹{total_cost:.2f}"
                }
            
            # Check portfolio size limit
            if len(stocks) >= config.max_portfolio_size:
                existing_symbols = [s["symbol"] for s in stocks]
                if symbol not in existing_symbols:
                    return {
                        "success": False,
                        "error": f"Portfolio limit reached ({config.max_portfolio_size} stocks)"
                    }
            
            # Check if stock already exists in portfolio
            existing_stock_index = None
            for i, stock in enumerate(stocks):
                if stock["symbol"] == symbol:
                    existing_stock_index = i
                    break
            
            if existing_stock_index is not None:
                # Update existing stock (average price calculation)
                old_quantity = stocks[existing_stock_index]["quantity"]
                old_avg_price = stocks[existing_stock_index]["avg_price"]
                old_total_value = old_quantity * old_avg_price
                
                new_quantity = old_quantity + quantity
                new_avg_price = (old_total_value + total_cost) / new_quantity
                
                stocks[existing_stock_index].update({
                    "quantity": new_quantity,
                    "avg_price": round(new_avg_price, 2),
                    "last_updated": datetime.now().isoformat()
                })
                
                transaction_type = "addition"
            else:
                # Add new stock
                stocks.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": price,
                    "company_name": stock_data.get("company_name", symbol),
                    "sector": stock_data.get("sector", "Unknown"),
                    "added_date": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                })
                
                transaction_type = "new_stock"
            
            # Update cash
            new_cash = cash - total_cost
            
            # Save to database
            db.update_portfolio(user_id, stocks, new_cash)
            
            return {
                "success": True,
                "message": f"Successfully {'added to' if transaction_type == 'addition' else 'bought'} {quantity} shares of {symbol}",
                "transaction": {
                    "type": transaction_type,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "total_cost": total_cost,
                    "new_cash_balance": new_cash
                }
            }
            
        except Exception as e:
            print(f"Error adding stock to portfolio: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def sell_stock_from_portfolio(self, user_id: str, symbol: str, quantity: int, price: float) -> Dict[str, Any]:
        """Sell stock from user portfolio with detailed validation"""
        
        try:
            portfolio = db.get_user_portfolio(user_id)
            stocks = portfolio.get("stocks", [])
            cash = portfolio.get("cash", 0)
            
            # Find stock in portfolio
            stock_index = None
            for i, stock in enumerate(stocks):
                if stock["symbol"] == symbol:
                    if stock["quantity"] >= quantity:
                        stock_index = i
                        break
            
            if stock_index is None:
                return {
                    "success": False,
                    "error": f"Stock {symbol} not found in portfolio or insufficient quantity"
                }
            
            current_stock = stocks[stock_index]
            avg_price = current_stock["avg_price"]
            
            # Calculate profit/loss
            gain_loss_per_share = price - avg_price
            total_gain_loss = gain_loss_per_share * quantity
            sale_value = quantity * price
            
            # Update stock quantity
            remaining_quantity = current_stock["quantity"] - quantity
            
            if remaining_quantity == 0:
                # Remove stock completely
                stocks.pop(stock_index)
                transaction_type = "complete_sale"
            else:
                # Update quantity
                stocks[stock_index].update({
                    "quantity": remaining_quantity,
                    "last_updated": datetime.now().isoformat()
                })
                transaction_type = "partial_sale"
            
            # Update cash
            new_cash = cash + sale_value
            
            # Save to database
            db.update_portfolio(user_id, stocks, new_cash)
            
            return {
                "success": True,
                "message": f"Successfully sold {quantity} shares of {symbol}",
                "transaction": {
                    "type": transaction_type,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "avg_cost": avg_price,
                    "sale_value": sale_value,
                    "gain_loss": round(total_gain_loss, 2),
                    "gain_loss_percent": round((gain_loss_per_share / avg_price) * 100, 2),
                    "new_cash_balance": new_cash,
                    "remaining_quantity": remaining_quantity if transaction_type == "partial_sale" else 0
                }
            }
            
        except Exception as e:
            print(f"Error selling stock from portfolio: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overview of default Indian stocks"""
        
        overview = {
            "stocks": [],
            "market_sentiment": "neutral",
            "last_updated": datetime.now().isoformat()
        }
        
        gainers = 0
        total_change = 0
        
        for symbol in self.default_stocks[:8]:  # Limit to 8 stocks
            stock_data = self.get_stock_data(symbol, period="1d")
            
            if "error" not in stock_data:
                change_pct = stock_data["change_percent"]
                
                overview["stocks"].append({
                    "symbol": symbol,
                    "company_name": stock_data.get("company_name", symbol),
                    "current_price": stock_data["current_price"],
                    "change": stock_data["change"],
                    "change_percent": change_pct,
                    "volume": stock_data["volume"],
                    "sector": stock_data.get("sector", "N/A"),
                    "market_cap": stock_data.get("market_cap", "N/A")
                })
                
                if change_pct > 0:
                    gainers += 1
                total_change += change_pct
        
        # Determine market sentiment
        if len(overview["stocks"]) > 0:
            avg_change = total_change / len(overview["stocks"])
            gainer_ratio = gainers / len(overview["stocks"])
            
            if avg_change > 1 and gainer_ratio > 0.6:
                overview["market_sentiment"] = "bullish"
            elif avg_change < -1 and gainer_ratio < 0.4:
                overview["market_sentiment"] = "bearish"
            else:
                overview["market_sentiment"] = "neutral"
        
        overview["summary"] = {
            "total_stocks": len(overview["stocks"]),
            "gainers": gainers,
            "losers": len(overview["stocks"]) - gainers,
            "avg_change": round(total_change / len(overview["stocks"]), 2) if overview["stocks"] else 0
        }
        
        return overview
    
    def get_stock_recommendation(self, symbol: str) -> Dict[str, Any]:
        """Generate basic stock recommendation based on technical indicators"""
        
        try:
            stock_data = self.get_stock_data(symbol, period="3mo")
            
            if "error" in stock_data:
                return {"error": stock_data["error"]}
            
            recommendation = {
                "symbol": symbol,
                "company_name": stock_data["company_name"],
                "current_price": stock_data["current_price"],
                "recommendation": "HOLD",
                "score": 0,
                "factors": [],
                "risks": [],
                "timestamp": datetime.now().isoformat()
            }
            
            score = 0
            
            # Price vs moving averages
            if stock_data.get("sma_20") and stock_data.get("sma_50"):
                if stock_data["current_price"] > stock_data["sma_20"]:
                    score += 1
                    recommendation["factors"].append("Price above 20-day SMA")
                
                if stock_data["sma_20"] > stock_data["sma_50"]:
                    score += 1
                    recommendation["factors"].append("20-day SMA above 50-day SMA")
            
            # Volume analysis
            if stock_data.get("volume_ratio", 1) > 1.5:
                score += 1
                recommendation["factors"].append("Above average volume")
            
            # 52-week position
            if stock_data.get("current_vs_52w_high", 0) > 80:
                score -= 1
                recommendation["risks"].append("Near 52-week high")
            elif stock_data.get("current_vs_52w_high", 0) < 30:
                score += 1
                recommendation["factors"].append("Significantly below 52-week high")
            
            # Volatility check
            if stock_data.get("volatility", 0) > 40:
                recommendation["risks"].append("High volatility stock")
                score -= 0.5
            
            # PE ratio analysis
            pe_ratio = stock_data.get("pe_ratio")
            if isinstance(pe_ratio, (int, float)) and pe_ratio > 0:
                if pe_ratio < 15:
                    score += 1
                    recommendation["factors"].append("Low P/E ratio")
                elif pe_ratio > 30:
                    score -= 1
                    recommendation["risks"].append("High P/E ratio")
            
            # Final recommendation
            recommendation["score"] = round(score, 1)
            
            if score >= 3:
                recommendation["recommendation"] = "BUY"
            elif score >= 1:
                recommendation["recommendation"] = "HOLD"
            elif score <= -1:
                recommendation["recommendation"] = "SELL"
            else:
                recommendation["recommendation"] = "HOLD"
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            return {"error": str(e)}

# Global financial data processor instance
financial_data = FinancialDataProcessor()