# Complete FinancialDataProcessor with Hardcoded Market Data and Portfolio Management

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random

class FinancialDataProcessor:
    """Handle financial data with comprehensive hardcoded market data and portfolio management"""
    
    def __init__(self):
        # Set seed for consistent daily prices
        random.seed(int(datetime.now().strftime("%Y%m%d")))
        
        # Default stocks list
        self.default_stocks = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
            "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS",
            "WIPRO.NS", "BAJFINANCE.NS"
        ]
        
        # Comprehensive hardcoded stock database
        self.stock_database = {
            "RELIANCE.NS": {
                "company_name": "Reliance Industries Limited",
                "sector": "Energy",
                "industry": "Oil & Gas Refining & Marketing",
                "base_price": 2520.00,
                "current_price": 2534.75,
                "prev_close": 2520.00,
                "change": 14.75,
                "change_percent": 0.59,
                "volume": 2847239,
                "avg_volume": 3250000,
                "market_cap": 17084570000000,  # 17.08 trillion
                "pe_ratio": 28.45,
                "pb_ratio": 2.1,
                "dividend_yield": 0.35,
                "roe": 7.2,
                "debt_to_equity": 0.21,
                "52_week_high": 2856.15,
                "52_week_low": 2220.30,
                "sma_20": 2498.30,
                "sma_50": 2445.80,
                "volatility": 18.5,
                "beta": 0.95
            },
            
            "TCS.NS": {
                "company_name": "Tata Consultancy Services Limited",
                "sector": "Technology",
                "industry": "Information Technology Services",
                "base_price": 3645.00,
                "current_price": 3678.20,
                "prev_close": 3645.00,
                "change": 33.20,
                "change_percent": 0.91,
                "volume": 1234567,
                "avg_volume": 1500000,
                "market_cap": 13456780000000,  # 13.46 trillion
                "pe_ratio": 25.8,
                "pb_ratio": 9.2,
                "dividend_yield": 1.2,
                "roe": 42.1,
                "debt_to_equity": 0.05,
                "52_week_high": 4043.75,
                "52_week_low": 3311.20,
                "sma_20": 3634.50,
                "sma_50": 3591.25,
                "volatility": 16.2,
                "beta": 0.78
            },
            
            "INFY.NS": {
                "company_name": "Infosys Limited",
                "sector": "Technology", 
                "industry": "Information Technology Services",
                "base_price": 1485.00,
                "current_price": 1492.35,
                "prev_close": 1485.00,
                "change": 7.35,
                "change_percent": 0.49,
                "volume": 3456789,
                "avg_volume": 4200000,
                "market_cap": 6234560000000,  # 6.23 trillion
                "pe_ratio": 24.3,
                "pb_ratio": 7.8,
                "dividend_yield": 2.1,
                "roe": 31.5,
                "debt_to_equity": 0.08,
                "52_week_high": 1729.35,
                "52_week_low": 1351.65,
                "sma_20": 1478.90,
                "sma_50": 1456.75,
                "volatility": 19.8,
                "beta": 0.82
            },
            
            "HDFCBANK.NS": {
                "company_name": "HDFC Bank Limited",
                "sector": "Financial Services",
                "industry": "Private Sector Bank",
                "base_price": 1598.00,
                "current_price": 1612.45,
                "prev_close": 1598.00,
                "change": 14.45,
                "change_percent": 0.90,
                "volume": 5678901,
                "avg_volume": 6800000,
                "market_cap": 12345670000000,  # 12.35 trillion
                "pe_ratio": 18.7,
                "pb_ratio": 2.9,
                "dividend_yield": 1.0,
                "roe": 16.8,
                "debt_to_equity": 0.12,
                "52_week_high": 1740.00,
                "52_week_low": 1363.55,
                "sma_20": 1585.30,
                "sma_50": 1561.20,
                "volatility": 15.4,
                "beta": 0.88
            },
            
            "ICICIBANK.NS": {
                "company_name": "ICICI Bank Limited",
                "sector": "Financial Services",
                "industry": "Private Sector Bank",
                "base_price": 985.00,
                "current_price": 991.80,
                "prev_close": 985.00,
                "change": 6.80,
                "change_percent": 0.69,
                "volume": 7890123,
                "avg_volume": 9500000,
                "market_cap": 6789012000000,  # 6.79 trillion
                "pe_ratio": 15.2,
                "pb_ratio": 2.1,
                "dividend_yield": 0.8,
                "roe": 15.3,
                "debt_to_equity": 0.15,
                "52_week_high": 1257.85,
                "52_week_low": 926.05,
                "sma_20": 978.45,
                "sma_50": 1012.30,
                "volatility": 22.1,
                "beta": 1.12
            },
            
            "HINDUNILVR.NS": {
                "company_name": "Hindustan Unilever Limited",
                "sector": "Consumer Goods",
                "industry": "Personal Care",
                "base_price": 2690.00,
                "current_price": 2698.50,
                "prev_close": 2690.00,
                "change": 8.50,
                "change_percent": 0.32,
                "volume": 1098765,
                "avg_volume": 1300000,
                "market_cap": 6321450000000,  # 6.32 trillion
                "pe_ratio": 58.9,
                "pb_ratio": 12.4,
                "dividend_yield": 1.5,
                "roe": 21.2,
                "debt_to_equity": 0.01,
                "52_week_high": 2844.95,
                "52_week_low": 2172.00,
                "sma_20": 2681.75,
                "sma_50": 2645.90,
                "volatility": 13.8,
                "beta": 0.65
            },
            
            "ITC.NS": {
                "company_name": "ITC Limited",
                "sector": "Consumer Goods",
                "industry": "Tobacco Products",
                "base_price": 448.00,
                "current_price": 451.25,
                "prev_close": 448.00,
                "change": 3.25,
                "change_percent": 0.73,
                "volume": 12345678,
                "avg_volume": 15000000,
                "market_cap": 5612340000000,  # 5.61 trillion
                "pe_ratio": 23.1,
                "pb_ratio": 4.2,
                "dividend_yield": 4.2,
                "roe": 18.5,
                "debt_to_equity": 0.03,
                "52_week_high": 487.25,
                "52_week_low": 399.35,
                "sma_20": 445.60,
                "sma_50": 441.85,
                "volatility": 16.7,
                "beta": 0.71
            },
            
            "SBIN.NS": {
                "company_name": "State Bank of India",
                "sector": "Financial Services",
                "industry": "Public Sector Bank",
                "base_price": 645.00,
                "current_price": 651.30,
                "prev_close": 645.00,
                "change": 6.30,
                "change_percent": 0.98,
                "volume": 23456789,
                "avg_volume": 28000000,
                "market_cap": 5812340000000,  # 5.81 trillion
                "pe_ratio": 12.8,
                "pb_ratio": 1.1,
                "dividend_yield": 1.8,
                "roe": 9.2,
                "debt_to_equity": 0.18,
                "52_week_high": 725.90,
                "52_week_low": 543.20,
                "sma_20": 638.25,
                "sma_50": 621.75,
                "volatility": 25.3,
                "beta": 1.15
            },
            
            "WIPRO.NS": {
                "company_name": "Wipro Limited",
                "sector": "Technology",
                "industry": "Information Technology Services",
                "base_price": 445.00,
                "current_price": 448.75,
                "prev_close": 445.00,
                "change": 3.75,
                "change_percent": 0.84,
                "volume": 4567890,
                "avg_volume": 5500000,
                "market_cap": 2456780000000,  # 2.46 trillion
                "pe_ratio": 22.4,
                "pb_ratio": 2.8,
                "dividend_yield": 1.9,
                "roe": 12.8,
                "debt_to_equity": 0.04,
                "52_week_high": 579.85,
                "52_week_low": 385.05,
                "sma_20": 442.90,
                "sma_50": 439.15,
                "volatility": 21.2,
                "beta": 0.89
            },
            
            "BAJFINANCE.NS": {
                "company_name": "Bajaj Finance Limited",
                "sector": "Financial Services",
                "industry": "Non Banking Financial Company (NBFC)",
                "base_price": 6420.00,
                "current_price": 6458.30,
                "prev_close": 6420.00,
                "change": 38.30,
                "change_percent": 0.60,
                "volume": 987654,
                "avg_volume": 1200000,
                "market_cap": 3987650000000,  # 3.99 trillion
                "pe_ratio": 31.2,
                "pb_ratio": 5.8,
                "dividend_yield": 0.4,
                "roe": 18.9,
                "debt_to_equity": 8.5,
                "52_week_high": 8192.20,
                "52_week_low": 6187.80,
                "sma_20": 6398.75,
                "sma_50": 6512.40,
                "volatility": 28.4,
                "beta": 1.23
            }
        }
        
        # User portfolios with updated structure
        self.user_portfolios = {
            "user_1": {
                "cash": 45000.0,
                "stocks": [
                    {
                        "symbol": "RELIANCE.NS",
                        "quantity": 10,
                        "avg_price": 2400.0,
                        "company_name": "Reliance Industries Limited",
                        "sector": "Energy",
                        "added_date": "2024-01-15T10:30:00",
                        "last_updated": "2024-01-15T10:30:00"
                    },
                    {
                        "symbol": "TCS.NS",
                        "quantity": 5,
                        "avg_price": 3600.0,
                        "company_name": "Tata Consultancy Services Limited",
                        "sector": "Technology",
                        "added_date": "2024-02-10T14:20:00",
                        "last_updated": "2024-02-10T14:20:00"
                    },
                    {
                        "symbol": "HDFCBANK.NS",
                        "quantity": 8,
                        "avg_price": 1550.0,
                        "company_name": "HDFC Bank Limited",
                        "sector": "Financial Services",
                        "added_date": "2024-03-05T11:45:00",
                        "last_updated": "2024-03-05T11:45:00"
                    },
                    {
                        "symbol": "INFY.NS",
                        "quantity": 12,
                        "avg_price": 1480.0,
                        "company_name": "Infosys Limited",
                        "sector": "Technology",
                        "added_date": "2024-02-20T09:15:00",
                        "last_updated": "2024-02-20T09:15:00"
                    }
                ]
            },
            
            "user_2": {
                "cash": 35000.0,
                "stocks": [
                    {
                        "symbol": "RELIANCE.NS",
                        "quantity": 8,
                        "avg_price": 2450.0,
                        "company_name": "Reliance Industries Limited",
                        "sector": "Energy",
                        "added_date": "2024-01-20T11:00:00",
                        "last_updated": "2024-01-20T11:00:00"
                    },
                    {
                        "symbol": "TCS.NS",
                        "quantity": 7,
                        "avg_price": 3550.0,
                        "company_name": "Tata Consultancy Services Limited",
                        "sector": "Technology",
                        "added_date": "2024-02-15T14:30:00",
                        "last_updated": "2024-02-15T14:30:00"
                    },
                    {
                        "symbol": "HDFCBANK.NS",
                        "quantity": 6,
                        "avg_price": 1600.0,
                        "company_name": "HDFC Bank Limited",
                        "sector": "Financial Services",
                        "added_date": "2024-03-01T10:45:00",
                        "last_updated": "2024-03-01T10:45:00"
                    },
                    {
                        "symbol": "INFY.NS",
                        "quantity": 15,
                        "avg_price": 1450.0,
                        "company_name": "Infosys Limited",
                        "sector": "Technology",
                        "added_date": "2024-02-25T16:20:00",
                        "last_updated": "2024-02-25T16:20:00"
                    },
                    {
                        "symbol": "ITC.NS",
                        "quantity": 25,
                        "avg_price": 420.0,
                        "company_name": "ITC Limited",
                        "sector": "Consumer Goods",
                        "added_date": "2024-01-25T09:30:00",
                        "last_updated": "2024-01-25T09:30:00"
                    }
                ]
            },
            
            "demo_user": {
                "cash": 50000.0,
                "stocks": [
                    {
                        "symbol": "RELIANCE.NS",
                        "quantity": 6,
                        "avg_price": 2380.0,
                        "company_name": "Reliance Industries Limited",
                        "sector": "Energy",
                        "added_date": "2024-01-10T12:15:00",
                        "last_updated": "2024-01-10T12:15:00"
                    },
                    {
                        "symbol": "TCS.NS",
                        "quantity": 4,
                        "avg_price": 3650.0,
                        "company_name": "Tata Consultancy Services Limited",
                        "sector": "Technology",
                        "added_date": "2024-02-05T15:45:00",
                        "last_updated": "2024-02-05T15:45:00"
                    },
                    {
                        "symbol": "HDFCBANK.NS",
                        "quantity": 10,
                        "avg_price": 1520.0,
                        "company_name": "HDFC Bank Limited",
                        "sector": "Financial Services",
                        "added_date": "2024-03-10T11:20:00",
                        "last_updated": "2024-03-10T11:20:00"
                    },
                    {
                        "symbol": "HINDUNILVR.NS",
                        "quantity": 8,
                        "avg_price": 2650.0,
                        "company_name": "Hindustan Unilever Limited",
                        "sector": "Consumer Goods",
                        "added_date": "2024-02-12T13:30:00",
                        "last_updated": "2024-02-12T13:30:00"
                    }
                ]
            },
            
            "cli_user": {
                "cash": 42000.0,
                "stocks": [
                    {
                        "symbol": "RELIANCE.NS",
                        "quantity": 12,
                        "avg_price": 2350.0,
                        "company_name": "Reliance Industries Limited",
                        "sector": "Energy",
                        "added_date": "2024-01-08T10:00:00",
                        "last_updated": "2024-01-08T10:00:00"
                    },
                    {
                        "symbol": "TCS.NS",
                        "quantity": 6,
                        "avg_price": 3580.0,
                        "company_name": "Tata Consultancy Services Limited",
                        "sector": "Technology",
                        "added_date": "2024-02-18T14:15:00",
                        "last_updated": "2024-02-18T14:15:00"
                    },
                    {
                        "symbol": "INFY.NS",
                        "quantity": 10,
                        "avg_price": 1420.0,
                        "company_name": "Infosys Limited",
                        "sector": "Technology",
                        "added_date": "2024-01-28T09:45:00",
                        "last_updated": "2024-01-28T09:45:00"
                    },
                    {
                        "symbol": "ICICIBANK.NS",
                        "quantity": 14,
                        "avg_price": 950.0,
                        "company_name": "ICICI Bank Limited",
                        "sector": "Financial Services",
                        "added_date": "2024-03-15T16:00:00",
                        "last_updated": "2024-03-15T16:00:00"
                    },
                    {
                        "symbol": "ITC.NS",
                        "quantity": 20,
                        "avg_price": 445.0,
                        "company_name": "ITC Limited",
                        "sector": "Consumer Goods",
                        "added_date": "2024-02-08T11:30:00",
                        "last_updated": "2024-02-08T11:30:00"
                    }
                ]
            },
            
            "test_user": {
                "cash": 38000.0,
                "stocks": [
                    {
                        "symbol": "RELIANCE.NS",
                        "quantity": 9,
                        "avg_price": 2420.0,
                        "company_name": "Reliance Industries Limited",
                        "sector": "Energy",
                        "added_date": "2024-01-12T13:20:00",
                        "last_updated": "2024-01-12T13:20:00"
                    },
                    {
                        "symbol": "HDFCBANK.NS",
                        "quantity": 12,
                        "avg_price": 1580.0,
                        "company_name": "HDFC Bank Limited",
                        "sector": "Financial Services",
                        "added_date": "2024-03-08T10:15:00",
                        "last_updated": "2024-03-08T10:15:00"
                    },
                    {
                        "symbol": "HINDUNILVR.NS",
                        "quantity": 6,
                        "avg_price": 2700.0,
                        "company_name": "Hindustan Unilever Limited",
                        "sector": "Consumer Goods",
                        "added_date": "2024-02-22T15:10:00",
                        "last_updated": "2024-02-22T15:10:00"
                    },
                    {
                        "symbol": "SBIN.NS",
                        "quantity": 30,
                        "avg_price": 620.0,
                        "company_name": "State Bank of India",
                        "sector": "Financial Services",
                        "added_date": "2024-01-30T12:45:00",
                        "last_updated": "2024-01-30T12:45:00"
                    }
                ]
            }
        }
        
        # Hardcoded market data cache (to reduce API calls during development)
        self.market_data_cache = {}
        
        # Configuration defaults
        self.max_portfolio_size = 20
    
    def get_user_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Get hardcoded user portfolio"""
        return self.user_portfolios.get(user_id, {
            "cash": 100000.0,
            "stocks": []
        })
    
    def update_portfolio(self, user_id: str, stocks: List[Dict], cash: float):
        """Update hardcoded portfolio data"""
        if user_id not in self.user_portfolios:
            self.user_portfolios[user_id] = {"cash": cash, "stocks": stocks}
        else:
            self.user_portfolios[user_id]["cash"] = cash
            self.user_portfolios[user_id]["stocks"] = stocks
        
        print(f"Portfolio updated for {user_id}: {len(stocks)} stocks, ₹{cash:.2f} cash")
    
    def save_market_data(self, symbol: str, data: Dict):
        """Save market data to cache (in-memory for development)"""
        self.market_data_cache[symbol] = {
            "data": data,
            "timestamp": datetime.now()
        }
    
    def get_stock_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Get comprehensive stock data - uses hardcoded data first, fallback to Yahoo Finance"""
        
        # Try hardcoded data first
        if symbol in self.stock_database:
            stock_data = self.stock_database[symbol].copy()
            
            # Add calculated fields
            stock_data.update({
                "symbol": symbol,
                "volume_ratio": round(stock_data["volume"] / stock_data["avg_volume"], 2),
                "current_vs_52w_high": round((stock_data["current_price"] / stock_data["52_week_high"]) * 100, 1),
                "currency": "INR",
                "exchange": "NSI",
                "last_updated": datetime.now().isoformat(),
                "data_source": "hardcoded"
            })
            
            return stock_data
        
        # Fallback to Yahoo Finance for other stocks
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
                "last_updated": datetime.now().isoformat(),
                "data_source": "yahoo_finance"
            }
            
            # Save to in-memory cache
            self.save_market_data(symbol, stock_data)
            
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
            portfolio = self.get_user_portfolio(user_id)
            
            # FIXED: Handle empty portfolio case with all required keys
            if not portfolio.get("stocks"):
                return {
                    "total_value": portfolio.get("cash", 100000.0),
                    "cash": portfolio.get("cash", 100000.0),
                    "stocks_value": 0,
                    "stocks": [],
                    "stock_count": 0,  # FIX: Added missing stock_count key
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
                "stock_count": len(stocks_data),  # This was already correct
                
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
            
            portfolio = self.get_user_portfolio(user_id)
            stocks = portfolio.get("stocks", [])
            cash = portfolio.get("cash", 0)
            
            total_cost = quantity * price
            
            # Check sufficient funds
            if cash < total_cost:
                return {
                    "success": False, 
                    "error": f"Insufficient funds. Available: ₹{cash:.2f}, Required: ₹{total_cost:.2f}"
                }
            
            # Check portfolio size limit (default 20 if config not available)
            max_portfolio_size = 20
            if len(stocks) >= max_portfolio_size:
                existing_symbols = [s["symbol"] for s in stocks]
                if symbol not in existing_symbols:
                    return {
                        "success": False,
                        "error": f"Portfolio limit reached ({max_portfolio_size} stocks)"
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
            
            # Update portfolio
            self.update_portfolio(user_id, stocks, new_cash)
            
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
            portfolio = self.get_user_portfolio(user_id)
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
            
            # Update portfolio
            self.update_portfolio(user_id, stocks, new_cash)
            
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
    
    def get_all_users(self) -> List[str]:
        """Get list of all hardcoded user IDs"""
        return list(self.user_portfolios.keys())
    
    def create_new_user(self, user_id: str, starting_cash: float = 100000.0) -> Dict[str, Any]:
        """Create a new user with starting cash"""
        if user_id in self.user_portfolios:
            return {"success": False, "error": "User already exists"}
        
        self.user_portfolios[user_id] = {
            "cash": starting_cash,
            "stocks": []
        }
        
        return {
            "success": True,
            "message": f"Created user {user_id} with ₹{starting_cash:.2f} starting cash",
            "user_id": user_id,
            "starting_cash": starting_cash
        }

# Global financial data processor instance
financial_data = FinancialDataProcessor()

# Example usage and testing
if __name__ == "__main__":
    # Test the system
    processor = financial_data
    
    print("=== Testing Hardcoded Personal Data System ===\n")
    
    # Show all users
    print("Available users:", processor.get_all_users())
    
    # Test portfolio summary
    print("\n=== User 1 Portfolio ===")
    summary = processor.get_portfolio_summary("user_1")
    if "error" not in summary:
        print(f"Total Value: ₹{summary['total_value']:,.2f}")
        print(f"Cash: ₹{summary['cash']:,.2f}")
        print(f"Stocks Value: ₹{summary['stocks_value']:,.2f}")
        print(f"Total Gain/Loss: ₹{summary['total_gain_loss']:,.2f} ({summary['total_gain_loss_percent']:+.2f}%)")
        print("\nHoldings:")
        for stock in summary['stocks']:
            print(f"  {stock['symbol']}: {stock['quantity']} shares @ ₹{stock['current_price']} (₹{stock['gain_loss']:+.2f})")
    
    # Test adding a stock
    print("\n=== Adding Stock to Demo User ===")
    result = processor.add_stock_to_portfolio("demo_user", "WIPRO.NS", 10, 450.0)
    print(result["message"] if result["success"] else result["error"])
    
    # Test market overview
    print("\n=== Market Overview ===")
    market = processor.get_market_overview()
    print(f"Market Sentiment: {market['market_sentiment'].upper()}")
    print(f"Gainers: {market['summary']['gainers']}, Losers: {market['summary']['losers']}")