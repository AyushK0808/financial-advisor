# src/database.py - Simple MongoDB Connection with Error Handling

import pymongo
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import config with error handling
try:
    from src.config import config
except ImportError:
    # Fallback configuration if config import fails
    class Config:
        mongodb_url = "mongodb://localhost:27017/"
        mongodb_db = "financial_advisor"
        save_user_data = True
    config = Config()

class FinancialDB:
    """Simple MongoDB wrapper with error handling"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.users = None
        self.portfolios = None
        self.queries = None
        self.market_data = None
        
        # Initialize with error handling
        try:
            self.client = MongoClient(config.mongodb_url, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.db = self.client[config.mongodb_db]
            
            # Collections
            self.users = self.db.users
            self.portfolios = self.db.portfolios
            self.queries = self.db.queries
            self.market_data = self.db.market_data
            
            print("✅ Connected to MongoDB successfully")
            
        except Exception as e:
            print(f"⚠️  MongoDB connection failed: {str(e)}")
            print("🔄 Running in offline mode - portfolio data will not persist")
            self._init_offline_mode()
    
    def _init_offline_mode(self):
        """Initialize offline mode with in-memory storage"""
        self._offline_portfolios = {}
        self._offline_queries = []
        self._offline_market_data = {}
        print("🔧 Initialized offline storage mode")
    
    def _is_online(self) -> bool:
        """Check if database is available"""
        return self.client is not None and self.db is not None
    
    def save_user_query(self, user_id: str, query: str, response: str, sources: List[str] = None):
        """Save user query with fallback"""
        if not config.save_user_data:
            return
        
        doc = {
            "user_id": user_id,
            "query": query,
            "response": response,
            "sources": sources or [],
            "timestamp": datetime.utcnow()
        }
        
        try:
            if self._is_online():
                self.queries.insert_one(doc)
            else:
                # Offline storage
                self._offline_queries.append(doc)
                # Keep only last 50 queries in memory
                if len(self._offline_queries) > 50:
                    self._offline_queries = self._offline_queries[-50:]
                    
        except Exception as e:
            print(f"⚠️  Error saving query: {str(e)}")
    
    def get_user_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Get user portfolio with fallback"""
        default_portfolio = {
            "user_id": user_id,
            "stocks": [],
            "cash": 100000.0,
            "created_at": datetime.utcnow()
        }
        
        try:
            if self._is_online():
                portfolio = self.portfolios.find_one({"user_id": user_id})
                
                if not portfolio:
                    portfolio = default_portfolio.copy()
                    self.portfolios.insert_one(portfolio)
                
                return portfolio
            else:
                # Offline mode
                if user_id not in self._offline_portfolios:
                    self._offline_portfolios[user_id] = default_portfolio.copy()
                
                return self._offline_portfolios[user_id]
                
        except Exception as e:
            print(f"⚠️  Error getting portfolio: {str(e)}")
            return default_portfolio
    
    def update_portfolio(self, user_id: str, stocks: List[Dict], cash: float):
        """Update portfolio with fallback"""
        update_doc = {
            "stocks": stocks,
            "cash": cash,
            "updated_at": datetime.utcnow()
        }
        
        try:
            if self._is_online():
                self.portfolios.update_one(
                    {"user_id": user_id},
                    {"$set": update_doc},
                    upsert=True
                )
            else:
                # Offline mode
                if user_id not in self._offline_portfolios:
                    self._offline_portfolios[user_id] = {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                
                self._offline_portfolios[user_id].update(update_doc)
                
            print(f"✅ Portfolio updated for {user_id}")
            
        except Exception as e:
            print(f"⚠️  Error updating portfolio: {str(e)}")
    
    def save_market_data(self, symbol: str, data: Dict[str, Any]):
        """Save market data with fallback"""
        doc = {
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.utcnow()
        }
        
        try:
            if self._is_online():
                # Remove old data for this symbol (keep only latest)
                self.market_data.delete_many({"symbol": symbol})
                self.market_data.insert_one(doc)
            else:
                # Offline mode
                self._offline_market_data[symbol] = doc
                
        except Exception as e:
            print(f"⚠️  Error saving market data: {str(e)}")
    
    def get_recent_queries(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent queries with fallback"""
        try:
            if self._is_online():
                cursor = self.queries.find(
                    {"user_id": user_id}
                ).sort("timestamp", -1).limit(limit)
                return list(cursor)
            else:
                # Offline mode
                user_queries = [q for q in self._offline_queries if q["user_id"] == user_id]
                return sorted(user_queries, key=lambda x: x["timestamp"], reverse=True)[:limit]
                
        except Exception as e:
            print(f"⚠️  Error getting recent queries: {str(e)}")
            return []
    
    def get_portfolio_stats(self, user_id: str) -> Dict[str, Any]:
        """Get portfolio statistics"""
        try:
            portfolio = self.get_user_portfolio(user_id)
            
            stats = {
                "total_stocks": len(portfolio.get("stocks", [])),
                "cash_balance": portfolio.get("cash", 0),
                "last_updated": portfolio.get("updated_at", portfolio.get("created_at")),
                "database_mode": "online" if self._is_online() else "offline"
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Error getting portfolio stats: {str(e)}")
            return {"error": str(e)}

# Global instance
db = FinancialDB()