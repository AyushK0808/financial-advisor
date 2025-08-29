# src/rag_system.py - RAG with ChromaDB + SearXNG (Fixed Telemetry)

import os
# Fix ChromaDB telemetry error - disable before importing
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_AUTHN_CREDENTIALS_FILE"] = ""
os.environ["CHROMA_CLIENT_AUTH_PROVIDER"] = ""

import chromadb
from chromadb.utils import embedding_functions
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

from src.config import config
from src.database import db
from src.llm_model import llm

class FinancialRAGSystem:
    """RAG system combining ChromaDB knowledge base with SearXNG web search"""
    
    def __init__(self):
        # Ensure telemetry is disabled
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        # ChromaDB setup with telemetry disabled
        try:
            self.chroma_client = chromadb.PersistentClient(path=config.chromadb_path)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name="financial_knowledge",
                    embedding_function=self.embedding_function
                )
                print("✅ Loaded existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="financial_knowledge",
                    embedding_function=self.embedding_function
                )
                self._populate_knowledge_base()
                print("✅ Created new ChromaDB collection")
                
        except Exception as e:
            print(f"⚠️  ChromaDB initialization warning: {str(e)}")
            print("🔄 Retrying with fallback configuration...")
            
            # Fallback initialization
            try:
                self.chroma_client = chromadb.Client()
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                self.collection = self.chroma_client.create_collection(
                    name="financial_knowledge_fallback",
                    embedding_function=self.embedding_function
                )
                self._populate_knowledge_base()
                print("✅ ChromaDB initialized with fallback configuration")
            except Exception as fallback_error:
                print(f"❌ ChromaDB failed to initialize: {str(fallback_error)}")
                self.collection = None
    
    def _populate_knowledge_base(self):
        """Populate ChromaDB with initial financial knowledge"""
        
        if not self.collection:
            print("⚠️  Skipping knowledge base population - ChromaDB not available")
            return
        
        knowledge_items = [
            {
                "id": "sip_basics",
                "content": "SIP (Systematic Investment Plan) allows regular investment in mutual funds. Benefits include rupee cost averaging, disciplined investing, and compound growth. Minimum investment starts from Rs.500 monthly. Best for long-term goals like retirement planning.",
                "metadata": {"category": "investment", "topic": "sip"}
            },
            {
                "id": "portfolio_diversification", 
                "content": "Diversification reduces risk by spreading investments across asset classes, sectors, and market caps. Recommended allocation for young investors: 70% equity, 20% debt, 10% gold. Rebalance annually when allocation deviates by 5-10%.",
                "metadata": {"category": "portfolio", "topic": "diversification"}
            },
            {
                "id": "tax_saving_investments",
                "content": "Section 80C allows Rs.1.5 lakh deduction through ELSS, PPF, EPF. ELSS has shortest lock-in of 3 years with potential 10-15% returns. PPF offers 15-year commitment with 7-8% tax-free returns. Additional Rs.50,000 via NPS under 80CCD.",
                "metadata": {"category": "tax", "topic": "saving"}
            },
            {
                "id": "stock_fundamental_analysis",
                "content": "Key fundamental analysis ratios: P/E ratio (compare with industry average), Debt-to-Equity (<0.5 preferred), ROE (>15% good), Current Ratio (>1.5 shows liquidity). Always compare with industry peers and analyze 3-5 year trends.",
                "metadata": {"category": "stocks", "topic": "analysis"}
            },
            {
                "id": "etf_vs_mutual_funds",
                "content": "ETFs have lower expense ratios (0.1-0.5%) vs mutual funds (1-2%) but require demat account and real-time trading. Mutual funds offer SIP facility and professional management. For beginners, start with mutual fund SIPs for convenience.",
                "metadata": {"category": "investment", "topic": "etf"}
            },
            {
                "id": "emergency_fund_planning",
                "content": "Emergency fund should be 6-12 months of expenses in liquid investments like savings account or liquid funds. Don't invest emergency funds in equity. Build gradually starting with 1 month expenses. Only use for genuine emergencies.",
                "metadata": {"category": "planning", "topic": "emergency"}
            },
            {
                "id": "risk_management_strategies",
                "content": "Risk management involves diversification, position sizing, stop-loss orders, and proper asset allocation. Never invest all money in single stock or sector. Use systematic approach to reduce emotional decisions and protect capital.",
                "metadata": {"category": "risk", "topic": "management"}
            }
        ]
        
        try:
            # Add to ChromaDB
            documents = [item["content"] for item in knowledge_items]
            metadatas = [item["metadata"] for item in knowledge_items]
            ids = [item["id"] for item in knowledge_items]
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(knowledge_items)} knowledge items to ChromaDB")
            
        except Exception as e:
            print(f"⚠️  Warning adding knowledge items: {str(e)}")
    
    def search_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search local knowledge base"""
        
        if not self.collection:
            print("⚠️  ChromaDB not available, using fallback knowledge")
            return self._fallback_knowledge_search(query)
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "relevance": 1 - results["distances"][0][i],  # Convert distance to relevance
                    "source": "knowledge_base"
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️  ChromaDB search error: {str(e)}")
            return self._fallback_knowledge_search(query)
    
    def _fallback_knowledge_search(self, query: str) -> List[Dict]:
        """Fallback knowledge search when ChromaDB is unavailable"""
        
        # Simple keyword matching fallback
        fallback_knowledge = {
            "sip": {
                "content": "SIP allows regular investment in mutual funds with rupee cost averaging benefits. Start with Rs.500 monthly.",
                "metadata": {"category": "investment", "topic": "sip"}
            },
            "diversif": {
                "content": "Diversify across asset classes: 70% equity, 20% debt, 10% gold for young investors.",
                "metadata": {"category": "portfolio", "topic": "diversification"}
            },
            "tax": {
                "content": "Section 80C: ELSS (3-year lock), PPF (15-year), EPF. ELSS offers highest returns potential.",
                "metadata": {"category": "tax", "topic": "saving"}
            },
            "stock": {
                "content": "Key ratios: P/E ratio, Debt-to-Equity (<0.5), ROE (>15%), Current Ratio (>1.5). Compare with industry peers.",
                "metadata": {"category": "stocks", "topic": "analysis"}
            }
        }
        
        results = []
        query_lower = query.lower()
        
        for keyword, data in fallback_knowledge.items():
            if keyword in query_lower:
                results.append({
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "relevance": 0.8,
                    "source": "fallback_knowledge"
                })
        
        return results[:3]
    
    def search_stock_trends_searxng(self, stock_symbol: str, query: str = None) -> List[Dict]:
        """Search for stock trends using SearXNG"""
        
        try:
            # Clean stock symbol for search
            clean_symbol = stock_symbol.replace(".NS", "").replace(".BO", "")
            
            # Prepare search query
            if query:
                search_query = f"{clean_symbol} stock {query} trends news India"
            else:
                search_query = f"{clean_symbol} stock price trends news analysis India latest"
            
            # Search via SearXNG API
            params = {
                "q": search_query,
                "categories": "news,general",
                "engines": "google,bing,duckduckgo",
                "language": "en",
                "time_range": "month",  # Recent news
                "format": "json"
            }
            
            response = requests.get(
                f"{config.searxng_url}/search",
                params=params,
                timeout=10,
                headers={"User-Agent": "FinancialAdvisor/1.0"}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("results", [])[:config.max_search_results]:
                    # Extract and clean content
                    content = self._extract_content_from_url(item.get("url", ""))
                    if content:
                        results.append({
                            "title": item.get("title", "")[:100],
                            "content": content[:400] + "...",  # Limit content
                            "url": item.get("url", ""),
                            "source": "web_search",
                            "relevance": 0.8,  # Default relevance for web results
                            "published": item.get("publishedDate", "")
                        })
                
                print(f"✅ Found {len(results)} web results for {stock_symbol}")
                return results
            else:
                print(f"⚠️  SearXNG search failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  SearXNG search error: {str(e)}")
        
        return []
    
    def _extract_content_from_url(self, url: str) -> str:
        """Extract text content from URL"""
        
        try:
            response = requests.get(
                url, 
                timeout=5, 
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = " ".join(chunk for chunk in chunks if chunk and len(chunk) > 20)
                
                return clean_text[:800]  # Limit text length
                
        except Exception as e:
            print(f"⚠️  Error extracting content from {url}: {str(e)}")
        
        return ""
    
    def get_comprehensive_context(self, user_query: str, stock_symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive context from both knowledge base and web search"""
        
        context = {
            "query": user_query,
            "knowledge_base_results": [],
            "web_search_results": [],
            "stock_symbol": stock_symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Search knowledge base
        kb_results = self.search_knowledge_base(user_query, n_results=3)
        context["knowledge_base_results"] = kb_results
        
        # Search web for stock-specific information
        if stock_symbol:
            web_results = self.search_stock_trends_searxng(stock_symbol, user_query)
            context["web_search_results"] = web_results
        
        return context
    
    def generate_rag_response(self, user_query: str, stock_symbol: str = None, user_id: str = "anonymous") -> Dict[str, Any]:
        """Generate response using RAG with both local knowledge and web search"""
        
        try:
            # Get comprehensive context
            context = self.get_comprehensive_context(user_query, stock_symbol)
            
            # Prepare context for LLM
            context_text = self._format_context_for_llm(context)
            
            # Generate response using local LLM
            if stock_symbol:
                enhanced_prompt = f"""Based on the financial knowledge and current market information below, provide helpful investment advice about {stock_symbol} for the user's question.

Context Information:
{context_text}

User Question: {user_query}

Please provide a comprehensive, practical answer that combines general financial principles with current market insights. Be specific and actionable, but always mention that this is for educational purposes and not personalized financial advice."""
            else:
                enhanced_prompt = f"""Based on the financial knowledge below, provide helpful investment advice for the user's question.

Context Information:
{context_text}

User Question: {user_query}

Please provide a comprehensive, practical answer based on sound financial principles. Be specific and actionable."""

            # Generate response
            try:
                llm_response = llm.generate_response(enhanced_prompt, max_length=200)
            except Exception as llm_error:
                print(f"⚠️  LLM generation error: {str(llm_error)}")
                # Fallback response if LLM fails
                llm_response = self._generate_fallback_response(user_query, context)
            
            # Compile sources
            sources = []
            sources.extend([f"Knowledge: {r['metadata']['topic']}" for r in context["knowledge_base_results"]])
            sources.extend([f"News: {r['title'][:30]}..." for r in context["web_search_results"]])
            
            response = {
                "answer": llm_response,
                "context_used": len(context["knowledge_base_results"]) + len(context["web_search_results"]),
                "sources": sources[:5],  # Limit sources shown
                "stock_symbol": stock_symbol,
                "confidence": self._calculate_confidence(context),
                "timestamp": datetime.now().isoformat(),
                "kb_results": len(context["knowledge_base_results"]),
                "web_results": len(context["web_search_results"])
            }
            
            # Save to database
            if config.save_user_data:
                try:
                    db.save_user_query(user_id, user_query, llm_response, sources)
                except Exception as db_error:
                    print(f"⚠️  Database save error: {str(db_error)}")
            
            return response
            
        except Exception as e:
            print(f"❌ Error in RAG response generation: {str(e)}")
            return {
                "answer": "I apologize, but I'm having trouble processing your question right now. This could be due to connectivity issues or system maintenance. Please try again in a moment.",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.1
            }
    
    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format context for LLM input"""
        
        formatted_parts = []
        
        # Add knowledge base results
        if context["knowledge_base_results"]:
            formatted_parts.append("Financial Knowledge:")
            for i, result in enumerate(context["knowledge_base_results"][:3]):
                formatted_parts.append(f"{i+1}. {result['content']}")
        
        # Add web search results
        if context["web_search_results"]:
            formatted_parts.append("\nCurrent Market Information:")
            for i, result in enumerate(context["web_search_results"][:2]):
                formatted_parts.append(f"{i+1}. {result['title']}: {result['content'][:150]}...")
        
        return "\n".join(formatted_parts)
    
    def _generate_fallback_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate fallback response when LLM is not available"""
        
        # Use knowledge base content
        if context["knowledge_base_results"]:
            top_result = context["knowledge_base_results"][0]
            topic = top_result['metadata']['topic']
            content = top_result['content']
            
            return f"Based on our knowledge about {topic}: {content[:200]}... For more personalized advice, consider consulting with a financial advisor."
        
        # Use web search content
        elif context["web_search_results"]:
            top_result = context["web_search_results"][0]
            return f"Based on recent market information: {top_result['content'][:200]}... Please verify this information with official sources."
        
        # Generic response
        else:
            return "I understand you're asking about financial topics. While I don't have specific information readily available, I recommend consulting reliable financial resources or speaking with a qualified financial advisor for personalized guidance."
    
    def _calculate_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence score based on available context"""
        
        kb_count = len(context["knowledge_base_results"])
        web_count = len(context["web_search_results"])
        
        # Base confidence on availability and relevance of context
        confidence = 0.3  # Base confidence
        
        if kb_count >= 2:
            confidence += 0.3
        elif kb_count >= 1:
            confidence += 0.2
            
        if web_count >= 2:
            confidence += 0.3
        elif web_count >= 1:
            confidence += 0.2
        
        # Bonus for having both types of context
        if kb_count > 0 and web_count > 0:
            confidence += 0.1
            
        return min(confidence, 0.95)  # Cap at 95%
    
    def add_knowledge(self, content: str, category: str, topic: str) -> bool:
        """Add new knowledge to the knowledge base"""
        
        if not self.collection:
            print("⚠️  Cannot add knowledge - ChromaDB not available")
            return False
        
        try:
            doc_id = f"{category}_{topic}_{int(datetime.now().timestamp())}"
            
            self.collection.add(
                documents=[content],
                metadatas=[{"category": category, "topic": topic, "added_at": datetime.now().isoformat()}],
                ids=[doc_id]
            )
            
            print(f"✅ Added knowledge: {doc_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding knowledge: {str(e)}")
            return False
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        if not self.collection:
            return {"error": "ChromaDB not available"}
        
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "last_updated": datetime.now().isoformat(),
                "collection_name": "financial_knowledge"
            }
        except Exception as e:
            print(f"❌ Error getting stats: {str(e)}")
            return {"error": str(e)}

# Global RAG system instance
rag_system = FinancialRAGSystem()