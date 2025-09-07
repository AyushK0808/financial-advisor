# src/rag_system.py - Enhanced RAG with Improved Generalization

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
import numpy as np
from collections import defaultdict

from src.config import config
from src.database import db

class FinancialRAGSystem:
    """Enhanced RAG system with improved generation capabilities and query transformation"""

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
                self._populate_enhanced_knowledge_base()
                print("✅ Created new ChromaDB collection")

        except Exception as e:
            print(f"⚠️ ChromaDB initialization warning: {str(e)}")
            print("🔄 Retrying with fallback configuration...")

            # Fallback initialization
            try:
                self.chroma_client = chromadb.Client()
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                self.collection = self.chroma_client.create_collection(
                    name="financial_knowledge_fallback",
                    embedding_function=self.embedding_function
                )
                self._populate_enhanced_knowledge_base()
                print("✅ ChromaDB initialized with fallback configuration")
            except Exception as fallback_error:
                print(f"❌ ChromaDB failed to initialize: {str(fallback_error)}")
                self.collection = None

        # Initialize query transformation patterns
        self.query_transformations = {
            "facts": "What are the key facts about {query}?",
            "analysis": "What is the financial analysis of {query}?", 
            "implications": "What are the investment implications of {query}?",
            "trends": "What are the current market trends related to {query}?",
            "expert_opinion": "What do financial experts say about {query}?",
            "risks": "What are the risks associated with {query}?",
            "opportunities": "What opportunities exist with {query}?",
            "comparison": "How does {query} compare with alternatives?",
            "strategy": "What investment strategy should I consider for {query}?",
            "timing": "When is the right time to invest in {query}?"
        }

    def _populate_enhanced_knowledge_base(self):
        """Populate ChromaDB with enhanced financial knowledge for better synthesis"""

        if not self.collection:
            print("⚠️ Skipping knowledge base population - ChromaDB not available")
            return

        # Enhanced knowledge items with more detailed, synthesis-friendly content
        enhanced_knowledge_items = [
            {
                "id": "sip_comprehensive_guide",
                "content": """SIP (Systematic Investment Plan) is a disciplined investment method that allows investors to invest a fixed amount regularly in mutual funds. The key benefits include rupee cost averaging, which helps average out market volatility over time, disciplined investing habits that prevent emotional decision-making, and the power of compounding that accelerates wealth creation. SIPs work particularly well for equity mutual funds and can start with as little as Rs.500 monthly. For young investors, aggressive growth funds through SIP can potentially deliver 12-15% annual returns over 10-15 years. The ideal approach is to gradually increase SIP amounts (step-up SIPs) by 10-15% annually as income grows. SIPs should be viewed as long-term investments with a minimum commitment of 5-10 years to realize full potential.""",
                "metadata": {"category": "investment_strategies", "topic": "sip", "complexity": "detailed", "target_audience": "all_investors"}
            },
            {
                "id": "portfolio_diversification_strategy",
                "content": """Portfolio diversification is the cornerstone of risk management that involves spreading investments across different asset classes, sectors, market capitalizations, and geographic regions. A well-diversified Indian portfolio should typically allocate 60-70% to equities (split between large-cap 40%, mid-cap 20%, small-cap 10%), 20-25% to debt instruments (including PPF, fixed deposits, government bonds), 5-10% to gold (physical or gold ETFs), and 5% to international equity exposure. Within equity allocation, sector diversification is crucial - avoid concentrating more than 15-20% in any single sector. The portfolio should be rebalanced annually or when any asset class deviates more than 10% from target allocation. Young investors can take higher equity exposure (up to 80%), while those nearing retirement should gradually shift to debt (up to 60%). Regular review and rebalancing ensure the portfolio stays aligned with financial goals and risk tolerance.""",
                "metadata": {"category": "portfolio_management", "topic": "diversification", "complexity": "detailed", "target_audience": "all_investors"}
            },
            {
                "id": "tax_optimization_comprehensive",
                "content": """Tax-efficient investing in India involves utilizing various sections of the Income Tax Act strategically. Section 80C allows Rs.1.5 lakh annual deduction through instruments like ELSS mutual funds (shortest 3-year lock-in with potential 12-15% returns), PPF (15-year commitment offering 7-8% tax-free returns), and EPF contributions. ELSS funds are particularly attractive for young investors due to market-linked returns and shorter lock-in period. Additional deductions include Rs.50,000 under Section 80CCD for NPS (National Pension System), Rs.25,000 for health insurance under 80D, and Rs.2 lakh for home loan interest under 24(b). For senior citizens, NPS provides additional tax benefits at withdrawal. The key strategy is to start early, diversify across different tax-saving instruments, and align investments with long-term financial goals rather than just tax savings. Systematic tax planning can significantly boost post-tax returns over time.""",
                "metadata": {"category": "tax_planning", "topic": "optimization", "complexity": "detailed", "target_audience": "tax_conscious_investors"}
            },
            {
                "id": "fundamental_analysis_framework", 
                "content": """Fundamental analysis involves evaluating a company's intrinsic value through financial metrics, business model analysis, and industry dynamics. Key financial ratios include P/E ratio (compare with industry average; below 15 generally attractive, above 30 expensive), P/B ratio (below 2-3 for most sectors), Debt-to-Equity ratio (below 0.5 preferred for most industries except capital-intensive ones), ROE (above 15% indicates efficient capital utilization), and ROA (above 8-10% shows good asset utilization). Qualitative factors include management quality, competitive advantages (moats), industry growth prospects, regulatory environment, and corporate governance standards. Revenue growth consistency over 5-7 years, profit margin trends, and cash flow generation capability are crucial indicators. Always compare metrics with industry peers and analyze business cycles. For Indian markets, consider monsoon impact on rural demand, government policies, and global commodity price fluctuations. A comprehensive analysis combines quantitative metrics with qualitative assessment to make informed investment decisions.""",
                "metadata": {"category": "stock_analysis", "topic": "fundamental_analysis", "complexity": "advanced", "target_audience": "equity_investors"}
            },
            {
                "id": "market_timing_and_psychology",
                "content": """Market timing and investor psychology play crucial roles in investment success. Markets are driven by fear and greed cycles - during market crashes, fear dominates leading to overselling, while during bull runs, greed creates overvaluation. Successful investing requires contrarian thinking - being greedy when others are fearful and fearful when others are greedy. Dollar-cost averaging through SIPs helps neutralize timing risks. Key behavioral biases to avoid include recency bias (giving too much weight to recent events), confirmation bias (seeking information that confirms existing beliefs), and herd mentality (following the crowd). Market corrections of 10-20% are normal and healthy - they provide buying opportunities for long-term investors. The Indian market has historically recovered from all major crashes within 2-3 years. Instead of timing the market, focus on time in the market. Regular investment during both up and down cycles has historically delivered superior returns compared to lump-sum timing attempts.""",
                "metadata": {"category": "market_psychology", "topic": "timing_and_behavior", "complexity": "intermediate", "target_audience": "behavioral_aware_investors"}
            },
            {
                "id": "retirement_planning_strategy",
                "content": """Retirement planning in India requires a systematic approach considering inflation, increasing life expectancy, and healthcare costs. Start with calculating post-retirement monthly expenses (typically 70-80% of current expenses) and factor in 6-7% annual inflation. A 30-year-old should ideally save 15-20% of income for retirement. The investment mix should be aggressive initially (80% equity, 20% debt) and gradually become conservative (40% equity, 60% debt) as retirement approaches. Key instruments include EPF (mandatory for salaried), NPS (additional tax benefits and professional management), equity mutual funds through SIP for wealth creation, and PPF for stable, tax-free returns. Post-retirement, create a withdrawal strategy that provides steady monthly income - consider systematic withdrawal plans (SWP) from mutual funds, annuity plans for guaranteed income, and maintaining 2-3 years of expenses in liquid funds for emergencies. Healthcare inflation runs at 12-15% annually, so adequate health insurance and dedicated healthcare corpus are essential components of retirement planning.""",
                "metadata": {"category": "retirement_planning", "topic": "comprehensive_strategy", "complexity": "detailed", "target_audience": "long_term_investors"}
            },
            {
                "id": "risk_management_comprehensive",
                "content": """Comprehensive risk management in investing involves identifying, measuring, and mitigating various types of risks. Market risk (systematic risk affecting entire market) can be managed through asset allocation and regular rebalancing. Company-specific risk (unsystematic risk) is addressed through diversification across sectors and stocks. Liquidity risk is managed by maintaining adequate emergency funds and investing in liquid instruments. Inflation risk erodes purchasing power - equity investments and inflation-indexed bonds help combat this. Currency risk affects international investments - hedge through domestic equity exposure or currency-hedged funds. Concentration risk arises from over-investing in single stocks/sectors - maintain maximum 5-10% allocation to any single stock and 15-20% to any sector. Interest rate risk impacts debt investments - use duration matching with investment horizon. Regulatory risk in sectors like telecom, banking, and pharma requires staying updated with policy changes. Create a risk tolerance questionnaire considering age, income stability, financial goals, and emotional capacity for volatility. Regular portfolio stress testing helps identify vulnerabilities during market downturns.""",
                "metadata": {"category": "risk_management", "topic": "comprehensive_framework", "complexity": "advanced", "target_audience": "sophisticated_investors"}
            },
            {
                "id": "mutual_fund_selection_criteria",
                "content": """Selecting the right mutual funds requires evaluating multiple parameters beyond just past returns. Fund manager track record and experience, especially during market downturns, is crucial - look for consistent performers across market cycles. Expense ratio significantly impacts long-term returns - prefer funds with expense ratios below 1.5% for equity funds and below 1% for debt funds. Fund size matters - very small funds (below Rs.100 crore) may have liquidity issues, while very large funds (above Rs.10,000 crore) may face deployment challenges in mid/small cap strategies. Portfolio concentration and overlap with existing holdings should be analyzed to avoid over-diversification. Risk-adjusted returns measured by Sharpe ratio and Alpha provide better performance metrics than absolute returns. Fund house stability, research capabilities, and adherence to investment mandate are important factors. For equity funds, consider the fund's performance during bear markets - funds that fall less during downturns often deliver superior risk-adjusted returns. Regular vs direct plans - direct plans have lower expense ratios and are suitable for informed investors. Exit loads and tax implications should be factored into the selection process.""",
                "metadata": {"category": "mutual_funds", "topic": "selection_criteria", "complexity": "intermediate", "target_audience": "fund_investors"}
            },
            {
                "id": "emergency_fund_planning",
                "content": """Emergency fund is the foundation of financial planning that provides financial security during unexpected situations like job loss, medical emergencies, or major repairs. The fund should cover 6-12 months of essential expenses (rent, groceries, loan EMIs, insurance premiums, utilities). For dual-income families, 6 months may suffice, while single-income households need 12 months coverage. Calculate monthly expenses conservatively and build the fund gradually - start with 1 month's expenses, then build to 3, 6, and finally the target amount. Keep the fund in highly liquid instruments like savings accounts, liquid mutual funds, or short-term fixed deposits. Avoid equity investments for emergency funds as they may be down when you need them most. Liquid funds offer better returns than savings accounts with same-day or T+1 liquidity. Maintain the fund separately from other investments to avoid the temptation of using it for non-emergencies. Replenish immediately after any withdrawal. Consider keeping 1-2 months expenses in savings account for immediate access and rest in liquid funds for better returns. Review and update the fund amount annually as expenses increase with inflation and lifestyle changes.""",
                "metadata": {"category": "financial_planning", "topic": "emergency_preparedness", "complexity": "basic", "target_audience": "all_investors"}
            },
            {
                "id": "goal_based_investing",
                "content": """Goal-based investing involves aligning investments with specific financial objectives, timelines, and risk profiles. Short-term goals (1-3 years) like vacation, car purchase, or home down payment require low-risk investments such as fixed deposits, liquid funds, or short-term debt funds to preserve capital. Medium-term goals (3-7 years) like child's education or home purchase can accommodate moderate risk through hybrid funds or conservative asset allocation. Long-term goals (7+ years) like retirement or child's higher education benefit from equity-heavy portfolios that can ride out market volatility. Each goal should have separate investment accounts or funds to avoid mixing purposes. Calculate the future value needed for each goal considering inflation - education costs inflate at 8-10% annually, healthcare at 12-15%. Start with high-priority goals like retirement and emergency funds, then add other goals systematically. Use SIP calculators to determine monthly investment required for each goal. Review progress annually and adjust contributions based on performance and changing circumstances. As goals approach, gradually shift to lower-risk investments to protect accumulated wealth. Goal-based approach provides clarity, discipline, and better emotional management during market volatility.""",
                "metadata": {"category": "investment_planning", "topic": "goal_alignment", "complexity": "intermediate", "target_audience": "systematic_investors"}
            }
        ]

        try:
            # Add to ChromaDB
            documents = [item["content"] for item in enhanced_knowledge_items]
            metadatas = [item["metadata"] for item in enhanced_knowledge_items]
            ids = [item["id"] for item in enhanced_knowledge_items]

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            print(f"✅ Added {len(enhanced_knowledge_items)} enhanced knowledge items to ChromaDB")

        except Exception as e:
            print(f"⚠️ Warning adding knowledge items: {str(e)}")

    def transform_query(self, original_query: str) -> List[str]:
        """Transform single query into multiple perspectives for comprehensive retrieval"""

        transformed_queries = []

        # Add the original query
        transformed_queries.append(original_query)

        # Generate perspective-based queries
        for perspective, template in self.query_transformations.items():
            try:
                transformed_query = template.format(query=original_query)
                if transformed_query != original_query:  # Avoid duplicates
                    transformed_queries.append(transformed_query)
            except:
                continue  # Skip if formatting fails

        # Add keyword-enriched variations
        financial_keywords = ["investment", "financial", "market", "strategy", "risk", "return", "portfolio"]
        for keyword in financial_keywords:
            if keyword.lower() not in original_query.lower():
                enriched_query = f"{original_query} {keyword}"
                transformed_queries.append(enriched_query)

        # Limit to avoid too many queries
        return transformed_queries[:8]

    def enhanced_knowledge_retrieval(self, query: str, n_results: int = 5) -> List[Dict]:
        """Enhanced retrieval with query transformation and relevance scoring"""

        if not self.collection:
            print("⚠️ ChromaDB not available, using fallback knowledge")
            return self._fallback_knowledge_search(query)

        try:
            # Transform query into multiple perspectives
            transformed_queries = self.transform_query(query)

            all_results = []
            seen_ids = set()

            # Retrieve for each transformed query
            for t_query in transformed_queries:
                try:
                    results = self.collection.query(
                        query_texts=[t_query],
                        n_results=min(3, n_results),
                        include=["documents", "metadatas", "distances"]
                    )

                    # Process results
                    for i in range(len(results["documents"][0])):
                        doc_id = results["metadatas"][0][i].get("id", f"doc_{i}")

                        if doc_id not in seen_ids:  # Avoid duplicates
                            relevance_score = 1 - results["distances"][0][i]  # Convert distance to relevance

                            all_results.append({
                                "content": results["documents"][0][i],
                                "metadata": results["metadatas"][0][i],
                                "relevance": relevance_score,
                                "source": "enhanced_knowledge_base",
                                "query_match": t_query
                            })
                            seen_ids.add(doc_id)

                except Exception as e:
                    print(f"⚠️ Query transformation error for '{t_query}': {str(e)}")
                    continue

            # Sort by relevance and return top results
            all_results.sort(key=lambda x: x["relevance"], reverse=True)
            return all_results[:n_results]

        except Exception as e:
            print(f"⚠️ Enhanced retrieval error: {str(e)}")
            return self._fallback_knowledge_search(query)

    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Dict]) -> float:
        """Evaluate the quality of retrieved documents for the given query"""

        if not retrieved_docs:
            return 0.0

        # Simple relevance scoring based on content similarity and metadata
        total_score = 0.0

        for doc in retrieved_docs:
            score = 0.0

            # Base relevance score
            score += doc.get("relevance", 0.0) * 0.6

            # Bonus for detailed content
            content_length = len(doc.get("content", ""))
            if content_length > 500:  # Detailed explanations are better for synthesis
                score += 0.2
            elif content_length > 200:
                score += 0.1

            # Bonus for comprehensive metadata
            metadata = doc.get("metadata", {})
            if metadata.get("complexity") == "detailed":
                score += 0.1
            if metadata.get("target_audience") in ["all_investors", "sophisticated_investors"]:
                score += 0.1

            total_score += score

        # Average score across all documents
        avg_score = total_score / len(retrieved_docs)
        return min(avg_score, 1.0)  # Cap at 1.0

    def corrective_retrieval(self, query: str, initial_results: List[Dict], quality_threshold: float = 0.6) -> List[Dict]:
        """Implement corrective retrieval when initial quality is low"""

        quality_score = self.evaluate_retrieval_quality(query, initial_results)

        if quality_score >= quality_threshold:
            return initial_results

        print(f"🔄 Initial retrieval quality ({quality_score:.2f}) below threshold ({quality_threshold}). Trying enhanced strategies...")

        # Strategy 1: Broader query expansion
        expanded_queries = [
            f"comprehensive guide to {query}",
            f"detailed analysis of {query}",
            f"investment strategy for {query}",
            f"financial planning {query}",
            f"expert advice on {query}"
        ]

        enhanced_results = []
        for expanded_query in expanded_queries:
            try:
                results = self.collection.query(
                    query_texts=[expanded_query],
                    n_results=2,
                    include=["documents", "metadatas", "distances"]
                )

                for i in range(len(results["documents"][0])):
                    enhanced_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "relevance": 1 - results["distances"][0][i],
                        "source": "corrective_retrieval",
                        "strategy": "query_expansion"
                    })
            except:
                continue

        # Combine with original results and re-rank
        combined_results = initial_results + enhanced_results

        # Remove duplicates and sort by relevance
        seen_content = set()
        unique_results = []
        for result in combined_results:
            content_hash = hash(result["content"][:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                unique_results.append(result)
                seen_content.add(content_hash)

        unique_results.sort(key=lambda x: x["relevance"], reverse=True)

        return unique_results[:5]

    def generate_enhanced_prompt(self, user_query: str, context_results: List[Dict], stock_symbol: str = None) -> str:
        """Generate enhanced prompt with explicit synthesis and generation instructions"""

        # Format context from multiple sources
        context_sections = []

        # Organize context by complexity and relevance
        high_relevance_docs = [doc for doc in context_results if doc.get("relevance", 0) > 0.7]
        medium_relevance_docs = [doc for doc in context_results if 0.4 <= doc.get("relevance", 0) <= 0.7]

        if high_relevance_docs:
            context_sections.append("## High-Relevance Financial Knowledge:")
            for i, doc in enumerate(high_relevance_docs[:3]):
                context_sections.append(f"### Source {i+1}:")
                context_sections.append(doc["content"])
                context_sections.append("")

        if medium_relevance_docs:
            context_sections.append("## Additional Context:")
            for i, doc in enumerate(medium_relevance_docs[:2]):
                context_sections.append(f"### Reference {i+1}:")
                context_sections.append(doc["content"])
                context_sections.append("")

        formatted_context = "".join(context_sections)

        # Create comprehensive generation prompt
        if stock_symbol:
            enhanced_prompt = f"""You are an expert financial advisor AI with deep knowledge of Indian markets and investment strategies. Your task is to provide comprehensive, personalized financial advice by synthesizing information from multiple sources.

## Context Information:
{formatted_context}

## User Question: 
{user_query}

## Stock Context: 
{stock_symbol}

## Instructions for Response:
You must provide a detailed, analytical response that:

1. **SYNTHESIZE INFORMATION**: Combine insights from multiple knowledge sources above - don't just repeat one source
2. **PROVIDE ORIGINAL ANALYSIS**: Add your own financial expertise and reasoning beyond what's in the context
3. **PERSONALIZE ADVICE**: Adapt your recommendations based on the specific question and stock context
4. **USE STRUCTURED REASONING**: Explain your thought process step-by-step
5. **ADDRESS NUANCES**: Consider different scenarios, risk levels, and investment horizons
6. **PRACTICAL GUIDANCE**: Provide actionable steps and specific recommendations
7. **ACKNOWLEDGE LIMITATIONS**: If context doesn't fully address the question, state what additional information would be helpful

## Response Framework:
- Start with a direct answer to the core question
- Explain the reasoning using context and your expertise
- Provide specific, actionable recommendations
- Address potential risks and considerations
- Suggest next steps or additional analysis needed

## Important Notes:
- This is for educational purposes and not personalized financial advice
- Always recommend consulting with qualified financial advisors for individual situations
- Consider current market conditions and regulatory environment
- Focus on long-term wealth building principles

## Your Expert Response:"""
        else:
            enhanced_prompt = f"""You are an expert financial advisor AI specializing in Indian markets and investment strategies. Your role is to synthesize financial knowledge and provide comprehensive, educational guidance.

## Knowledge Base Context:
{formatted_context}

## User Question:
{user_query}

## Response Guidelines:
Your response must demonstrate sophisticated financial analysis by:

1. **COMPREHENSIVE SYNTHESIS**: Integrate information from multiple sources above to create new insights
2. **EXPERT ANALYSIS**: Apply advanced financial principles and market understanding
3. **CONTEXTUAL ADAPTATION**: Tailor advice to the specific question and implied investor profile
4. **STRUCTURED REASONING**: Use clear, logical flow from analysis to recommendations
5. **PRACTICAL APPLICATION**: Translate complex concepts into actionable strategies
6. **RISK ASSESSMENT**: Address potential downsides and mitigation strategies
7. **CURRENT RELEVANCE**: Consider present market conditions and regulatory environment

## Response Structure:
- **Direct Answer**: Start with clear response to the main question
- **Analysis**: Detailed reasoning combining multiple knowledge sources
- **Strategic Recommendations**: Specific, prioritized action items
- **Risk Considerations**: Potential challenges and how to address them
- **Implementation Guide**: Step-by-step approach with timelines
- **Additional Considerations**: Related factors to keep in mind

## Quality Standards:
- Demonstrate understanding beyond simple retrieval
- Show connection between different financial concepts
- Provide insights that add value to the raw information
- Maintain educational focus while being practically useful

## Educational Disclaimer:
This analysis is for educational purposes. Individual circumstances vary significantly, and personalized financial advice from qualified professionals is always recommended for specific investment decisions.

## Your Comprehensive Analysis:"""

        return enhanced_prompt

    def search_stock_trends_with_quality_check(self, stock_symbol: str, query: str = None) -> List[Dict]:
        """Enhanced web search with quality evaluation"""

        web_results = self.search_stock_trends_searxng(stock_symbol, query)

        # Filter and enhance web results
        enhanced_web_results = []
        for result in web_results:
            # Quality scoring for web results
            quality_score = 0.5  # Base score

            # Title relevance
            if stock_symbol.replace(".NS", "").replace(".BO", "") in result.get("title", ""):
                quality_score += 0.2

            # Content length and quality
            content_length = len(result.get("content", ""))
            if content_length > 300:
                quality_score += 0.2
            elif content_length > 150:
                quality_score += 0.1

            # Recent publication bonus
            published_date = result.get("published", "")
            if "2024" in published_date or "2025" in published_date:
                quality_score += 0.1

            result["quality_score"] = quality_score
            result["relevance"] = quality_score  # For consistency

            if quality_score > 0.5:  # Only include decent quality results
                enhanced_web_results.append(result)

        return enhanced_web_results[:3]  # Limit to top 3 quality results

    def search_stock_trends_searxng(self, stock_symbol: str, query: str = None) -> List[Dict]:
        """Original web search method (unchanged for compatibility)"""

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
                print(f"⚠️ SearXNG search failed: HTTP {response.status_code}")

        except Exception as e:
            print(f"⚠️ SearXNG search error: {str(e)}")

        return []

    def _extract_content_from_url(self, url: str) -> str:
        """Extract text content from URL (unchanged for compatibility)"""

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
            print(f"⚠️ Error extracting content from {url}: {str(e)}")

        return ""

    def get_comprehensive_context(self, user_query: str, stock_symbol: str = None) -> Dict[str, Any]:
        """Get enhanced context with quality evaluation and corrective retrieval"""

        context = {
            "query": user_query,
            "knowledge_base_results": [],
            "web_search_results": [],
            "stock_symbol": stock_symbol,
            "timestamp": datetime.now().isoformat(),
            "retrieval_strategy": "enhanced",
            "quality_score": 0.0
        }

        # Enhanced knowledge base search
        kb_results = self.enhanced_knowledge_retrieval(user_query, n_results=5)

        # Apply corrective retrieval if needed
        kb_results = self.corrective_retrieval(user_query, kb_results)

        context["knowledge_base_results"] = kb_results
        context["kb_quality_score"] = self.evaluate_retrieval_quality(user_query, kb_results)

        # Enhanced web search for stock-specific information
        if stock_symbol:
            web_results = self.search_stock_trends_with_quality_check(stock_symbol, user_query)
            context["web_search_results"] = web_results

        # Overall quality assessment
        kb_score = context["kb_quality_score"]
        web_score = 0.7 if context["web_search_results"] else 0.0  # Assume good quality if we have web results

        context["quality_score"] = (kb_score * 0.7) + (web_score * 0.3)  # Weight KB more than web

        return context

    def generate_rag_response(self, user_query: str, stock_symbol: str = None, user_id: str = "anonymous") -> Dict[str, Any]:
        """Generate enhanced RAG response with improved synthesis and generation"""

        try:
            # Get comprehensive context with quality evaluation
            context = self.get_comprehensive_context(user_query, stock_symbol)

            # Generate enhanced prompt for better synthesis
            enhanced_prompt = self.generate_enhanced_prompt(
                user_query, 
                context["knowledge_base_results"] + context["web_search_results"],
                stock_symbol
            )

            # Import LLM here to avoid circular imports
            from src.llm_model import llm

            # Generate response with longer max length for comprehensive answers
            try:
                llm_response = llm.generate_response(enhanced_prompt, max_length=300, debug=False)

                # Post-process response to ensure it's comprehensive
                if len(llm_response.strip()) < 100:  # Too short, likely just retrieval
                    print("⚠️ Response too short, regenerating with fallback prompt...")
                    fallback_prompt = self._create_fallback_synthesis_prompt(user_query, context["knowledge_base_results"])
                    llm_response = llm.generate_response(fallback_prompt, max_length=250)

            except Exception as llm_error:
                print(f"⚠️ LLM generation error: {str(llm_error)}")
                # Enhanced fallback response with synthesis
                llm_response = self._generate_enhanced_fallback_response(user_query, context)

            # Compile comprehensive sources
            sources = []
            for result in context["knowledge_base_results"][:3]:
                topic = result["metadata"].get("topic", "financial_knowledge")
                sources.append(f"Knowledge: {topic.replace('_', ' ').title()}")

            for result in context["web_search_results"][:2]:
                title = result["title"][:40] + "..." if len(result["title"]) > 40 else result["title"]
                sources.append(f"News: {title}")

            # Enhanced response compilation
            response = {
                "answer": llm_response,
                "context_used": len(context["knowledge_base_results"]) + len(context["web_search_results"]),
                "sources": sources,
                "stock_symbol": stock_symbol,
                "confidence": self._calculate_enhanced_confidence(context, llm_response),
                "timestamp": datetime.now().isoformat(),
                "kb_results": len(context["knowledge_base_results"]),
                "web_results": len(context["web_search_results"]),
                "quality_score": context["quality_score"],
                "retrieval_strategy": context["retrieval_strategy"],
                "synthesis_quality": self._evaluate_synthesis_quality(llm_response, context)
            }

            # Save enhanced interaction to database
            if config.save_user_data:
                try:
                    enhanced_query_data = {
                        "user_query": user_query,
                        "response": llm_response,
                        "sources": sources,
                        "quality_metrics": {
                            "confidence": response["confidence"],
                            "quality_score": response["quality_score"],
                            "synthesis_quality": response["synthesis_quality"]
                        }
                    }
                    db.save_user_query(user_id, user_query, llm_response, sources)
                except Exception as db_error:
                    print(f"⚠️ Database save error: {str(db_error)}")

            return response

        except Exception as e:
            print(f"❌ Error in enhanced RAG response generation: {str(e)}")
            return {
                "answer": "I apologize, but I'm experiencing technical difficulties processing your question right now. This could be due to system maintenance or connectivity issues. Please try rephrasing your question or try again in a moment.",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.1,
                "quality_score": 0.0,
                "synthesis_quality": 0.0
            }

    def _create_fallback_synthesis_prompt(self, query: str, kb_results: List[Dict]) -> str:
        """Create fallback prompt focused on synthesis when main prompt fails"""

        if not kb_results:
            return f"As a financial advisor, provide comprehensive guidance on: {query}. Include practical strategies, risk considerations, and actionable steps."

        top_content = kb_results[0]["content"][:500] + "..." if len(kb_results[0]["content"]) > 500 else kb_results[0]["content"]

        return f"""Based on this financial knowledge: "{top_content}"

Question: {query}

Provide a comprehensive response that:
1. Goes beyond the given information by adding your own financial expertise
2. Gives specific, actionable recommendations
3. Explains the reasoning step-by-step
4. Addresses potential risks and considerations
5. Suggests practical implementation steps

Your detailed financial advice:"""

    def _generate_enhanced_fallback_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate enhanced fallback response when LLM is not available"""

        # Use knowledge base content with synthesis
        if context["knowledge_base_results"]:
            top_results = context["knowledge_base_results"][:2]

            synthesized_content = "Based on comprehensive financial analysis: "

            for result in top_results:
                content = result["content"]
                topic = result["metadata"].get("topic", "financial planning")

                # Extract key points for synthesis
                key_sentences = content.split(". ")[:3]  # Take first 3 sentences
                synthesized_content += f"Regarding {topic.replace('_', ' ')}, {'. '.join(key_sentences)}. "

            synthesized_content += f"For your specific question about {query}, I recommend taking a systematic approach that balances risk and returns while aligning with your financial goals. "
            synthesized_content += "Please consult with a qualified financial advisor for personalized advice tailored to your individual circumstances."

            return synthesized_content

        # Use web search content with analysis
        elif context["web_search_results"]:
            top_result = context["web_search_results"][0]
            return f"Based on current market information: {top_result['content'][:300]}... This suggests a cautious approach to {query}. Consider diversifying your investments and consulting with financial professionals for comprehensive planning. Please verify this information with official sources and adapt strategies to your personal financial situation."

        # Generic but helpful response
        else:
            return f"Regarding {query}, I recommend focusing on fundamental investment principles: diversification across asset classes, regular systematic investing through SIPs, maintaining adequate emergency funds, and aligning investments with your time horizon and risk tolerance. These time-tested strategies form the foundation of successful financial planning. For specific guidance tailored to your situation, please consult with a qualified financial advisor."

    def _calculate_enhanced_confidence(self, context: Dict[str, Any], llm_response: str) -> float:
        """Calculate enhanced confidence score based on multiple factors"""

        confidence = 0.3  # Base confidence

        # Knowledge base quality contribution
        kb_quality = context.get("kb_quality_score", 0.0)
        confidence += kb_quality * 0.4  # Up to 40% from KB quality

        # Web search results contribution
        web_count = len(context.get("web_search_results", []))
        if web_count >= 2:
            confidence += 0.2
        elif web_count >= 1:
            confidence += 0.1

        # Response quality indicators
        response_length = len(llm_response)
        if response_length > 200:  # Comprehensive response
            confidence += 0.2
        elif response_length > 100:
            confidence += 0.1

        # Synthesis indicators in response
        synthesis_keywords = ["based on", "considering", "analysis shows", "strategy involves", "recommend", "however", "additionally"]
        synthesis_count = sum(1 for keyword in synthesis_keywords if keyword in llm_response.lower())
        if synthesis_count >= 3:
            confidence += 0.1

        # Overall context richness
        total_context_sources = len(context.get("knowledge_base_results", [])) + len(context.get("web_search_results", []))
        if total_context_sources >= 5:
            confidence += 0.1

        return min(confidence, 0.95)  # Cap at 95%

    def _evaluate_synthesis_quality(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate how well the response synthesizes rather than just retrieves"""

        synthesis_score = 0.0

        # Check for synthesis indicators
        synthesis_phrases = [
            "combining", "considering both", "analysis reveals", "this suggests", 
            "taking into account", "comprehensive approach", "strategy involves",
            "recommendation is", "key factors include", "balanced perspective"
        ]

        synthesis_count = sum(1 for phrase in synthesis_phrases if phrase in response.lower())
        synthesis_score += min(synthesis_count * 0.1, 0.3)  # Up to 30% for synthesis language

        # Check for original reasoning
        reasoning_phrases = [
            "because", "therefore", "as a result", "consequently", "due to",
            "this means", "implication", "leads to", "results in"
        ]

        reasoning_count = sum(1 for phrase in reasoning_phrases if phrase in response.lower())
        synthesis_score += min(reasoning_count * 0.1, 0.2)  # Up to 20% for reasoning

        # Check for structured response
        if any(marker in response for marker in ["1.", "2.", "first", "second", "additionally", "furthermore"]):
            synthesis_score += 0.2  # 20% for structured thinking

        # Check if response goes beyond simple context repetition
        if context.get("knowledge_base_results"):
            first_kb_content = context["knowledge_base_results"][0].get("content", "")

            # Simple check: if response is not just copying the knowledge base
            overlap_ratio = len(set(response.lower().split()) & set(first_kb_content.lower().split())) / max(len(response.split()), 1)
            if overlap_ratio < 0.7:  # Less than 70% overlap indicates original synthesis
                synthesis_score += 0.3

        return min(synthesis_score, 1.0)

    def _fallback_knowledge_search(self, query: str) -> List[Dict]:
        """Enhanced fallback knowledge search when ChromaDB is unavailable"""

        # Enhanced fallback knowledge base with synthesis-friendly content
        fallback_knowledge = {
            "sip": {
                "content": "SIP (Systematic Investment Plan) is a disciplined investment approach that helps average market volatility through rupee cost averaging. By investing a fixed amount regularly, you can benefit from compounding returns over the long term. For young investors, starting with Rs.500-1000 monthly in diversified equity funds can potentially build substantial wealth over 10-15 years. The key is consistency and gradually increasing the investment amount as income grows.",
                "metadata": {"category": "investment_strategies", "topic": "sip", "complexity": "detailed"}
            },
            "diversif": {
                "content": "Portfolio diversification involves spreading investments across different asset classes, sectors, and market capitalizations to reduce overall risk. A balanced approach for Indian investors includes 60-70% equity allocation (mixed across large, mid, small cap), 20-25% in debt instruments like PPF and bonds, 5-10% in gold, and 5% in international equity. This strategy helps weather market volatility while capturing growth opportunities across different economic cycles.",
                "metadata": {"category": "portfolio_management", "topic": "diversification", "complexity": "detailed"}
            },
            "tax": {
                "content": "Tax-efficient investing in India leverages multiple sections of the Income Tax Act. Section 80C allows Rs.1.5 lakh deduction through ELSS mutual funds (3-year lock-in with potential 12-15% returns), PPF (15-year commitment with 7-8% tax-free returns), and EPF. Additional benefits include Rs.50,000 under 80CCD for NPS and Rs.25,000 for health insurance under 80D. The strategy is to balance tax savings with long-term wealth creation goals.",
                "metadata": {"category": "tax_planning", "topic": "optimization", "complexity": "detailed"}
            },
            "stock": {
                "content": "Fundamental stock analysis requires evaluating both quantitative metrics and qualitative factors. Key ratios include P/E ratio (compare with industry peers), Debt-to-Equity ratio (below 0.5 preferred), ROE (above 15% indicates good management efficiency), and revenue growth consistency. Qualitative factors include management quality, competitive advantages, industry prospects, and regulatory environment. Always analyze 3-5 year trends and compare with industry benchmarks before making investment decisions.",
                "metadata": {"category": "stock_analysis", "topic": "fundamental_analysis", "complexity": "advanced"}
            },
            "mutual": {
                "content": "Mutual fund selection involves analyzing fund performance across market cycles, expense ratios (prefer below 1.5% for equity funds), fund manager track record, and portfolio concentration. Look for funds with consistent risk-adjusted returns measured by Sharpe ratio and Alpha. Consider fund size - avoid very small funds (below Rs.100 crore) and check for style drift in the portfolio. Direct plans have lower costs and are suitable for informed investors.",
                "metadata": {"category": "mutual_funds", "topic": "selection_criteria", "complexity": "intermediate"}
            },
            "emergency": {
                "content": "Emergency fund planning requires maintaining 6-12 months of essential expenses in highly liquid instruments. Build this fund gradually, starting with 1 month's expenses and progressively reaching the target. Keep funds in savings accounts for immediate access and liquid mutual funds for better returns. Avoid equity investments for emergency funds as they may be down when needed most. Replenish immediately after any withdrawal and review annually.",
                "metadata": {"category": "financial_planning", "topic": "emergency_preparedness", "complexity": "basic"}
            },
            "retirement": {
                "content": "Retirement planning requires starting early and leveraging the power of compounding. Calculate future expenses considering inflation and aim to save 15-20% of income for retirement. Use a mix of EPF, NPS, equity mutual funds, and PPF for tax-efficient wealth building. Asset allocation should shift from aggressive (80% equity) in youth to conservative (40% equity) as retirement approaches. Plan for healthcare inflation and maintain adequate insurance coverage.",
                "metadata": {"category": "retirement_planning", "topic": "comprehensive_strategy", "complexity": "detailed"}
            }
        }

        results = []
        query_lower = query.lower()

        # Enhanced matching with scoring
        for keyword, data in fallback_knowledge.items():
            score = 0.8  # Base score

            if keyword in query_lower:
                score = 0.9
                results.append({
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "relevance": score,
                    "source": "enhanced_fallback_knowledge"
                })

        # If no direct matches, return most relevant general advice
        if not results:
            general_advice = {
                "content": "For successful investing, focus on fundamental principles: start early to benefit from compounding, diversify across asset classes and sectors, invest regularly through SIPs to average market volatility, maintain adequate emergency funds, align investments with your time horizon and risk tolerance, and review your portfolio annually. These time-tested strategies form the foundation of long-term wealth creation.",
                "metadata": {"category": "general_advice", "topic": "investment_principles", "complexity": "basic"},
                "relevance": 0.6,
                "source": "general_fallback"
            }
            results.append(general_advice)

        return results[:3]

    # Keep other methods for compatibility
    def search_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Legacy method for backward compatibility - redirects to enhanced version"""
        return self.enhanced_knowledge_retrieval(query, n_results)

    def add_knowledge(self, content: str, category: str, topic: str) -> bool:
        """Add new knowledge to the knowledge base (unchanged)"""

        if not self.collection:
            print("⚠️ Cannot add knowledge - ChromaDB not available")
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
        """Get knowledge base statistics (unchanged)"""

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

# Global enhanced RAG system instance
rag_system = FinancialRAGSystem()