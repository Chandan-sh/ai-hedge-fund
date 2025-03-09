from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import math

class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def ben_graham_agent(state: AgentState):
    data, end_date, tickers = state["data"], state["data"]["end_date"], state["data"]["tickers"]
    analysis_data, graham_analysis = {}, {}

    # Batch fetch all financial data to reduce API calls
    all_metrics = {t: get_financial_metrics(t, end_date, period="annual", limit=10) for t in tickers}
    all_financials = {t: search_line_items(t, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10) for t in tickers}
    all_market_caps = {t: get_market_cap(t, end_date) for t in tickers}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Analyzing stock")
        metrics, financials, market_cap = all_metrics[ticker], all_financials[ticker], all_market_caps[ticker]
        
        earnings_analysis = analyze_earnings_stability(metrics, financials)
        strength_analysis = analyze_financial_strength(metrics, financials)
        valuation_analysis = analyze_valuation_graham(metrics, financials, market_cap)
        
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_score = 15
        signal = "bullish" if total_score >= 0.7 * max_score else "bearish" if total_score <= 0.3 * max_score else "neutral"
        
        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}
        
        graham_output = generate_graham_output(ticker, analysis_data, state["metadata"]["model_name"], state["metadata"]["model_provider"])
        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")
    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}

def analyze_earnings_stability(metrics, financials):
    if not financials:
        return {"score": 0, "details": "Insufficient data for earnings stability analysis"}
    
    eps_vals = [item.earnings_per_share for item in financials if item.earnings_per_share is not None]
    if len(eps_vals) < 2:
        return {"score": 0, "details": "Not enough multi-year EPS data."}
    
    positive_years = sum(1 for e in eps_vals if e > 0)
    score = 3 if positive_years == len(eps_vals) else 2 if positive_years >= len(eps_vals) * 0.8 else 0
    score += 1 if eps_vals[-1] > eps_vals[0] else 0
    
    return {"score": score, "details": "EPS stability and growth analyzed."}

def analyze_financial_strength(metrics, financials):
    if not financials:
        return {"score": 0, "details": "No data for financial strength analysis"}
    
    latest = financials[-1]
    assets, liabilities, c_assets, c_liabilities = latest.total_assets, latest.total_liabilities, latest.current_assets, latest.current_liabilities
    score = 2 if c_liabilities and (c_assets / c_liabilities) >= 2 else 1 if c_liabilities and (c_assets / c_liabilities) >= 1.5 else 0
    score += 2 if assets and liabilities / assets < 0.5 else 1 if liabilities / assets < 0.8 else 0
    score += 1 if any(item.dividends_and_other_cash_distributions and item.dividends_and_other_cash_distributions < 0 for item in financials) else 0
    
    return {"score": score, "details": "Liquidity, debt ratio, and dividends analyzed."}

def analyze_valuation_graham(metrics, financials, market_cap):
    if not financials or not market_cap:
        return {"score": 0, "details": "Insufficient data for valuation"}
    
    latest = financials[-1]
    c_assets, liabilities, book_value_ps, eps, shares_outstanding = latest.current_assets, latest.total_liabilities, latest.book_value_per_share, latest.earnings_per_share, latest.outstanding_shares
    net_current_assets = c_assets - liabilities
    price_per_share = market_cap / shares_outstanding if shares_outstanding else 0
    score = 4 if net_current_assets > market_cap else 2 if net_current_assets / shares_outstanding >= price_per_share * 0.67 else 0
    graham_number = math.sqrt(22.5 * eps * book_value_ps) if eps > 0 and book_value_ps > 0 else 0
    margin_safety = (graham_number - price_per_share) / price_per_share if graham_number and price_per_share else 0
    score += 3 if margin_safety > 0.5 else 1 if margin_safety > 0.2 else 0
    
    return {"score": score, "details": "Graham valuation principles applied."}

def generate_graham_output(ticker, analysis_data, model_name, model_provider):
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a Benjamin Graham AI agent providing value-based investment decisions."),
        ("human", "Analysis Data for {ticker}: {analysis_data}")
    ])
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})
    return call_llm(prompt, model_name, model_provider, BenGrahamSignal, "ben_graham_agent", lambda: BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Defaulting to neutral due to error."))
