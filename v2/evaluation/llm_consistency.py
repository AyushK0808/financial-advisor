# evaluation/llm_consistency.py
"""
LLM Consistency Evaluator
Runs the same news context through the sentiment LLM N times and
computes score variance, std-dev, and coefficient of variation per factor.
This directly addresses Reviewer 1's critique:
  "no analysis of LLM consistency across repeated runs, a critical gap
   given LLMs' known non-determinism"
"""

import json
import numpy as np
from typing import List, Dict
import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

FACTORS = [
    "company_performance",
    "management_and_governance",
    "industry_and_sector_health",
    "competitive_landscape",
    "regulatory_risk",
    "macroeconomic_exposure",
    "overall_sentiment",
]

ANALYSIS_PROMPT = """You are an expert financial analyst. Analyze the news context below and score
each investment factor from 1 (Very Negative) to 10 (Very Positive).

Return ONLY a valid JSON object with this exact structure:
{{
  "company_performance":       {{"score": <1-10>, "justification": "..."}},
  "management_and_governance": {{"score": <1-10>, "justification": "..."}},
  "industry_and_sector_health":{{"score": <1-10>, "justification": "..."}},
  "competitive_landscape":     {{"score": <1-10>, "justification": "..."}},
  "regulatory_risk":           {{"score": <1-10>, "justification": "..."}},
  "macroeconomic_exposure":    {{"score": <1-10>, "justification": "..."}},
  "overall_sentiment":         {{"score": <1-10>, "justification": "..."}},
  "risk_flags":   ["..."],
  "opportunities":["..."]
}}

News Context:
{context}"""


def _call_llm(context: str) -> Dict:
    """Single LLM call; returns parsed JSON or raises."""
    # Keep context short enough to stay within llama3.2's default 2048-token window
    prompt = ANALYSIS_PROMPT.format(context=context[:1500])
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt,
              "stream": False, "format": "json",
              "options": {"temperature": 0.3, "num_ctx": 4096}},
        timeout=300,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_consistency_analysis(
    context: str,
    n_runs: int = 5,
) -> Dict:
    """
    Run the sentiment LLM n_runs times on the same context.

    Returns
    -------
    {
      "n_runs": int,
      "per_factor": {
          factor_name: {
              "scores":  [s1, s2, ...],
              "mean":    float,
              "std":     float,
              "cv_pct":  float,   # coefficient of variation (std/mean*100)
              "min":     float,
              "max":     float,
          }
      },
      "overall_consistency_score": float,   # 100 - mean(cv_pct) across factors
      "runs_succeeded": int,
    }
    """
    all_scores: Dict[str, List[float]] = {f: [] for f in FACTORS}
    runs_succeeded = 0

    for i in range(n_runs):
        try:
            result = _call_llm(context)
            for factor in FACTORS:
                score = result.get(factor, {}).get("score")
                if isinstance(score, (int, float)) and 1 <= score <= 10:
                    all_scores[factor].append(float(score))
            runs_succeeded += 1
            print(f"  Run {i+1}/{n_runs} succeeded.")
        except Exception as e:
            print(f"  Run {i+1}/{n_runs} failed: {e}")

    per_factor = {}
    cv_values  = []

    for factor, scores in all_scores.items():
        if len(scores) < 2:
            per_factor[factor] = {
                "scores": scores, "mean": scores[0] if scores else None,
                "std": 0.0, "cv_pct": 0.0, "min": None, "max": None,
            }
            continue
        arr  = np.array(scores)
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1))
        cv   = (std / mean * 100) if mean > 0 else 0.0
        cv_values.append(cv)
        per_factor[factor] = {
            "scores":  scores,
            "mean":    round(mean, 3),
            "std":     round(std, 3),
            "cv_pct":  round(cv, 2),
            "min":     float(np.min(arr)),
            "max":     float(np.max(arr)),
        }

    overall = round(100 - np.mean(cv_values), 2) if cv_values else None

    return {
        "n_runs":                    n_runs,
        "runs_succeeded":            runs_succeeded,
        "per_factor":                per_factor,
        "overall_consistency_score": overall,
    }


def print_consistency_report(result: Dict, ticker: str = ""):
    header = f"LLM CONSISTENCY REPORT - {ticker}" if ticker else "LLM CONSISTENCY REPORT"
    print(f"\n{'='*65}")
    print(f"  {header}")
    print(f"  Runs: {result['runs_succeeded']}/{result['n_runs']}   "
          f"Overall Consistency Score: {result['overall_consistency_score']:.1f}/100")
    print(f"{'='*65}")
    print(f"{'Factor':<35} {'Mean':>6} {'Std':>6} {'CV%':>7} {'Range':>12}")
    print(f"{'-'*65}")
    for factor, stats in result["per_factor"].items():
        rng = f"{stats['min']:.1f}-{stats['max']:.1f}" if stats["min"] is not None else "N/A"
        print(f"{factor:<35} {stats['mean']:>6.2f} {stats['std']:>6.3f} "
              f"{stats['cv_pct']:>6.1f}% {rng:>12}")
    print(f"{'='*65}\n")
    print("Interpretation:")
    print("  CV% < 10  -> High consistency (publishable)")
    print("  CV% 10-20 -> Moderate consistency (acceptable with caveat)")
    print("  CV% > 20  -> Low consistency (prompt/temperature adjustment needed)\n")


if __name__ == "__main__":
    import sys
    import os
    import json as _json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from stock_news import NewsFetcher, CompanyProfileFetcher, DatabaseManager, DATABASE_FILE

    args = sys.argv[1:]
    # Last arg is n_runs if it's a plain integer, otherwise default 5
    if args and args[-1].isdigit():
        n_runs = int(args[-1])
        tickers = args[:-1]
    else:
        n_runs = 5
        tickers = args

    if not tickers:
        tickers = ["AAPL"]

    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
    db      = DatabaseManager(DATABASE_FILE)
    fetcher = NewsFetcher(NEWS_API_KEY, db)

    all_results = {}
    for ticker in tickers:
        print(f"\n{'-'*65}")
        print(f"  Processing {ticker} ({tickers.index(ticker)+1}/{len(tickers)})")
        print(f"{'-'*65}")
        profile = CompanyProfileFetcher.fetch_profile(ticker)
        if not profile:
            print(f"  Could not fetch profile for {ticker}, skipping.")
            continue
        sections = fetcher.fetch_comprehensive_news(profile, ["United States"])
        context  = "\n".join(v for v in sections.values() if v)
        print(f"  Running {n_runs} consistency runs for {ticker}...")
        result = run_consistency_analysis(context, n_runs=n_runs)
        print_consistency_report(result, ticker)
        all_results[ticker] = result

    out_path = os.path.join(os.path.dirname(__file__), "..", "consistency_results.json")
    with open(out_path, "w") as f:
        _json.dump(all_results, f, indent=2)
    print(f"All results saved to consistency_results.json")
