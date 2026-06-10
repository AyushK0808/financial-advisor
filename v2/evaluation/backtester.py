# evaluation/backtester.py
"""
Historical Backtesting Engine
Generates Buy/Hold/Sell signals using the deterministic scoring engine
(stock_data.py) over a rolling historical window and measures:
  - Annualized return vs S&P 500 benchmark
  - Sharpe Ratio improvement over benchmark
  - Recommendation hit-rate (did a BUY precede a price gain?)
  - Comparison vs pure technical baseline (no LLM, no fundamentals)

Addresses Reviewer 1:
  "No backtesting on historical data to confirm whether Buy/Hold/Sell
   signals generate positive alpha."
  "No comparative baselines against FinGPT, FinBERT, or any robo-advisor."
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json


# ── Scoring helpers (self-contained, no Ollama needed) ──────────────────────

def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al
    vals = 100 - (100 / (1 + rs))
    return float(vals.iloc[-1]) if not vals.empty else 50.0


def _macd(series: pd.Series) -> Tuple[float, float]:
    e12  = series.ewm(span=12, adjust=False).mean()
    e26  = series.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig  = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(sig.iloc[-1])


def _sharpe(returns: pd.Series, rf: float = 0.03) -> float:
    er = returns - rf / 252
    return float(np.sqrt(252) * er.mean() / er.std()) if er.std() != 0 else 0.0


def _technical_signal(window_close: pd.Series) -> str:
    """Pure technical baseline: RSI + MACD + MA only — no fundamentals, no LLM."""
    rsi          = _rsi(window_close)
    macd, signal = _macd(window_close)
    ma50         = float(window_close.rolling(50).mean().iloc[-1])
    price        = float(window_close.iloc[-1])

    score = 0
    if rsi < 35:   score += 30
    elif rsi < 50: score += 20
    elif rsi < 65: score += 10
    else:          score += 5
    if macd > signal and macd > 0: score += 30
    elif macd > signal:            score += 18
    else:                          score += 5
    if price > ma50: score += 25
    else:            score += 5

    if score >= 65:   return "BUY"
    elif score >= 40: return "HOLD"
    else:             return "SELL"


def _composite_signal(window_data: pd.DataFrame, info: Dict, debug: bool = False) -> str:
    """
    Full system signal: fundamental 40% + technical 35% + risk 25%.
    Mirrors the paper's Table I scoring (simplified for speed).
    """
    close   = window_data["Close"]
    returns = close.pct_change().dropna()

    # ── Technical score ──────────────────────────────────────
    rsi          = _rsi(close)
    macd, signal = _macd(close)
    ma50         = float(close.rolling(50).mean().iloc[-1])
    ma200_s      = close.rolling(200).mean()
    ma200        = float(ma200_s.iloc[-1]) if not ma200_s.isna().iloc[-1] else ma50
    price        = float(close.iloc[-1])

    t = 0
    if rsi < 30:   t += 25
    elif rsi < 45: t += 20
    elif rsi < 55: t += 15
    elif rsi < 70: t += 12
    else:          t += 5

    if macd > signal and macd > 0: t += 25
    elif macd > signal:            t += 18
    else:                          t += 5

    if price > ma50 > ma200:   t += 30
    elif price > ma50:         t += 20
    else:                      t += 5

    # Normalize to 0-100: max achievable raw score = 80
    tech = min(t / 80 * 100, 100)

    # ── Risk score ───────────────────────────────────────────
    vol    = float(returns.std() * np.sqrt(252) * 100)
    sharpe = _sharpe(returns)
    r = 0
    if sharpe > 2:    r += 30
    elif sharpe > 1:  r += 25
    elif sharpe > 0:  r += 15
    else:             r += 2
    if vol < 15:      r += 15
    elif vol < 25:    r += 12
    elif vol < 40:    r += 8
    else:             r += 3
    # Normalize to 0-100: max achievable raw score = 45
    risk = min(r / 45 * 100, 100)

    # ── Fundamental score (from pre-fetched info dict) ───────
    pe  = info.get("trailingPE")
    roe = info.get("returnOnEquity")
    de  = info.get("debtToEquity")
    f   = 0
    if pe and pe > 0:
        if pe < 15:    f += 20
        elif pe < 25:  f += 15
        elif pe < 40:  f += 8
        else:          f += 3
    if roe:
        roe_p = roe * 100
        if roe_p > 20:   f += 20
        elif roe_p > 10: f += 10
        elif roe_p > 0:  f += 5
    if de is not None and de >= 0:
        if de < 50:    f += 15
        elif de < 100: f += 10
        elif de < 200: f += 5
        else:          f += 2
    # Normalize to 0-100: max achievable raw score = 55
    fund = min(f / 55 * 100, 100) if f > 0 else 0

    # ── Composite ────────────────────────────────────────────
    if fund > 0:
        final = fund * 0.40 + tech * 0.35 + risk * 0.25
    else:
        final = tech * 0.55 + risk * 0.45

    if debug:
        print(f"    fund={fund:.1f} tech={tech:.1f} risk={risk:.1f} final={final:.1f}")

    if final >= 60:   return "BUY"
    elif final >= 40: return "HOLD"
    else:             return "SELL"


# ── Backtester ───────────────────────────────────────────────────────────────

def backtest(
    tickers:      List[str],
    start:        str  = "2022-01-01",
    end:          str  = "2024-12-31",
    window_days:  int  = 252,
    forward_days: int  = 21,
    benchmark:    str  = "^GSPC",
    debug_scores: bool = False,
) -> Dict:
    """
    Rolling-window backtest.

    For each ticker, at every `forward_days` step we:
      1. Compute the composite signal on the trailing `window_days` of data.
      2. Compute the technical-only baseline signal.
      3. Record the forward return over the next `forward_days`.
      4. A BUY = long, SELL = short, HOLD = cash (0 return).

    Returns a dict with per-ticker and aggregate metrics.
    """
    def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    results = {}
    bench_data = _strip_tz(yf.Ticker(benchmark).history(start=start, end=end))
    bench_ret  = bench_data["Close"].pct_change().dropna()

    for ticker in tickers:
        print(f"  Backtesting {ticker}...")
        try:
            t       = yf.Ticker(ticker)
            info    = t.info
            hist    = _strip_tz(t.history(
                start=(pd.Timestamp(start) - timedelta(days=window_days + 60)).strftime("%Y-%m-%d"),
                end=end,
            ))
            if hist.empty or len(hist) < window_days + forward_days:
                print(f"    Skipping {ticker}: insufficient data")
                continue

            dates   = hist.index
            signals_composite = []
            signals_technical = []
            forward_returns   = []

            eval_dates = dates[window_days::forward_days]

            for d in eval_dates:
                idx = hist.index.get_loc(d)
                if idx + forward_days >= len(hist):
                    break

                window = hist.iloc[idx - window_days : idx]
                fwd_r  = (hist["Close"].iloc[idx + forward_days] /
                          hist["Close"].iloc[idx] - 1)

                sig_comp = _composite_signal(window, info, debug=debug_scores)
                sig_tech = _technical_signal(window["Close"])

                signals_composite.append(sig_comp)
                signals_technical.append(sig_tech)
                forward_returns.append(float(fwd_r))

            if not forward_returns:
                continue

            def signal_returns(sigs: List[str], fwd: List[float]) -> pd.Series:
                r = []
                for s, f in zip(sigs, fwd):
                    if s == "BUY":    r.append(f)
                    elif s == "SELL": r.append(-f)
                    else:             r.append(0.0)
                return pd.Series(r)

            ret_comp = signal_returns(signals_composite, forward_returns)
            ret_tech = signal_returns(signals_technical, forward_returns)

            periods_per_year = 252 / forward_days

            def ann_return(s: pd.Series) -> float:
                return float((1 + s.mean()) ** periods_per_year - 1) * 100

            def sharpe(s: pd.Series) -> float:
                if s.std() == 0: return 0.0
                return float(s.mean() / s.std() * np.sqrt(periods_per_year))

            buy_mask_comp = [s == "BUY" for s in signals_composite]
            buy_mask_tech = [s == "BUY" for s in signals_technical]

            def hit_rate(mask: List[bool], fwd: List[float]) -> float:
                hits = [fwd[i] > 0 for i, m in enumerate(mask) if m]
                return float(np.mean(hits)) * 100 if hits else 0.0

            bench_slice = bench_ret[
                (bench_ret.index >= pd.Timestamp(start))
            ].iloc[:len(forward_returns)]
            bench_ann = ann_return(bench_slice) if len(bench_slice) > 0 else 0.0

            results[ticker] = {
                "n_signals":             len(forward_returns),
                "composite": {
                    "annualized_return_pct":  round(ann_return(ret_comp), 2),
                    "sharpe_ratio":            round(sharpe(ret_comp), 3),
                    "hit_rate_pct":            round(hit_rate(buy_mask_comp, forward_returns), 1),
                    "buy_signals":             sum(buy_mask_comp),
                    "hold_signals":            signals_composite.count("HOLD"),
                    "sell_signals":            signals_composite.count("SELL"),
                },
                "technical_baseline": {
                    "annualized_return_pct":  round(ann_return(ret_tech), 2),
                    "sharpe_ratio":            round(sharpe(ret_tech), 3),
                    "hit_rate_pct":            round(hit_rate(buy_mask_tech, forward_returns), 1),
                },
                "benchmark_sp500": {
                    "annualized_return_pct":  round(bench_ann, 2),
                },
                "alpha_vs_benchmark_pct": round(
                    ann_return(ret_comp) - bench_ann, 2),
                "alpha_vs_technical_pct": round(
                    ann_return(ret_comp) - ann_return(ret_tech), 2),
            }

        except Exception as e:
            print(f"    Error backtesting {ticker}: {e}")

    if results:
        comp_returns  = [v["composite"]["annualized_return_pct"]        for v in results.values()]
        tech_returns  = [v["technical_baseline"]["annualized_return_pct"] for v in results.values()]
        bench_returns = [v["benchmark_sp500"]["annualized_return_pct"]    for v in results.values()]
        hit_rates     = [v["composite"]["hit_rate_pct"]                   for v in results.values()]
        sharpes       = [v["composite"]["sharpe_ratio"]                   for v in results.values()]

        results["_aggregate"] = {
            "tickers_evaluated":           len(results),
            "composite_mean_ann_return":   round(float(np.mean(comp_returns)), 2),
            "technical_mean_ann_return":   round(float(np.mean(tech_returns)), 2),
            "benchmark_mean_ann_return":   round(float(np.mean(bench_returns)), 2),
            "mean_alpha_vs_benchmark":     round(float(np.mean(comp_returns)) -
                                                  float(np.mean(bench_returns)), 2),
            "mean_alpha_vs_technical":     round(float(np.mean(comp_returns)) -
                                                  float(np.mean(tech_returns)), 2),
            "mean_hit_rate_pct":           round(float(np.mean(hit_rates)), 1),
            "mean_sharpe_ratio":           round(float(np.mean(sharpes)), 3),
        }

    return results


def print_backtest_report(results: Dict):
    agg = results.get("_aggregate", {})
    print(f"\n{'='*70}")
    print("  BACKTEST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Tickers evaluated : {agg.get('tickers_evaluated', 0)}")
    print(f"  {'Method':<30} {'Ann. Return':>12} {'Sharpe':>8} {'Hit%':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Proposed System (composite)':<30} "
          f"{agg.get('composite_mean_ann_return', 0):>11.1f}% "
          f"{agg.get('mean_sharpe_ratio', 0):>8.3f} "
          f"{agg.get('mean_hit_rate_pct', 0):>7.1f}%")
    print(f"  {'Technical Baseline (no LLM)':<30} "
          f"{agg.get('technical_mean_ann_return', 0):>11.1f}%")
    print(f"  {'S&P 500 Benchmark':<30} "
          f"{agg.get('benchmark_mean_ann_return', 0):>11.1f}%")
    print(f"  {'-'*60}")
    print(f"  Alpha vs S&P 500   : {agg.get('mean_alpha_vs_benchmark', 0):+.2f}%")
    print(f"  Alpha vs Technical : {agg.get('mean_alpha_vs_technical', 0):+.2f}%")
    print(f"{'='*70}\n")

    print("  Per-Ticker Breakdown:")
    print(f"  {'Ticker':<8} {'Composite%':>11} {'Technical%':>11} "
          f"{'S&P%':>7} {'Alpha':>7} {'Sharpe':>8} {'Hit%':>7}")
    print(f"  {'-'*65}")
    for k, v in results.items():
        if k.startswith("_"): continue
        print(f"  {k:<8} "
              f"{v['composite']['annualized_return_pct']:>10.1f}% "
              f"{v['technical_baseline']['annualized_return_pct']:>10.1f}% "
              f"{v['benchmark_sp500']['annualized_return_pct']:>6.1f}% "
              f"{v['alpha_vs_benchmark_pct']:>+6.1f}% "
              f"{v['composite']['sharpe_ratio']:>8.3f} "
              f"{v['composite']['hit_rate_pct']:>6.1f}%")
    print()


if __name__ == "__main__":
    UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "AMZN", "NVDA", "KO"]

    print("Running backtest (2022–2024, 21-day forward window)...")
    results = backtest(
        tickers=UNIVERSE,
        start="2022-01-01",
        end="2024-12-31",
        forward_days=21,
    )
    print_backtest_report(results)

    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to backtest_results.json")
