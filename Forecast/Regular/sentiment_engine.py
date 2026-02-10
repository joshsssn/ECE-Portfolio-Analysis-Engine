"""
FinBERT + OpenRouter Hybrid Sentiment Engine
=============================================
Layer 1: FinBERT-tone (local) — fast headline scoring
Layer 2: OpenRouter LLM (API) — deep article analysis (optional)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────
# Streamlit compatibility: tqdm / transformers need
# sys.stdout.isatty() — Streamlit's RealTimeLogger
# replaces stdout and may lack this method.
# ─────────────────────────────────────────────────
def _patch_streams():
    """Ensure sys.stdout and sys.stderr have isatty()."""
    for attr in ('stdout', 'stderr'):
        stream = getattr(sys, attr)
        if not hasattr(stream, 'isatty'):
            try:
                stream.isatty = lambda: False
            except (AttributeError, TypeError):
                pass


# ─────────────────────────────────────────────────
# Lazy-loaded globals
# ─────────────────────────────────────────────────
_finbert_pipeline = None


def _get_finbert():
    """Lazy-load FinBERT on first call."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        _patch_streams()  # Ensure isatty() exists before transformers loads
        print("   🔄 Loading ProsusAI/finbert model (first time only)...")
        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert", use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512
        )
        print("   ✅ ProsusAI/finbert loaded (safetensors).")
    return _finbert_pipeline


# ─────────────────────────────────────────────────
# 1. News Fetching
# ─────────────────────────────────────────────────
def fetch_news(ticker, days=30):
    """
    Fetch news headlines for a ticker.
    Primary: Refinitiv  |  Fallback: yfinance
    Returns list of dicts: [{title, date, source, body}]
    """
    headlines = []

    # --- Attempt 1: Refinitiv ---
    try:
        headlines = _fetch_refinitiv_news(ticker, days)
        if headlines:
            print(f"   📡 Fetched {len(headlines)} headlines from Refinitiv")
            return headlines
    except Exception as e:
        print(f"   [WARN] Refinitiv news fetch failed: {e}")

    # --- Attempt 2: yfinance fallback ---
    try:
        headlines = _fetch_yfinance_news(ticker)
        if headlines:
            print(f"   📰 Fetched {len(headlines)} headlines from yfinance")
            return headlines
    except Exception as e:
        print(f"   [WARN] yfinance news fetch failed: {e}")

    print(f"   ⚠️ No news found for {ticker}")
    return headlines


def _fetch_refinitiv_news(ticker, days=30):
    """Fetch news from Refinitiv Data Library."""
    import refinitiv.data as rd

    # Convert ticker format (e.g., AAPL -> AAPL.O)
    ric = ticker if '.' in ticker else f"{ticker}.O"

    headlines_df = rd.news.get_headlines(
        query=f"R:{ric}",
        count=100,
        start=datetime.now() - timedelta(days=days),
        end=datetime.now()
    )

    if headlines_df is None or headlines_df.empty:
        return []

    results = []
    for _, row in headlines_df.iterrows():
        item = {
            'title': str(row.get('headline', row.get('text', ''))),
            'date': str(row.name if hasattr(row, 'name') else ''),
            'source': 'Refinitiv',
            'body': None,
            'story_id': row.get('storyId', None)
        }

        # Try to fetch full article body
        if item['story_id']:
            try:
                story = rd.news.get_story(item['story_id'])
                if story:
                    item['body'] = str(story)
            except Exception:
                pass

        results.append(item)

    return results


def _fetch_yfinance_news(ticker):
    """Fetch news from yfinance (fallback)."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    news_list = stock.news

    if not news_list:
        return []

    results = []
    for article in news_list:
        content = article.get('content', {})
        item = {
            'title': content.get('title', article.get('title', 'N/A')),
            'date': content.get('pubDate', article.get('providerPublishTime', '')),
            'source': 'yfinance',
            'body': content.get('summary', None)
        }

        # Convert Unix timestamp if needed
        if isinstance(item['date'], (int, float)):
            item['date'] = datetime.fromtimestamp(item['date']).isoformat()

        results.append(item)

    return results


# ─────────────────────────────────────────────────
# 2. FinBERT Scoring (Layer 1)
# ─────────────────────────────────────────────────
def score_with_finbert(headlines):
    """
    Score headlines using FinBERT.
    Returns enriched list with finbert_label and finbert_score.
    """
    if not headlines:
        return headlines

    pipe = _get_finbert()
    texts = [h['title'] for h in headlines if h.get('title')]

    if not texts:
        return headlines

    # Re-patch in case Streamlit replaced stdout since module import
    _patch_streams()

    # Batch inference
    results = pipe(texts, batch_size=16)

    for i, (headline, result) in enumerate(zip(headlines, results)):
        label = result['label'].lower()
        score = result['score']

        # Convert to -1 to +1 scale
        if label == 'positive':
            sentiment_score = score
        elif label == 'negative':
            sentiment_score = -score
        else:  # neutral
            sentiment_score = 0.0

        headline['finbert_label'] = label
        headline['finbert_confidence'] = round(score, 4)
        headline['finbert_score'] = round(sentiment_score, 4)

    return headlines


# ─────────────────────────────────────────────────
# 3. OpenRouter LLM Analysis (Layer 2)
# ─────────────────────────────────────────────────
def analyze_with_llm(headlines, api_key, model="openai/gpt-oss-20b-free"):
    """
    Deep analysis of article bodies using OpenRouter LLM.
    Only processes articles that have a body.
    Returns enriched headlines with llm_summary field.
    """
    if not api_key:
        return headlines

    articles_with_body = [h for h in headlines if h.get('body')]
    if not articles_with_body:
        print("   [INFO] No article bodies available for LLM analysis.")
        return headlines

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/joshsssn/ECE", # Required for OpenRouter rankings
            "X-Title": "ECE Financial Analysis",              # Optional but recommended
        }
    )

    print(f"   🤖 Starting deep analysis on {len(articles_with_body)} articles...")
    
    analyzed = 0
    for i, headline in enumerate(articles_with_body):
        try:
            print(f"      👉 [{i+1}/{len(articles_with_body)}] Analyzing: {headline['title'][:50]}...")
            body_text = headline['body'][:4000]  # Limit to avoid token overflow

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial analyst. Analyze the article. "
                            "Output STRICT JSON ONLY. No markdown. No conversational text.\n"
                            "Format:\n"
                            "{\n"
                            '  "sentiment_score": float (-1.0 to 1.0),\n'
                            '  "key_risks": ["risk1", "risk2", ...],\n'
                            '  "key_opportunities": ["opp1", "opp2", ...],\n'
                            '  "summary": "One sentence summary"\n'
                            "}"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Title: {headline['title']}\n\n{body_text}"
                    }
                ],
                temperature=0.1,
                max_tokens=1000  # Increased to prevent JSON truncation
            )

            raw = response.choices[0].message.content.strip()
            
            # Parsing Logic
            import re
            parsed = {}
            
            # 1. Try direct JSON parse
            try:
                # Clean markdown blocks
                clean_raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_raw)
            except json.JSONDecodeError:
                # 2. Fallback: Regex extraction
                # Extract sentiment
                sent_match = re.search(r'"sentiment_score"\s*:\s*([-+]?\d*\.\d+|[-+]?\d+)', raw)
                if sent_match:
                    parsed['sentiment_score'] = float(sent_match.group(1))
                
                # Extract summary
                summary_match = re.search(r'"summary"\s*:\s*"(.*?)"', raw, re.DOTALL)
                if summary_match:
                    parsed['summary'] = summary_match.group(1)
                
                # Extract lists (rudimentary)
                # We can skip complex list parsing in fallback for now or try to salvage
                pass

            headline['llm_sentiment'] = parsed.get('sentiment_score', 0)
            headline['llm_risks'] = parsed.get('key_risks', [])
            headline['llm_opportunities'] = parsed.get('key_opportunities', [])
            headline['llm_summary'] = parsed.get('summary', raw[:200]) # Fallback to raw text if no summary
            analyzed += 1

        except Exception as e:
            print(f"   [WARN] LLM analysis failed for article: {e}")
            continue

    print(f"   🤖 LLM analysis completed: {analyzed}/{len(articles_with_body)} articles")
    return headlines


# ─────────────────────────────────────────────────
# 4. Output Generation
# ─────────────────────────────────────────────────
def save_sentiment_report(headlines, ticker, output_dir):
    """
    Save sentiment CSV and plot.
    Returns summary dict for master integration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not headlines:
        print(f"   ⚠️ No headlines to save for {ticker}")
        return None

    # --- CSV ---
    df = pd.DataFrame(headlines)

    # Select columns that exist
    cols = ['date', 'title', 'source', 'finbert_label', 'finbert_confidence', 'finbert_score']
    optional_cols = ['llm_sentiment', 'llm_summary', 'llm_risks', 'llm_opportunities']
    for c in optional_cols:
        if c in df.columns:
            cols.append(c)

    existing_cols = [c for c in cols if c in df.columns]
    df_out = df[existing_cols].copy()

    csv_path = output_dir / f"{ticker}_sentiment.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"   💾 Saved sentiment report: {csv_path}")

    # --- Plots ---
    try:
        # 1. FinBERT Plot
        _plot_generic(
            df_out, 
            score_col='finbert_score',
            title=f'📰 {ticker} — FinBERT Sentiment Analysis',
            filename=f"{ticker}_sentiment.png",
            output_dir=output_dir
        )
        
        # 2. LLM Plot (if available)
        if 'llm_sentiment' in df_out.columns and df_out['llm_sentiment'].notna().sum() > 0:
             _plot_generic(
                df_out, 
                score_col='llm_sentiment',
                title=f'🤖 {ticker} — AI Deep Analysis',
                filename=f"{ticker}_sentiment_llm.png",
                output_dir=output_dir
            )
            
    except Exception as e:
        print(f"   [WARN] Could not generate sentiment plots: {e}")

    # --- Summary ---
    summary = _compute_summary(df_out, ticker)
    return summary


def _compute_summary(df, ticker):
    """Compute aggregate sentiment summary."""
    if 'finbert_score' not in df.columns or df['finbert_score'].isna().all():
        return None

    scores = df['finbert_score'].dropna()
    avg_score = scores.mean()
    n_positive = (df['finbert_label'] == 'positive').sum() if 'finbert_label' in df.columns else 0
    n_negative = (df['finbert_label'] == 'negative').sum() if 'finbert_label' in df.columns else 0
    n_neutral = (df['finbert_label'] == 'neutral').sum() if 'finbert_label' in df.columns else 0
    total = len(df)

    # Overall label
    if avg_score > 0.15:
        overall = "BULLISH 🟢"
    elif avg_score < -0.15:
        overall = "BEARISH 🔴"
    else:
        overall = "NEUTRAL 🟡"

    summary = {
        'ticker': ticker,
        'overall_sentiment': overall,
        'avg_score': round(avg_score, 4),
        'n_headlines': total,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_neutral': n_neutral,
        'pct_positive': round(n_positive / total * 100, 1) if total > 0 else 0,
        'pct_negative': round(n_negative / total * 100, 1) if total > 0 else 0,
    }

    # Add LLM insights if available
    if 'llm_risks' in df.columns:
        all_risks = []
        for r in df['llm_risks'].dropna():
            if isinstance(r, list):
                all_risks.extend(r)
        summary['top_risks'] = list(set(all_risks))[:5]

    if 'llm_opportunities' in df.columns:
        all_opps = []
        for o in df['llm_opportunities'].dropna():
            if isinstance(o, list):
                all_opps.extend(o)
        summary['top_opportunities'] = list(set(all_opps))[:5]

    return summary


def _plot_generic(df, score_col, title, filename, output_dir):
    """Generate generic sentiment timeline plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Filter data for this specific score column
    plot_df = df[df[score_col].notna()].copy()
    
    if plot_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#1a1a2e')

    # Parse dates
    # dates = pd.to_datetime(plot_df['date'], errors='coerce') 
    scores = plot_df[score_col].values

    # Colors by sentiment
    colors = []
    for s in scores:
        if s > 0.1:
            colors.append('#4cc9f0')  # Positive - cyan
        elif s < -0.1:
            colors.append('#e63946')  # Negative - red
        else:
            colors.append('#888888')  # Neutral - grey

    # ─── Top panel: Sentiment scores bar chart ───
    ax1.set_facecolor('#16213e')
    ax1.bar(range(len(scores)), scores, color=colors, alpha=0.85, width=0.8)
    ax1.axhline(y=0, color='#ffffff', linewidth=0.5, alpha=0.3)
    ax1.axhline(y=0.15, color='#4cc9f0', linewidth=0.5, alpha=0.3, linestyle='--')
    ax1.axhline(y=-0.15, color='#e63946', linewidth=0.5, alpha=0.3, linestyle='--')

    ax1.set_ylabel('Sentiment Score', color='white', fontsize=11)
    ax1.set_title(title, color='white',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.tick_params(colors='white', labelsize=8)
    ax1.set_xlim(-0.5, len(scores) - 0.5)
    ax1.set_ylim(-1.1, 1.1)
    ax1.spines['bottom'].set_color('#333')
    ax1.spines['left'].set_color('#333')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add avg line
    avg = np.mean(scores)
    ax1.axhline(y=avg, color='#f0c808', linewidth=1.5, linestyle='-', alpha=0.7,
                label=f'Avg: {avg:.3f}')
    ax1.legend(loc='upper right', fontsize=9, facecolor='#16213e', edgecolor='#333',
               labelcolor='white')

    # ─── Bottom panel: Distribution ───
    ax2.set_facecolor('#16213e')
    n_pos = sum(1 for s in scores if s > 0.1)
    n_neg = sum(1 for s in scores if s < -0.1)
    n_neu = len(scores) - n_pos - n_neg

    bars = ax2.barh(['Negative', 'Neutral', 'Positive'],
                    [n_neg, n_neu, n_pos],
                    color=['#e63946', '#888888', '#4cc9f0'], alpha=0.85)

    for bar, val in zip(bars, [n_neg, n_neu, n_pos]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(val), color='white', va='center', fontsize=10)

    ax2.set_xlabel('Count', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#333')
    ax2.spines['left'].set_color('#333')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"   📊 Saved sentiment plot: {plot_path}")


# ─────────────────────────────────────────────────
# 5. Main Entry Point
# ─────────────────────────────────────────────────
def run_sentiment_analysis(ticker, output_dir, days=30,
                           enable_openrouter=False, openrouter_key=None,
                           openrouter_model="openai/gpt-oss-20b-free"):
    """
    Full sentiment analysis pipeline for a single ticker.

    Args:
        ticker: Stock ticker (e.g. 'AAPL')
        output_dir: Path to save outputs
        days: Number of days of news to look back
        enable_openrouter: Whether to use OpenRouter LLM for deep analysis
        openrouter_key: OpenRouter API key
        openrouter_model: OpenRouter model ID

    Returns:
        Summary dict or None
    """
    print(f"\n   🔍 Sentiment Analysis for {ticker}")
    print(f"   ├─ Layer 1: FinBERT (local)")
    if enable_openrouter:
        print(f"   └─ Layer 2: OpenRouter ({openrouter_model})")

    # Step 1: Fetch news
    headlines = fetch_news(ticker, days=days)
    if not headlines:
        return None

    # Step 2: FinBERT scoring (always)
    headlines = score_with_finbert(headlines)

    # Step 3: OpenRouter LLM analysis (optional)
    if enable_openrouter:
        api_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            headlines = analyze_with_llm(headlines, api_key, model=openrouter_model)
        else:
            print("   [WARN] OpenRouter enabled but no API key found. Skipping LLM analysis.")

    # Step 4: Save results
    summary = save_sentiment_report(headlines, ticker, output_dir)

    if summary:
        print(f"   ✅ {ticker}: {summary['overall_sentiment']} "
              f"(avg={summary['avg_score']:.3f}, "
              f"+{summary['pct_positive']}% / -{summary['pct_negative']}%)")

    return summary
