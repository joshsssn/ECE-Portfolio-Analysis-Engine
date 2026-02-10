import streamlit as st
import pandas as pd
import sys
import shutil
import os
from pathlib import Path
import time
import zipfile
import html
import streamlit.components.v1 as components

# Add current directory to path to allow imports of local modules
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the analysis module
try:
    from run_from_screener import run_from_screener, MAX_STOCKS
    from config import AnalysisConfig
    from portfolio_loader import load_holdings_csv, load_sector_targets_csv, DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS
except ImportError:
    st.error("Could not import modules. Please make sure they are in the same directory.")
    st.stop()

# Instantiate default config to get default values
default_config = AnalysisConfig()

# =============================================================================
# UI CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ECE Portfolio Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ECE Portfolio Analysis Engine")
st.markdown("""
Upload a screener result CSV file to run the full analysis pipeline.
You can configure the analysis parameters in the sidebar.
""")

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.header("Configuration")

# Top N Selection
use_all_stocks = st.sidebar.checkbox("Analyze ALL stocks", value=False, help="Uncheck to limit the number of stocks.")
if use_all_stocks:
    top_n = None
else:
    top_n = st.sidebar.number_input("Top N Stocks", min_value=1, max_value=500, value=5, step=1)

st.sidebar.markdown("---")

# Analysis Steps
st.sidebar.subheader("Pipeline Steps")
run_portfolio = not st.sidebar.checkbox("Skip Portfolio Reconstruction", value=False)
run_optimal = not st.sidebar.checkbox("Skip Optimal Allocation", value=False)
run_backtests = not st.sidebar.checkbox("Skip Backtests", value=False)
run_valuation = not st.sidebar.checkbox("Skip Valuation", value=False)

only_valuation = st.sidebar.checkbox("Only Run Valuation", value=False, help="Overrides above skip settings")
if only_valuation:
    run_portfolio = False
    run_optimal = False
    run_backtests = False
    run_valuation = True

st.sidebar.markdown("---")

# Multi-Allocation
enable_multi_alloc = st.sidebar.checkbox("Enable Multi-Allocation Analysis", value=False)
multi_alloc_step = st.sidebar.number_input("Step Granularity (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, disabled=not enable_multi_alloc)

st.sidebar.markdown("---")

# Advanced Configuration
with st.sidebar.expander("Advanced Configuration"):
    st.markdown("### Portfolio Parameters")
    risk_aversion = st.slider("Risk Aversion (λ)", 0.5, 10.0, float(default_config.risk_aversion), 0.1)
    conc_penalty = st.slider("Concentration Penalty (γ)", 0.0, 50.0, float(default_config.concentration_penalty), 0.1)
    
    st.markdown("### Allocation Constraints")
    min_rec_alloc = st.number_input("Min Allocation (%)", 0.0, 50.0, float(default_config.min_recommended_allocation * 100), 0.5) / 100.0
    # min_alloc removed as per user request (defaults to 0 in config)
    max_alloc = st.number_input("Max Allocation (%)", 0.0, 100.0, float(default_config.max_allocation * 100), 1.0) / 100.0
    
    st.markdown("### Market Parameters")
    risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, float(default_config.risk_free_rate * 100), 0.1) / 100.0
    benchmark = st.text_input("Benchmark Ticker", default_config.benchmark_ticker)
    tech_etf = st.text_input("Correlation Ticker", default_config.tech_etf_ticker)
    
    st.markdown("### Simulation Settings")
    lookback = st.number_input("Lookback Years", 1, 20, default_config.lookback_years)
    resample = st.selectbox("Resample Frequency", ["W", "D", "M"], index=["W", "D", "M"].index(default_config.resample_freq))
    n_sims = st.number_input("Monte Carlo Sims", min_value=1, max_value=1000000000, value=default_config.n_simulations, step=1000)

# Sprint 1 Features - Risk Management
with st.sidebar.expander("🛡️ Risk Management (NEW)", expanded=False):
    st.markdown("### Drawdown Protection")
    enable_drawdown_protection = st.checkbox("Enable Drawdown Protection", value=True, 
        help="Automatically reduce allocation when portfolio is in drawdown (Bridgewater-style)")
    dd_threshold = st.slider("Drawdown Threshold (%)", 5.0, 30.0, 
        float(default_config.drawdown_reduction_threshold * 100), 1.0,
        help="Drawdown level that triggers position reduction") / 100.0
    dd_reduction = st.slider("Reduction Factor", 0.1, 1.0, 
        float(default_config.drawdown_reduction_factor), 0.1,
        help="Multiply allocation by this factor during deep drawdown")
    
    st.markdown("### Stress Testing")
    enable_stress_test = st.checkbox("Run Stress Tests", value=True,
        help="Apply historical crisis scenarios (2008, COVID) to measure tail risk")
    stress_portfolio_value = st.number_input("Portfolio Value ($)", 
        min_value=10000, max_value=1000000000, value=1000000, step=100000,
        help="Used for stress test dollar loss calculations")
    
    st.markdown("### Covariance Estimation")
    use_ledoit_wolf = st.checkbox("Use Ledoit-Wolf Shrinkage", value=True,
        help="Stabilizes covariance matrix for better allocation (recommended)")

# Sprint 1 Features - Rebalancing
with st.sidebar.expander("⚖️ Rebalancing (NEW)", expanded=False):
    enable_rebalancing = st.checkbox("Generate Rebalancing Orders", value=False,
        help="Generate executable trade orders to reach target weights")
    if enable_rebalancing:
        current_portfolio_value = st.number_input("Current Portfolio Value ($)", 
            min_value=1000, max_value=1000000000, value=100000, step=10000)
        min_trade_value = st.number_input("Min Trade Size ($)", 
            min_value=10, max_value=10000, value=100, step=50,
            help="Ignore trades smaller than this")
        round_to_lots = st.checkbox("Round to 100-share lots", value=False,
            help="For institutional compatibility")
    else:
        # Defaults when rebalancing is disabled
        current_portfolio_value = 100000
        min_trade_value = 100
        round_to_lots = False
        
# Sprint 0 Features - FinOracle (Forecasting & Sentiment) 
with st.sidebar.expander("🔮 FinOracle (Forecasting & AI)", expanded=False):
    
    st.markdown("### 🧠 Forecasting")
    enable_finoracle = st.checkbox("Enable Forecasting Engine", value=False,
        help="Run AI-powered price forecasting using tiered models")
    
    # Model Selection (Multi-Select)
    fo_models = st.multiselect(
        "Select Models",
        options=['arimax', 'lstm', 'gru', 'xgboost', 'random_forest', 'transformer', 'fts'],
        default=['arimax', 'lstm', 'fts'],
        disabled=not enable_finoracle,
        help="Choose which models to run."
    )
    
    enable_ensemble = st.checkbox("Enable Ensemble Average", value=True,
        help="Calculate an average forecast from all selected models.", disabled=not enable_finoracle)
    
    st.markdown("##### 📡 Data Fetching (Refinitiv)")
    fo_freq = st.selectbox("Frequency", ["d", "w", "m", "1h", "5min", "1min", "tick"],
        index=0, help="Data frequency for fetching & inference", disabled=not enable_finoracle)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fo_days = st.number_input("Days (N)", min_value=0, value=0, step=30,
            help="Fetch last N days. Set 0 to use Years instead.", disabled=not enable_finoracle)
        fo_days = fo_days if fo_days > 0 else None
    with col_d2:
        fo_years = st.number_input("Years", min_value=1, max_value=20, value=5, step=1,
            help="Fetch last N years (ignored if Days > 0)", disabled=not enable_finoracle)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fo_start = st.text_input("Start (YYYY-MM-DD)", value="",
            help="Specific start date. Overrides Years.", disabled=not enable_finoracle)
        fo_start = fo_start if fo_start.strip() else None
    with col_s2:
        fo_end = st.text_input("End (YYYY-MM-DD)", value="",
            help="Specific end date. Default: today.", disabled=not enable_finoracle)
        fo_end = fo_end if fo_end.strip() else None
    
    fo_skip_fetch = st.checkbox("Skip Fetch (reuse cached data)", value=False,
        help="Skip Refinitiv download — reuse existing local CSVs", disabled=not enable_finoracle)
    
    st.markdown("### 📰 Sentiment Analysis")
    enable_sentiment = st.checkbox("Enable FinBERT Sentiment", value=False,
        help="Analyze news headlines for sentiment scores (Independent of forecasting)")
    
    sentiment_days = 30
    enable_openrouter = False
    openrouter_model = 'openrouter/free'
    openrouter_key = None
    
    if enable_sentiment:
        sentiment_days = st.slider("News Lookback (days)", min_value=7, max_value=90, value=30, step=7,
            help="How many days of news headlines to analyze")
        enable_openrouter = st.checkbox("Enable OpenRouter LLM (deep analysis)", value=False,
            help="Use a free LLM via OpenRouter API for full article analysis (1000 req/day free)")
        if enable_openrouter:
            openrouter_model = st.selectbox("OpenRouter Model", [
                'openrouter/free',
                'openai/gpt-oss-20b:free',
                'nvidia/nemotron-3-nano-30b-a3b:free',
                'tngtech/deepseek-r1t2-chimera:free',
                'meta-llama/llama-3.3-70b-instruct:free',
            ], index=0, help="Free model to use for deep article analysis")
            openrouter_key = st.text_input("OpenRouter API Key", type="password",
                value=os.environ.get("OPENROUTER_API_KEY", ""),
                help="Get yours at openrouter.ai/keys")

    st.markdown("##### ⚙️ Forecast Settings")
    fo_horizon = st.number_input("Forecast Horizon (H)", min_value=1, max_value=256, value=16, step=1,
        help="Number of future steps to predict. Applies to ALL selected models.", disabled=not enable_finoracle)

    # FTS Specifics (Hidden if FTS not selected to clean UI)
    if 'fts' in fo_models and enable_finoracle:
        st.markdown("##### ⚙️ FTS Model Config")
        fo_context = st.number_input("Context Length (L)", min_value=32, max_value=1024, value=128, step=32,
            help="Number of past data points FTS uses as context.")
        
        fo_gpu = st.checkbox("Use GPU", value=True)
        fo_optimize = st.checkbox("AutoML Hyperopt", value=False)
        
        if fo_optimize:
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                fo_trials = st.number_input("Trials", min_value=5, value=20)
            with col_o2:
                fo_folds = st.number_input("Folds", min_value=2, value=3)
        else:
            fo_trials, fo_folds = 20, 3
    else:
        # Defaults
        fo_context = 128
        fo_gpu, fo_optimize = True, False
        fo_trials, fo_folds = 20, 3

    fo_skip_inference = st.checkbox("Skip Inference (re-visualize)", value=False,
        help="Skip model run entirely. Use to re-display old results.", disabled=not enable_finoracle)

st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Data")
uploaded_holdings = st.sidebar.file_uploader("Upload Holdings CSV", type=['csv'])
uploaded_sectors = st.sidebar.file_uploader("Upload Sector Targets CSV", type=['csv'])

# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.subheader("Screener Input")
use_default_screener = st.checkbox(
    "📋 Use Default Screener (for testing)", 
    value=False,
    help="Check this if you don't have a screener result file. Uses a default set of healthcare stocks."
)

if use_default_screener:
    st.info("ℹ️ Using default screener results. This includes a sample of healthcare stocks for testing.")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("Choose a Screener CSV file", type=['csv'])

if uploaded_file is not None or use_default_screener:
    # Determine which file to use
    if use_default_screener:
        default_screener_path = current_dir / "default" / "default_screener_results.csv"
        if not default_screener_path.exists():
            st.error(f"Default screener file not found at: {default_screener_path}")
            st.stop()
        file_to_read = default_screener_path
        file_source = "default"
    else:
        file_to_read = uploaded_file
        file_source = "uploaded"
    
    # Display preview
    try:
        if file_source == "default":
            df = pd.read_csv(file_to_read)
        else:
            df = pd.read_csv(file_to_read)
        
        st.subheader("Screener Preview")
        st.dataframe(df.head())
        st.write(f"Total rows: {len(df)}")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

    # Run Button
    if st.button("🚀 Run Analysis", type="primary"):
        # Save file temporarily
        temp_dir = current_dir / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "uploaded_screener.csv"
        
        if file_source == "default":
            # Copy default file
            shutil.copy(file_to_read, temp_path)
        else:
            # Save uploaded file
            with open(temp_path, "wb") as f:
                f.write(file_to_read.getbuffer())
        
        # Prepare inputs
        multi_alloc_val = (multi_alloc_step / 100.0) if enable_multi_alloc else None
        
        st.markdown("**📜 Execution Log**")
        log_placeholder = st.empty()
        
        class RealTimeLogger:
            def __init__(self, placeholder):
                self.terminal = sys.stdout
                self.placeholder = placeholder
                self.log_buffer = ""
                # Initialize session state log if needed
                if 'execution_log' not in st.session_state:
                    st.session_state['execution_log'] = ""

            def write(self, message):
                self.terminal.write(message)
                self.log_buffer += message
                st.session_state['execution_log'] = self.log_buffer # Sync to session state
                
                # Use an iframe with smart scrolling via sessionStorage
                html_content = f"""
                <html>
                <head>
                    <style>
                        body {{
                            background-color: #1e1e1e;
                            color: #e0e0e0;
                            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                            font-size: 16px;
                            margin: 0;
                            padding: 10px;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                        }}
                        /* Custom scrollbar */
                        ::-webkit-scrollbar {{
                            width: 10px;
                            height: 10px;
                        }}
                        ::-webkit-scrollbar-track {{
                            background: #1e1e1e; 
                        }}
                        ::-webkit-scrollbar-thumb {{
                            background: #555; 
                            border-radius: 5px;
                        }}
                        ::-webkit-scrollbar-thumb:hover {{
                            background: #888; 
                        }}
                    </style>
                </head>
                <body>
                    <div id="log-content">{html.escape(self.log_buffer)}</div>
                    <script>
                        const scrollKey = 'streamlit_log_scroll_pos';
                        const bottomKey = 'streamlit_log_at_bottom';
                        let isAutoScrolling = false;
                        
                        function restoreScroll() {{
                            const storedAtBottom = sessionStorage.getItem(bottomKey);
                            const storedPos = sessionStorage.getItem(scrollKey);
                            
                            // Default to true (at bottom) if simple or null
                            const isAtBottom = storedAtBottom === null || storedAtBottom === 'true';
                            
                            if (isAtBottom) {{
                                isAutoScrolling = true;
                                window.scrollTo(0, document.body.scrollHeight);
                                setTimeout(() => {{ isAutoScrolling = false; }}, 50); // Release lock
                            }} else if (storedPos) {{
                                window.scrollTo(0, parseInt(storedPos));
                            }}
                        }}

                        function saveScroll() {{
                            if (isAutoScrolling) return; // Ignore script-triggered scrolls
                            
                            const scrollTop = window.scrollY || document.documentElement.scrollTop || document.body.scrollTop;
                            const scrollHeight = document.documentElement.scrollHeight || document.body.scrollHeight;
                            const clientHeight = document.documentElement.clientHeight || window.innerHeight;
                            
                            // Check if at bottom (with 50px buffer to be generous)
                            const atBottom = (scrollTop + clientHeight) >= (scrollHeight - 50);
                            
                            sessionStorage.setItem(scrollKey, scrollTop);
                            sessionStorage.setItem(bottomKey, atBottom);
                        }}

                        // Restore scroll position
                        restoreScroll();
                        
                        // Listen for scroll events to save state, but wait a moment for initial layout
                        setTimeout(() => {{
                            window.addEventListener('scroll', saveScroll);
                            window.addEventListener('resize', saveScroll);
                        }}, 100);
                    </script>
                </body>
                </html>
                """
                # Render the iframe
                with self.placeholder.container():
                     components.html(html_content, height=600, scrolling=True)


            def flush(self):
                self.terminal.flush()

        # Capture logic
        original_stdout = sys.stdout
        
        try:
            # Clear previous log
            st.session_state['execution_log'] = ""
            
            # Create logger
            logger = RealTimeLogger(log_placeholder)
            sys.stdout = logger
            
            with st.spinner('Running analysis pipeline...'):
                # 1. Setup Configuration
                config = AnalysisConfig(
                    risk_aversion=risk_aversion,
                    concentration_penalty=conc_penalty,
                    min_recommended_allocation=min_rec_alloc,
                    # min_allocation defaults to 0.0 in config.py
                    max_allocation=max_alloc,
                    benchmark_ticker=benchmark,
                    risk_free_rate=risk_free,
                    lookback_years=lookback,
                    resample_freq=resample,
                    tech_etf_ticker=tech_etf,
                    n_simulations=n_sims,
                    # Sprint 1: Drawdown Protection
                    drawdown_reduction_threshold=dd_threshold if enable_drawdown_protection else 1.0,
                    drawdown_reduction_factor=dd_reduction if enable_drawdown_protection else 1.0,
                    # Sprint 1: Ledoit-Wolf
                    use_ledoit_wolf=use_ledoit_wolf,
                    # FinOracle Config (all flags)
                    enable_finoracle=enable_finoracle,
                    enable_ensemble=enable_ensemble,
                    enable_sentiment=enable_sentiment, # New independent flag
                    sentiment_days=sentiment_days,
                    enable_openrouter=enable_openrouter,
                    openrouter_model=openrouter_model,
                    openrouter_api_key=openrouter_key,
                    finoracle_models=fo_models, # New model list
                    finoracle_freq=fo_freq,
                    finoracle_days=fo_days,
                    finoracle_years=fo_years,
                    finoracle_start=fo_start,
                    finoracle_end=fo_end,
                    finoracle_skip_fetch=fo_skip_fetch,
                    finoracle_context_len=fo_context,
                    finoracle_horizon_len=fo_horizon,
                    finoracle_optimize=fo_optimize,
                    finoracle_trials=fo_trials,
                    finoracle_folds=fo_folds,
                    finoracle_use_gpu=fo_gpu,
                    finoracle_skip_inference=fo_skip_inference
                )
                
                # Store Sprint 1 options for pipeline use
                sprint1_options = {
                    'enable_stress_test': enable_stress_test,
                    'stress_portfolio_value': stress_portfolio_value,
                    'use_ledoit_wolf': use_ledoit_wolf,
                    'enable_rebalancing': enable_rebalancing,
                    'rebalancing_portfolio_value': current_portfolio_value if enable_rebalancing else None,
                    'min_trade_value': min_trade_value if enable_rebalancing else None,
                    'round_to_lots': round_to_lots if enable_rebalancing else False,
                }
                
                # 2. Load Portfolio Data
                holdings = DEFAULT_TOP_HOLDINGS
                if uploaded_holdings:
                    try:
                        # Save temp
                        h_path = temp_dir / "uploaded_holdings.csv"
                        with open(h_path, "wb") as f:
                            f.write(uploaded_holdings.getbuffer())
                        holdings = load_holdings_csv(str(h_path))
                        st.success(f"Loaded {len(holdings)} holdings from CSV")
                    except Exception as e:
                        st.error(f"Error loading holdings CSV: {e}")
                
                sector_targets = DEFAULT_SECTOR_TARGETS
                if uploaded_sectors:
                    try:
                        # Save temp
                        s_path = temp_dir / "uploaded_sectors.csv"
                        with open(s_path, "wb") as f:
                            f.write(uploaded_sectors.getbuffer())
                        sector_targets = load_sector_targets_csv(str(s_path))
                        st.success(f"Loaded {len(sector_targets)} sector targets from CSV")
                    except Exception as e:
                        st.error(f"Error loading sectors CSV: {e}")

                # Execute analysis
                orchestrator, output_dir = run_from_screener(
                    csv_path=str(temp_path),
                    config=config,
                    holdings=holdings,
                    sector_targets=sector_targets,
                    top_n=top_n,
                    run_portfolio=run_portfolio,
                    run_optimal=run_optimal,
                    run_backtests=run_backtests,
                    run_valuations=run_valuation,
                    multi_alloc_granularity=multi_alloc_val,
                    sprint1_options=sprint1_options,
                    run_forecasting=enable_finoracle
                )
            
            st.success("Analysis Complete!")
            
            # Save state for persistence
            st.session_state['latest_run'] = {
                'output_dir': output_dir,
                'enable_finoracle': enable_finoracle,
                'timestamp': int(time.time())
            }

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
            st.exception(e)
        finally:
            # Restore stdout
            sys.stdout = original_stdout

# =============================================================================
# PERSISTENT RESULTS DISPLAY
# =============================================================================
if 'latest_run' in st.session_state:
    try:
        run_data = st.session_state['latest_run']
        output_dir = run_data['output_dir']
        enable_finoracle = run_data['enable_finoracle']
        ts = run_data['timestamp']

        # Ensure output_dir is a Path object
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if output_dir.exists():
            # --- FinOracle Results Display ---
            if enable_finoracle:
                st.markdown("---")
                st.subheader("🔮 FinOracle Forecasts")
                
                finoracle_files = []
                for stock_folder in output_dir.iterdir():
                   if stock_folder.is_dir():
                       fo_dir = stock_folder / "finoracle"
                       if fo_dir.exists():
                           finoracle_files.append(fo_dir)
                           stock_ticker = stock_folder.name
                           plots = list(fo_dir.glob("*_forecast.png"))
                           if plots:
                               with st.expander(f"Forecast: {stock_ticker}", expanded=False):
                                   st.image(str(plots[0]), caption=f"{stock_ticker} Forecast", use_container_width=True)

                if not finoracle_files:
                    st.info("No FinOracle results found in output directory.")
            
            # --- Sentiment Analysis Insights ---
            st.markdown("---")
            st.subheader("📊 Sentiment Analysis Insights")
            
            sentiment_results = {}
            for stock_folder in output_dir.iterdir():
                if stock_folder.is_dir():
                    sentiment_dir = stock_folder / "sentiment"
                    if sentiment_dir.exists():
                        ticker = stock_folder.name
                        csv_file = sentiment_dir / f"{ticker}_sentiment.csv"
                        plot_file_finbert = sentiment_dir / f"{ticker}_sentiment.png"
                        plot_file_llm = sentiment_dir / f"{ticker}_sentiment_llm.png"
                        
                        if csv_file.exists():
                            try:
                                df = pd.read_csv(csv_file)
                                sentiment_results[ticker] = {
                                    'df': df,
                                    'plot_finbert': plot_file_finbert if plot_file_finbert.exists() else None,
                                    'plot_llm': plot_file_llm if plot_file_llm.exists() else None
                                }
                            except Exception as e:
                                st.warning(f"Could not load sentiment data for {ticker}: {e}")
            
            if sentiment_results:
                for ticker, data in sentiment_results.items():
                    df = data['df']
                    plot_finbert = data['plot_finbert']
                    plot_llm = data['plot_llm']
                    
                    # Calculate summary stats
                    n_total = len(df)
                    n_ai_analyzed = df['llm_summary'].notna().sum() if 'llm_summary' in df.columns else 0
                    avg_sentiment = df['finbert_score'].mean() if 'finbert_score' in df.columns else 0
                    
                    # Determine overall sentiment
                    if avg_sentiment > 0.15:
                        sentiment_emoji = "🟢 BULLISH"
                        sentiment_color = "green"
                    elif avg_sentiment < -0.15:
                        sentiment_emoji = "🔴 BEARISH"
                        sentiment_color = "red"
                    else:
                        sentiment_emoji = "🟡 NEUTRAL"
                        sentiment_color = "orange"
                    
                    with st.expander(f"{ticker} — {sentiment_emoji} (avg: {avg_sentiment:.3f})", expanded=True):
                        # Show plots in tabs if both exist, otherwise just show FinBERT
                        if plot_llm:
                            tab1, tab2 = st.tabs(["📰 FinBERT (Headlines)", "🤖 AI/LLM (Deep Analysis)"])
                            with tab1:
                                if plot_finbert:
                                    st.image(str(plot_finbert), caption=f"{ticker} FinBERT Sentiment", use_container_width=True)
                            with tab2:
                                st.image(str(plot_llm), caption=f"{ticker} AI Deep Sentiment", use_container_width=True)
                        elif plot_finbert:
                            st.image(str(plot_finbert), caption=f"{ticker} FinBERT Sentiment", use_container_width=True)
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Headlines Analyzed", n_total)
                        with col2:
                            st.metric("AI Deep Analysis", f"{n_ai_analyzed}/{n_total}")
                        with col3:
                            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}", delta_color="normal")
                        
                        # AI Insights Table (only if LLM analysis was done)
                        if n_ai_analyzed > 0 and 'llm_summary' in df.columns:
                            st.markdown("##### 🤖 AI-Generated Insights")
                            
                            # Filter to only articles with AI analysis
                            ai_df = df[df['llm_summary'].notna()].copy()
                            
                            # Prepare display dataframe
                            display_cols = ['date', 'title']
                            if 'llm_sentiment' in ai_df.columns:
                                display_cols.append('llm_sentiment')
                            if 'llm_summary' in ai_df.columns:
                                display_cols.append('llm_summary')
                            if 'llm_risks' in ai_df.columns:
                                display_cols.append('llm_risks')
                            if 'llm_opportunities' in ai_df.columns:
                                display_cols.append('llm_opportunities')
                            
                            # Clean up column names for display
                            display_df = ai_df[display_cols].copy()
                            display_df.columns = [c.replace('llm_', '').replace('_', ' ').title() for c in display_df.columns]
                            
                            # Parse stringified lists back to python lists
                            import ast
                            def safe_parse_list(val):
                                try:
                                    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                                        return ast.literal_eval(val)
                                    return val
                                except:
                                    return val

                            if 'Risks' in display_df.columns:
                                display_df['Risks'] = display_df['Risks'].apply(safe_parse_list)
                            if 'Opportunities' in display_df.columns:
                                display_df['Opportunities'] = display_df['Opportunities'].apply(safe_parse_list)

                            # Display as interactive table
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Sentiment": st.column_config.NumberColumn(
                                        "AI Sentiment",
                                        help="LLM sentiment score (-1 to +1)",
                                        format="%.3f"
                                    ),
                                    "Summary": st.column_config.TextColumn(
                                        "Summary",
                                        help="AI-generated article summary",
                                        width="large"
                                    ),
                                    "Risks": st.column_config.ListColumn(
                                        "Key Risks",
                                        help="Risks identified by AI"
                                    ),
                                    "Opportunities": st.column_config.ListColumn(
                                        "Key Opportunities",
                                        help="Opportunities identified by AI"
                                    )
                                }
                            )
                        else:
                            st.info("No AI deep analysis available for this ticker. Enable OpenRouter in settings to get detailed insights.")
            else:
                st.info("No sentiment analysis results found. Enable sentiment analysis in the sidebar to see insights.")

            
            # --- Main Download Button ---
            st.markdown("---")
            # Create ZIP archive (cached by timestamp to avoid re-zipping on every rerun)
            zip_name = f"analysis_results_{ts}"
            zip_path = current_dir / f"{zip_name}.zip"
            
            # Check if zip already exists for this run, else create it
            if not zip_path.exists():
                with st.spinner("Preparing download archive..."):
                     shutil.make_archive(str(current_dir / zip_name), 'zip', output_dir)
            
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="📥 Download Results (ZIP)",
                    data=f,
                    file_name=f"{zip_name}.zip",
                    mime="application/zip",
                    key="download_results_persistent"
                )
            
            # --- Persistent Execution Log ---
            if 'execution_log' in st.session_state and st.session_state['execution_log']:
                with st.expander("📜 Previous Execution Log", expanded=False):
                     st.text_area("Log Output", st.session_state['execution_log'], height=400, disabled=True)

        else:
            st.warning("Output directory from previous run no longer exists.")
            del st.session_state['latest_run'] # clear stale state

    except Exception as e:
        st.error(f"Error displaying cached results: {e}")


else:
    st.info("Please upload a CSV file to proceed.")


# =============================================================================
# HELP SECTION
# =============================================================================
st.markdown("---")
with st.expander("📖 Help & Documentation"):
    md_dir = current_dir / "MD"
    
    if md_dir.exists():
        # Get list of .md files
        md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
        
        if md_files:
            # Sort files for consistent display
            md_files.sort()
            
            # Simple radio or selectbox for navigation
            selected_md = st.selectbox(
                "Select a documentation page:",
                md_files,
                index=md_files.index("UI.md") if "UI.md" in md_files else 0
            )
            
            # Read and display the selected file
            md_path = md_dir / selected_md
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                st.markdown(md_content)
            except Exception as e:
                st.error(f"Error reading {selected_md}: {e}")
        else:
            st.warning("No markdown files found in MD folder.")
    else:
        # Fallback to UI.md in root if MD doesn't exist (safety)
        readme_path = current_dir / "UI.md"
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            st.markdown(readme_content)
        else:
            st.error("Documentation folder (MD/) not found.")
