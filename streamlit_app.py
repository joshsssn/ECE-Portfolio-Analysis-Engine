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

            def write(self, message):
                self.terminal.write(message)
                self.log_buffer += message
                
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
                    n_simulations=n_sims
                )
                
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
                    multi_alloc_granularity=multi_alloc_val
                )
            
            st.success("Analysis Complete!")
            
            # Create ZIP archive
            st.write("Preparing download...")
            archive_name = "analysis_results"
            archive_path = shutil.make_archive(str(current_dir / archive_name), 'zip', output_dir)
            
            # Download Button
            with open(archive_path, "rb") as f:
                st.download_button(
                    label="📥 Download Results (ZIP)",
                    data=f,
                    file_name=f"analysis_results_{int(time.time())}.zip",
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
            st.exception(e)
        finally:
            # Restore stdout
            sys.stdout = original_stdout

else:
    st.info("Please upload a CSV file to proceed.")

# =============================================================================
# HELP SECTION
# =============================================================================
st.markdown("---")
with st.expander("📖 Help & Documentation"):
    readme_path = current_dir / "UI.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content)
    else:
        st.error("UI.md not found")
