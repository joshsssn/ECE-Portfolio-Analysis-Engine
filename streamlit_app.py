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
# We import run_from_screener function directly
try:
    from run_from_screener import run_from_screener, MAX_STOCKS
except ImportError:
    st.error("Could not import 'run_from_screener.py'. Please make sure it is in the same directory.")
    st.stop()

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

# =============================================================================
# MAIN INTERFACE
# =============================================================================
uploaded_file = st.file_uploader("Choose a Screener CSV file", type=['csv'])

if uploaded_file is not None:
    # Display preview
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Screener Preview")
        st.dataframe(df.head())
        st.write(f"Total rows: {len(df)}")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

    # Run Button
    if st.button("🚀 Run Analysis", type="primary"):
        # Save uploaded file temporarily
        temp_dir = current_dir / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "uploaded_screener.csv"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
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
                # Execute analysis
                orchestrator, output_dir = run_from_screener(
                    csv_path=str(temp_path),
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
