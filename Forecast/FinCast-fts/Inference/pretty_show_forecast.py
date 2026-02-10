"""
Pretty Forecast Viewer — Terminal table + dual plots (forecast-only + context+forecast).

Usage:
    python pretty_show_forecast.py                        # auto-select latest CSV
    python pretty_show_forecast.py path/to/forecast.csv   # specific CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
import sys


def _find_latest_forecast(base_dir: str) -> str | None:
    """Find the most recently modified fincast_full_*.csv in base_dir or base_dir/output/."""
    # 1. Check base_dir directly
    pattern = os.path.join(base_dir, "fincast_full_*.csv")
    files = glob.glob(pattern)
    
    # 2. If nothing, check base_dir/output
    if not files:
        output_dir = os.path.join(base_dir, "output")
        pattern = os.path.join(output_dir, "fincast_full_*.csv")
        files = glob.glob(pattern)
        
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _find_data_csv(base_dir: str) -> str | None:
    """Find data.csv (the input data used for inference)."""
    path = os.path.join(base_dir, "data.csv")
    return path if os.path.exists(path) else None


def _load_forecast(csv_path: str) -> tuple:
    """Load forecast CSV and parse into a clean DataFrame."""
    df = pd.read_csv(csv_path, index_col=0)
    row_name = df.index[0]
    data = df.iloc[0]

    days = sorted(set(int(re.search(r't\+(\d+)', c).group(1)) for c in df.columns))
    stats = ['mean'] + [f'q{i}' for i in range(1, 10)]

    records = []
    for day in days:
        record = {'Day': day}
        for stat in stats:
            col_name = f"{stat}_t+{day}"
            if col_name in df.columns:
                record[stat.capitalize()] = data[col_name]
        records.append(record)

    forecast_df = pd.DataFrame(records).set_index('Day')
    return row_name, forecast_df, len(days)


def show_table(row_name: str, forecast_df: pd.DataFrame, horizon: int):
    """Print a pretty terminal table."""
    print("\n" + "=" * 80)
    print(f"       FINCAST FUTURE FORECAST: {row_name.upper()}")
    print("=" * 80)

    display_cols = ['Mean', 'Q1', 'Q3', 'Q5', 'Q7', 'Q9']
    valid_display = [c for c in display_cols if c in forecast_df.columns]
    print(forecast_df[valid_display].to_string(float_format=lambda x: f"{x:8.4f}"))

    start_price = forecast_df['Mean'].iloc[0]
    end_price = forecast_df['Mean'].iloc[-1]
    change_pct = ((end_price / start_price) - 1) * 100

    print("\n" + "-" * 40)
    print(f"SUMMARY ({horizon}-STEP OUTLOOK):")
    print(f"  Trend:        {'📈 UP' if change_pct > 0 else '📉 DOWN'}")
    print(f"  Expected:     {end_price:.4f} ({change_pct:+.2f}%)")
    if 'Q1' in forecast_df.columns and 'Q9' in forecast_df.columns:
        print(f"  Conf. Range:  [{forecast_df['Q1'].iloc[-1]:.4f} - {forecast_df['Q9'].iloc[-1]:.4f}] (80% confidence)")
    print("-" * 40)


def plot_forecast_only(row_name: str, forecast_df: pd.DataFrame, save_path: str):
    """Plot forecast with confidence bands (original style)."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    x = forecast_df.index.values

    # Confidence bands
    if 'Q1' in forecast_df.columns and 'Q9' in forecast_df.columns:
        ax.fill_between(x, forecast_df['Q1'], forecast_df['Q9'],
                        color='#00d2ff', alpha=0.08, label='80% CI (Q1–Q9)')
    if 'Q3' in forecast_df.columns and 'Q7' in forecast_df.columns:
        ax.fill_between(x, forecast_df['Q3'], forecast_df['Q7'],
                        color='#00d2ff', alpha=0.18, label='40% CI (Q3–Q7)')

    # Median & Mean
    if 'Q5' in forecast_df.columns:
        ax.plot(x, forecast_df['Q5'], color='#a8dadc', linestyle='--',
                linewidth=1.5, label='Median (Q5)')
    ax.plot(x, forecast_df['Mean'], color='#e63946', linewidth=2.5,
            marker='o', markersize=4, label='Mean Forecast')

    ax.set_title(f"Forecast: {row_name}", fontsize=16, fontweight='bold',
                 color='white', pad=15)
    ax.set_xlabel("Horizon (Steps)", fontsize=12, color='#ccc')
    ax.set_ylabel("Price / Value", fontsize=12, color='#ccc')
    ax.tick_params(colors='#aaa')
    ax.grid(True, linestyle=':', alpha=0.3, color='#555')
    ax.legend(loc='upper left', frameon=True, facecolor='#1a1a2e',
              edgecolor='#555', labelcolor='white')

    for spine in ax.spines.values():
        spine.set_color('#333')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"\n📊 Forecast plot saved to: {save_path}")
    plt.close()


def plot_context_and_forecast(row_name: str, forecast_df: pd.DataFrame,
                               data_csv_path: str, save_path: str,
                               context_len: int = 128):
    """
    Plot historical context + forecast continuation (like the notebook).
    Reads the input data.csv to get the last `context_len` data points,
    then appends the forecast.
    """
    try:
        input_df = pd.read_csv(data_csv_path)
    except Exception as e:
        print(f"⚠️  Could not load input data for context plot: {e}")
        return

    # Find the Close column
    close_col = None
    for candidate in ['Close', 'close', 'CLOSE', 'TRDPRC_1', 'MID_PRICE']:
        if candidate in input_df.columns:
            close_col = candidate
            break

    if close_col is None:
        print(f"⚠️  No 'Close' column found in {data_csv_path}")
        return

    close_values = pd.to_numeric(input_df[close_col], errors='coerce').dropna().values

    # Take last context_len points
    ctx = close_values[-context_len:]
    L = len(ctx)
    H = len(forecast_df)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    x_ctx = np.arange(L)
    x_fut = np.arange(L, L + H)

    # Historical context
    ax.plot(x_ctx, ctx, color='#4cc9f0', linewidth=1.8, label='Historical (Context)')

    # Mean forecast
    ax.plot(x_fut, forecast_df['Mean'].values, color='#e63946',
            linewidth=2.5, marker='o', markersize=4, label='Forecast (Mean)')

    # Quantile lines
    quantile_colors = {
        'Q1': ('#f77f00', ':', 'q1 (10th)'),
        'Q3': ('#fcbf49', '--', 'q3 (30th)'),
        'Q5': ('#a8dadc', '--', 'q5 (Median)'),
        'Q7': ('#90be6d', '--', 'q7 (70th)'),
        'Q9': ('#43aa8b', ':', 'q9 (90th)'),
    }
    for q_col, (color, ls, label) in quantile_colors.items():
        if q_col in forecast_df.columns:
            ax.plot(x_fut, forecast_df[q_col].values, color=color,
                    linestyle=ls, linewidth=1.3, label=label)

    # Confidence band fill
    if 'Q1' in forecast_df.columns and 'Q9' in forecast_df.columns:
        ax.fill_between(x_fut, forecast_df['Q1'].values, forecast_df['Q9'].values,
                        color='#00d2ff', alpha=0.07)

    # Vertical separator line
    ax.axvline(x=L - 0.5, color='#888', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(L - 1, ax.get_ylim()[1], ' Forecast →', color='#aaa',
            fontsize=9, va='top', ha='left')

    ax.set_title(f"{row_name}  |  Context={L}, Horizon={H}",
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel("Time (relative index)", fontsize=12, color='#ccc')
    ax.set_ylabel("Value", fontsize=12, color='#ccc')
    ax.tick_params(colors='#aaa')
    ax.grid(True, alpha=0.2, color='#555')
    ax.legend(loc='best', frameon=True, facecolor='#1a1a2e',
              edgecolor='#555', labelcolor='white', fontsize=9)

    for spine in ax.spines.values():
        spine.set_color('#333')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"📊 Context + Forecast plot saved to: {save_path}")
    plt.close()


def show_forecast(csv_path: str, context_len: int = 128):
    """Full pipeline: table + both plots."""
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    base_dir = os.path.dirname(os.path.abspath(csv_path))

    # Load forecast
    row_name, forecast_df, horizon = _load_forecast(csv_path)

    # 1. Terminal table
    show_table(row_name, forecast_df, horizon)

    # 2. Forecast-only plot
    forecast_plot_path = csv_path.replace('.csv', '_forecast.png')
    plot_forecast_only(row_name, forecast_df, forecast_plot_path)

    # 3. Context + Forecast plot (like the notebook)
    data_csv = _find_data_csv(base_dir)
    if data_csv:
        ctx_plot_path = csv_path.replace('.csv', '_context.png')
        plot_context_and_forecast(row_name, forecast_df, data_csv,
                                  ctx_plot_path, context_len)

        # 4. Zoomed plot (few history points + forecast)
        zoom_plot_path = csv_path.replace('.csv', '_zoomed.png')
        plot_zoomed_forecast(row_name, forecast_df, data_csv, zoom_plot_path)
    else:
        print("⚠️  data.csv not found — skipping context and zoomed plots.")


def plot_zoomed_forecast(row_name: str, forecast_df: pd.DataFrame,
                          data_csv_path: str, save_path: str,
                          tail_points: int = 10):
    """
    Zoomed-in plot: only the last `tail_points` of history + full forecast.
    Gives a clear view of the transition from real data to prediction.
    """
    try:
        input_df = pd.read_csv(data_csv_path)
    except Exception as e:
        print(f"⚠️  Could not load input data for zoomed plot: {e}")
        return

    # Find the Close column
    close_col = None
    for candidate in ['Close', 'close', 'CLOSE', 'TRDPRC_1', 'MID_PRICE']:
        if candidate in input_df.columns:
            close_col = candidate
            break

    if close_col is None:
        print(f"⚠️  No 'Close' column found in {data_csv_path}")
        return

    close_values = pd.to_numeric(input_df[close_col], errors='coerce').dropna().values

    # Take only the last tail_points
    tail = close_values[-tail_points:]
    T = len(tail)
    H = len(forecast_df)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    x_hist = np.arange(T)
    x_fut = np.arange(T, T + H)

    # Historical tail
    ax.plot(x_hist, tail, color='#4cc9f0', linewidth=2.5,
            marker='o', markersize=5, label='Recent History')

    # Mean forecast
    ax.plot(x_fut, forecast_df['Mean'].values, color='#e63946',
            linewidth=2.5, marker='o', markersize=5, label='Forecast (Mean)')

    # Connect history to forecast with a dashed line
    ax.plot([x_hist[-1], x_fut[0]],
            [tail[-1], forecast_df['Mean'].values[0]],
            color='#888', linestyle='--', linewidth=1.5)

    # Confidence bands
    if 'Q1' in forecast_df.columns and 'Q9' in forecast_df.columns:
        ax.fill_between(x_fut, forecast_df['Q1'].values, forecast_df['Q9'].values,
                        color='#00d2ff', alpha=0.10, label='80% CI (Q1–Q9)')
    if 'Q3' in forecast_df.columns and 'Q7' in forecast_df.columns:
        ax.fill_between(x_fut, forecast_df['Q3'].values, forecast_df['Q7'].values,
                        color='#00d2ff', alpha=0.20, label='40% CI (Q3–Q7)')

    # Median line
    if 'Q5' in forecast_df.columns:
        ax.plot(x_fut, forecast_df['Q5'].values, color='#a8dadc',
                linestyle='--', linewidth=1.5, label='Median (Q5)')

    # Vertical separator
    ax.axvline(x=T - 0.5, color='#f77f00', linestyle='-', linewidth=1.5, alpha=0.7)
    mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    ax.text(T - 0.3, ax.get_ylim()[1], '  Forecast →',
            color='#f77f00', fontsize=10, fontweight='bold', va='top', ha='left')

    ax.set_title(f"{row_name}  |  Zoomed View  |  Horizon={H}",
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel("Steps", fontsize=12, color='#ccc')
    ax.set_ylabel("Value", fontsize=12, color='#ccc')
    ax.tick_params(colors='#aaa')
    ax.grid(True, alpha=0.2, color='#555')
    ax.legend(loc='best', frameon=True, facecolor='#1a1a2e',
              edgecolor='#555', labelcolor='white', fontsize=9)

    for spine in ax.spines.values():
        spine.set_color('#333')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"🔍 Zoomed forecast plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Accept optional CLI argument for a specific CSV path
    # Accept optional CLI argument for a specific CSV path or FOLDER
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            # If folder provided, find latest forecast inside it
            csv_file = _find_latest_forecast(target)
            if not csv_file:
                # Fallback: check output/ subfolder
                csv_file = _find_latest_forecast(os.path.join(target, "output"))
                
            if not csv_file:
                 print(f"❌ No forecast files found in: {target}")
                 sys.exit(1)
        elif os.path.exists(target):
            csv_file = target
        else:
            print(f"❌ File/Folder not found: {target}")
            sys.exit(1)
    else:
        # Auto-select latest forecast
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = _find_latest_forecast(base_dir)
        if not csv_file:
            print(f"❌ No forecast files found in: {base_dir}")
            print("   Run the inference pipeline first.")
            sys.exit(1)

    print(f"📂 Using forecast: {os.path.basename(csv_file)}")
    show_forecast(csv_file)
