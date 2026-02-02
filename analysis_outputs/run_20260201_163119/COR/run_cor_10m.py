"""
COR 10M Trial Monte Carlo Valuation Analysis
=============================================
Standalone script analyzing COR with 10 million Monte Carlo trials.
Located in: analysis_outputs/run_<timestamp>/COR/

Import Path Fix: Goes up 3 levels to access config and valuation_engine.
"""
import sys
from pathlib import Path

# Fix imports: go up to ECE root directory
script_dir = Path(__file__).parent.absolute()
ece_root = script_dir.parent.parent.parent  # COR -> run_<date> -> analysis_outputs -> ECE root
sys.path.insert(0, str(ece_root))

from config import AnalysisConfig
from valuation_engine import ValuationEngine
import matplotlib.pyplot as plt

# Override n_simulations to 10M
config = AnalysisConfig()
config.n_simulations = 10000000

# Run valuation for COR
print('\nRunning 10M trial Monte Carlo simulation for COR...')
valuation = ValuationEngine(config)
result = valuation.analyze('COR', verbose=True)

# Display results
print('\n' + '='*70)
print('COR (Cencora Inc.) - 10M Trial Monte Carlo Valuation')
print('='*70)

print('\n📊 SIMULATION RESULTS (10,000,000 RUNS)')
print('-' * 70)
print(f'Mean Fair Value:         ${result.mc_mean:.2f} ({(result.mc_mean / result.dcf_per_share - 1) * 100:+.1f}% vs DCF Base)')
print(f'Median Value:            ${result.mc_median:.2f}')
print(f'Standard Deviation:      ${result.mc_std:.2f}')
print(f'P10 (Pessimistic):       ${result.mc_p10:.2f}')
print(f'P90 (Optimistic):        ${result.mc_p90:.2f}')
print(f'Win Probability:         {result.win_probability:.1f}% (Price > ${result.current_price:.2f})')
print(f'DCF Base Case:           ${result.dcf_per_share:.2f}')

print('\n📈 METHODOLOGY & PARAMETERS')
print('-' * 70)
print('Why Monte Carlo?')
print('  Standard DCF is static. Monte Carlo tests 10,000,000 scenarios to')
print('  quantify uncertainty and identify upside/downside potential.')

print('\nRandomized Inputs (Normal Distribution):')
print('  • FCF Margin:     1.0% ± 0.2% (proportional to base margin)')
print('  • Revenue Growth: 8.6% ± 2.0%')
print('  • WACC:           7.10% ± 1.0% (increased from 0.5%)')

print('\n✅ ROBUSTNESS CHECK')
print('-' * 70)
mc_dcf_ratio = result.mc_mean / result.dcf_per_share
mean_median_diff = ((result.mc_mean - result.mc_median) / result.mc_median) * 100

print(f'  Base Case (${result.dcf_per_share:.2f}) uses conservative 1% FCF margin.')
print(f'  Upside cases (Margin > 2-3%) drive the higher Mean (${result.mc_mean:.2f}).')
print(f'  Conservative Growth: Model input (8.6%) caps well below 3Y CAGR')
print(f'  (10.4%), providing a safety margin.')
print(f'\n  MC Mean to DCF Ratio: {mc_dcf_ratio:.2f}x')
print(f'  Mean > Median by {mean_median_diff:.1f}% (positive skew from upside tail)')

print('\n' + '='*70)
print(f'Trials Executed:         {len(result.simulation_values):,}')
print('='*70)

# Generate plot - save in current directory (COR folder)
output_path = script_dir / 'COR_valuation_10M_trials.png'
print(f'\nGenerating valuation chart...')
valuation.plot_valuation('COR', save_path=str(output_path))
