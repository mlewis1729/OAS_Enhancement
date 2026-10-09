import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy import stats

from oas_simulation import (sample_covariance, oas_original, oas_extended,
                             rblw, ledoit_wolf)

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'CVX', 'PG', 'KO',
    'WMT', 'V', 'MA', 'UNH', 'HD', 'DIS', 'MCD', 'BAC', 'PFE', 'CSCO',
    'INTC', 'IBM', 'ABT', 'MRK', 'MMM', 'CAT', 'GE', 'HON', 'GS', 'AXP',
    'LMT', 'RTX', 'UPS', 'BA', 'WFC', 'MS', 'MO', 'PM', 'MDT',
    'AMGN', 'GILD', 'TGT', 'COST', 'NKE', 'SBUX', 'TXN', 'QCOM', 'T',
    'VZ', 'BRK-B',
]

IN_WINDOW  = 252
OUT_WINDOW = 21
STEP       = 21
START      = '2010-01-01'
END        = '2023-12-31'
STRESS_VIX = 20.0


def download_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    rets = np.log(raw / raw.shift(1)).iloc[1:]
    valid = rets.columns[rets.isna().mean() < 0.02]
    rets = rets[valid].dropna()

    vix_raw = yf.download('^VIX', start=start, end=end,
                          progress=False)['Close']
    vix = vix_raw.reindex(rets.index).ffill().bfill()

    return rets.values, list(rets.columns), pd.DatetimeIndex(rets.index), vix.values


def gmv_weights(Sigma):
    p = Sigma.shape[0]
    try:
        w = np.linalg.solve(Sigma, np.ones(p))
        return w / w.sum()
    except np.linalg.LinAlgError:
        return np.ones(p) / p


def compute_alpha_hat(X, S):
    tr_S2 = np.trace(S @ S)
    tr2_S = np.trace(S) ** 2
    return np.mean(np.sum(X ** 2, axis=1) ** 2) - tr2_S - 2 * tr_S2


def run_rolling(returns_matrix, vix_series, date_index):
    T, p = returns_matrix.shape
    n = IN_WINDOW

    window_dates, alphas, window_vix = [], [], []
    var_dict = {k: [] for k in ['Sample', 'LW', 'RBLW', 'OAS', 'OAS-Ext']}

    t = n
    while t + OUT_WINDOW <= T:
        X_in  = returns_matrix[t - n : t]
        X_out = returns_matrix[t : t + OUT_WINDOW]
        X_in  = X_in - X_in.mean(axis=0)

        S = sample_covariance(X_in)
        alphas.append(compute_alpha_hat(X_in, S))
        window_dates.append(date_index[t])
        window_vix.append(np.mean(vix_series[t - n : t]))

        estimators = {
            'Sample':  S,
            'LW':      ledoit_wolf(X_in),
            'RBLW':    rblw(S, n, p),
            'OAS':     oas_original(S, n, p),
            'OAS-Ext': oas_extended(X_in, S, n, p),
        }
        for name, Sigma_hat in estimators.items():
            w = gmv_weights(Sigma_hat)
            var_dict[name].append(np.var(X_out @ w, ddof=1))

        t += STEP

    return (window_dates, np.array(alphas),
            np.array(window_vix), var_dict)


def paired_test(var_a, var_b, label_a='OAS', label_b='OAS-Ext', n_boot=10000, seed=0):
    """Paired t-test and bootstrap CI for var_a - var_b (positive = b is better)."""
    d = np.array(var_a) - np.array(var_b)
    t_stat, p_val = stats.ttest_1samp(d, 0)
    ci_t = stats.t.interval(0.95, len(d) - 1,
                             loc=d.mean(), scale=stats.sem(d))
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, size=len(d), replace=True).mean()
            for _ in range(n_boot)]
    ci_boot = np.percentile(boot, [2.5, 97.5])
    return {'mean_diff': d.mean(), 't_stat': t_stat, 'p_val': p_val,
            'ci_t': ci_t, 'ci_boot': ci_boot}


def regime_stats(var_dict, mean_vix, threshold=STRESS_VIX):
    stress = mean_vix >= threshold
    calm   = ~stress
    out = {}
    for label, mask in [('Calm', calm), ('Stress', stress)]:
        n_wins = mask.sum()
        oas_v  = np.array(var_dict['OAS'])[mask]
        out[label] = {'n': int(n_wins)}
        for k in ['LW', 'RBLW', 'OAS-Ext']:
            k_v = np.array(var_dict[k])[mask]
            vol_oas = np.sqrt(oas_v.mean() * 252) * 100
            vol_k   = np.sqrt(k_v.mean()   * 252) * 100
            out[label][k] = {
                'ann_vol': vol_k,
                'pct_imp': 100 * (vol_oas - vol_k) / vol_oas,
            }
        out[label]['OAS'] = {'ann_vol': np.sqrt(oas_v.mean() * 252) * 100}
    return out


def generate_figure(window_dates, alphas, window_vix, var_dict, fig_path):
    C_BLUE  = '#4477AA'
    C_RED   = '#EE6677'
    C_GREEN = '#228833'
    C_GREY  = '#BBBBBB'

    stress_mask = window_vix >= STRESS_VIX

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.40)

    # --- Left: alpha_hat over time, stress periods shaded ---
    ax1.plot(window_dates, alphas, color=C_BLUE, lw=0.9, zorder=3)
    ax1.axhline(0, color='#aaaaaa', lw=0.8, ls='--', zorder=2)

    # shade stress windows
    in_stress = False
    stress_start = None
    for i, (d, s) in enumerate(zip(window_dates, stress_mask)):
        if s and not in_stress:
            stress_start = d
            in_stress = True
        elif not s and in_stress:
            ax1.axvspan(stress_start, d, alpha=0.15, color=C_RED, zorder=1)
            in_stress = False
    if in_stress:
        ax1.axvspan(stress_start, window_dates[-1], alpha=0.15, color=C_RED, zorder=1)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel(r'$\hat{\alpha}$', fontsize=12)
    ax1.set_title(r'$\hat{\alpha}$ over time  (shading: VIX $\geq 20$)', fontsize=9)
    ax1.grid(True, lw=0.4, color='#dddddd', zorder=0)
    ax1.spines[['top', 'right']].set_visible(False)

    # --- Right: % improvement over OAS by regime ---
    regimes  = ['Calm\n(VIX < 20)', 'Stress\n(VIX ≥ 20)']
    masks    = [~stress_mask, stress_mask]
    methods  = ['LW', 'RBLW', 'OAS-Ext']
    colors   = [C_BLUE, C_GREY, C_RED]
    n_groups = 2
    n_bars   = len(methods)
    width    = 0.22
    x        = np.arange(n_groups)

    for i, (method, color) in enumerate(zip(methods, colors)):
        imps = []
        for mask in masks:
            oas_v   = np.array(var_dict['OAS'])[mask]
            m_v     = np.array(var_dict[method])[mask]
            vol_oas = np.sqrt(oas_v.mean() * 252) * 100
            vol_m   = np.sqrt(m_v.mean()   * 252) * 100
            imps.append(100 * (vol_oas - vol_m) / vol_oas)
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax2.bar(x + offset, imps, width, color=color,
                       label=method, edgecolor='white')
        for bar, val in zip(bars, imps):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     max(val, 0) + 0.05,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5)

    ax2.axhline(0, color='#888888', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, fontsize=10)
    ax2.set_ylabel('Vol reduction vs OAS original (%)', fontsize=10)
    ax2.set_title('GMV portfolio: improvement by market regime', fontsize=9)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(True, axis='y', lw=0.4, color='#dddddd')
    ax2.spines[['top', 'right']].set_visible(False)

    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    print(f'Figure saved to {fig_path}')


if __name__ == '__main__':
    print('Downloading data...')
    returns_matrix, tickers_used, date_index, vix_series = download_data(
        TICKERS, START, END)
    T, p = returns_matrix.shape
    print(f'p={p} tickers, T={T} trading days')

    print('Running rolling-window study...')
    window_dates, alphas, window_vix, var_dict = run_rolling(
        returns_matrix, vix_series, date_index)
    N = len(window_dates)
    n_stress = int((window_vix >= STRESS_VIX).sum())
    n_calm   = N - n_stress
    print(f'N={N} windows  ({n_calm} calm, {n_stress} stress)')

    print(f'\nalpha_hat:  mean={alphas.mean():.6f}  '
          f'median={np.median(alphas):.6f}  frac>0={100*np.mean(alphas>0):.1f}%')

    print('\nOverall annualised realised vol (%):')
    ann = {k: np.sqrt(np.mean(v) * 252) * 100 for k, v in var_dict.items()}
    for k, v in ann.items():
        print(f'  {k:10s}: {v:.4f}%')

    print('\nPaired tests (OAS baseline):')
    for cmp in ['OAS-Ext', 'LW', 'RBLW']:
        res = paired_test(var_dict['OAS'], var_dict[cmp], label_b=cmp)
        print(f'  OAS vs {cmp:10s}: mean_diff={res["mean_diff"]:.6f}  '
              f't={res["t_stat"]:.3f}  p={res["p_val"]:.4f}  '
              f'95%CI_boot=[{res["ci_boot"][0]:.6f}, {res["ci_boot"][1]:.6f}]')

    print('\nRegime breakdown:')
    reg = regime_stats(var_dict, window_vix)
    for regime, data in reg.items():
        print(f'  {regime} (n={data["n"]}):')
        print(f'    OAS vol = {data["OAS"]["ann_vol"]:.4f}%')
        for k in ['LW', 'RBLW', 'OAS-Ext']:
            print(f'    {k:10s}: vol={data[k]["ann_vol"]:.4f}%  '
                  f'imp={data[k]["pct_imp"]:.2f}%')

    generate_figure(window_dates, alphas, window_vix, var_dict,
                    fig_path='./oas_financial.pdf')
