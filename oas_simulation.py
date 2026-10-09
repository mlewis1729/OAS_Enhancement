import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.covariance import LedoitWolf


def sample_covariance(X):
    n = X.shape[0]
    return X.T @ X / n


def rblw(S, n, p):
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    tr2_S = tr_S ** 2

    numer = ((n - 2) / n) * tr_S2 + tr2_S
    denom = (n + 2) * (tr_S2 - tr2_S / p)

    rho = min(1.0, numer / denom) if denom > 0 else 1.0
    F = (tr_S / p) * np.eye(p)
    return (1 - rho) * S + rho * F


def oas_original(S, n, p):
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    tr2_S = tr_S ** 2

    M = (1 - 2/p) * tr_S2 + tr2_S
    denom = (n + 1 - 2/p) * (tr_S2 - tr2_S / p)

    rho = min(1.0, M / denom) if denom > 0 else 1.0
    F = (tr_S / p) * np.eye(p)
    return (1 - rho) * S + rho * F


def oas_extended(X, S, n, p):
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    tr2_S = tr_S ** 2

    alpha_hat = np.mean(np.sum(X ** 2, axis=1) ** 2) - tr2_S - 2 * tr_S2

    M = (1 - 2/p) * tr_S2 + tr2_S
    beta_hat = alpha_hat * (1 - 1/p) / M if M != 0 else 0.0
    phi_hat = (tr_S2 - tr2_S / p) / M if M != 0 else 0.0

    denom = (n + 1 - 2/p) * phi_hat
    rho = min(1.0, (1 + beta_hat) / denom) if denom > 0 else 1.0

    F = (tr_S / p) * np.eye(p)
    return (1 - rho) * S + rho * F


def ledoit_wolf(X):
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X)
    return lw.covariance_


def normalized_frobenius_loss(Sigma_hat, Sigma):
    diff = Sigma_hat - Sigma
    return np.sqrt(np.trace(diff @ diff)) / np.sqrt(np.trace(Sigma @ Sigma))


def draw_samples(rng, n, p, L, dist, df):
    Z = rng.standard_normal((n, p))
    if dist == 'student_t':
        scale = np.sqrt((df - 2) / df)
        chi2 = rng.chisquare(df, size=n)
        Z = scale * Z / np.sqrt(chi2 / df)[:, None]
    return Z @ L.T


def run_simulation(p, n, Sigma, dist, df, n_trials=500, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma)
    losses = {'Sample': [], 'LW': [], 'RBLW': [], 'OAS': [], 'OAS-Ext': []}

    for _ in range(n_trials):
        X = draw_samples(rng, n, p, L, dist, df)
        S = sample_covariance(X)
        losses['Sample'].append(normalized_frobenius_loss(S, Sigma))
        losses['LW'].append(normalized_frobenius_loss(ledoit_wolf(X), Sigma))
        losses['RBLW'].append(normalized_frobenius_loss(rblw(S, n, p), Sigma))
        losses['OAS'].append(normalized_frobenius_loss(oas_original(S, n, p), Sigma))
        losses['OAS-Ext'].append(normalized_frobenius_loss(oas_extended(X, S, n, p), Sigma))

    return {k: np.mean(v) for k, v in losses.items()}


def ar1_sigma(p, rho=0.5):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_figure(results_curve, fig_path='oas_simulation.pdf'):
    """Two-panel figure.
    Left:  loss curves for t(df=5) — all four estimators.
    Right: relative improvement over OAS original for OAS-Ext and LW,
           both non-Gaussian distributions.
    """
    C_GAUSSIAN = '#888888'
    C_T10      = '#4477AA'
    C_T5       = '#EE6677'

    LS_SAMPLE = (0, (1, 1))         # dotted
    LS_LW     = (0, (4, 1.5))      # long dash
    LS_RBLW   = (0, (3, 1, 1, 1))  # dash-dot
    LS_OAS    = '--'                # dashed
    LS_EXT    = '-'                 # solid

    LW_WIDTH = 1.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.38)

    ratios = sorted(results_curve.keys())

    # --- Left panel: all estimators for t(df=5) only ---
    dist_key = 't (df=5)'
    s_vals    = [results_curve[r][dist_key]['Sample']  for r in ratios]
    lw_vals   = [results_curve[r][dist_key]['LW']      for r in ratios]
    rblw_vals = [results_curve[r][dist_key]['RBLW']    for r in ratios]
    oas_vals  = [results_curve[r][dist_key]['OAS']     for r in ratios]
    ext_vals  = [results_curve[r][dist_key]['OAS-Ext'] for r in ratios]

    ax1.plot(ratios, s_vals,    color=C_T5, lw=LW_WIDTH, ls=LS_SAMPLE)
    ax1.plot(ratios, lw_vals,   color=C_T5, lw=LW_WIDTH, ls=LS_LW)
    ax1.plot(ratios, rblw_vals, color=C_T5, lw=LW_WIDTH, ls=LS_RBLW)
    ax1.plot(ratios, oas_vals,  color=C_T5, lw=LW_WIDTH, ls=LS_OAS)
    ax1.plot(ratios, ext_vals,  color=C_T5, lw=LW_WIDTH, ls=LS_EXT)

    h_sample = mlines.Line2D([], [], color='#444444', lw=LW_WIDTH, ls=LS_SAMPLE, label='Sample cov.')
    h_lw     = mlines.Line2D([], [], color='#444444', lw=LW_WIDTH, ls=LS_LW,     label='Ledoit--Wolf')
    h_rblw   = mlines.Line2D([], [], color='#444444', lw=LW_WIDTH, ls=LS_RBLW,   label='RBLW')
    h_oas    = mlines.Line2D([], [], color='#444444', lw=LW_WIDTH, ls=LS_OAS,    label='OAS (original)')
    h_ext    = mlines.Line2D([], [], color='#444444', lw=LW_WIDTH, ls=LS_EXT,    label='OAS (extended)')
    ax1.legend(handles=[h_sample, h_lw, h_rblw, h_oas, h_ext], fontsize=8,
               frameon=False, loc='upper right')

    ax1.set_xlabel(r'$n/p$', fontsize=11)
    ax1.set_ylabel('Normalised Frobenius loss', fontsize=10)
    ax1.set_title(r'Loss vs sample ratio — $t\,(\nu\!=\!5)$', fontsize=9)
    ax1.set_xscale('log')
    ax1.grid(True, which='both', lw=0.4, color='#dddddd')
    ax1.spines[['top', 'right']].set_visible(False)

    # --- Right panel: % improvement over OAS, OAS-Ext and LW, both non-Gaussian dists ---
    for dist_key, color, dist_label in [
        ('t (df=10)', C_T10, r'$t\,(\nu\!=\!10)$'),
        ('t (df=5)',  C_T5,  r'$t\,(\nu\!=\!5)$'),
    ]:
        oas_v  = [results_curve[r][dist_key]['OAS']     for r in ratios]
        ext_v  = [results_curve[r][dist_key]['OAS-Ext'] for r in ratios]
        lw_v   = [results_curve[r][dist_key]['LW']      for r in ratios]
        rblw_v = [results_curve[r][dist_key]['RBLW']    for r in ratios]

        rel_ext  = [100 * (o - e) / o for o, e in zip(oas_v, ext_v)]
        rel_lw   = [100 * (o - l) / o for o, l in zip(oas_v, lw_v)]
        rel_rblw = [100 * (o - r) / o for o, r in zip(oas_v, rblw_v)]

        ax2.plot(ratios, rel_ext,  color=color, lw=LW_WIDTH, ls=LS_EXT,
                 label=f'OAS-Ext, {dist_label}')
        ax2.plot(ratios, rel_lw,   color=color, lw=LW_WIDTH, ls=LS_LW,
                 label=f'LW, {dist_label}')
        ax2.plot(ratios, rel_rblw, color=color, lw=LW_WIDTH, ls=LS_RBLW,
                 label=f'RBLW, {dist_label}')

    ax2.axhline(0, color='#aaaaaa', lw=0.8)
    ax2.set_xlabel(r'$n/p$', fontsize=11)
    ax2.set_ylabel('Loss reduction vs OAS original (%)', fontsize=10)
    ax2.set_title('Improvement over OAS original', fontsize=9)
    ax2.set_xscale('log')
    ax2.grid(True, which='both', lw=0.4, color='#dddddd')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=7.5, frameon=False, loc='lower right')

    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
    print(f'Figure saved to {fig_path}')


if __name__ == '__main__':
    p = 120
    Sigma = ar1_sigma(p, rho=0.5)

    distributions = [
        ('gaussian',  None, 'Gaussian'),
        ('student_t', 10,   't (df=10)'),
        ('student_t', 5,    't (df=5)'),
    ]

    # Table: 4 key n/p values, 500 trials
    table_ratios = [0.25, 0.5, 1.0, 2.0, 5.0]
    rows = []
    for n_ratio in table_ratios:
        n = int(p * n_ratio)
        for dist, df, label in distributions:
            res = run_simulation(p, n, Sigma, dist, df, n_trials=500)
            rows.append({
                'n/p': n_ratio,
                'Distribution': label,
                'Sample Cov': f"{res['Sample']:.4f}",
                'LW': f"{res['LW']:.4f}",
                'RBLW': f"{res['RBLW']:.4f}",
                'OAS (original)': f"{res['OAS']:.4f}",
                'OAS (extended)': f"{res['OAS-Ext']:.4f}",
            })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    # Curves: more n/p values, 200 trials each
    curve_ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    results_curve = {}
    for n_ratio in curve_ratios:
        n = int(p * n_ratio)
        results_curve[n_ratio] = {}
        for dist, df, label in distributions:
            results_curve[n_ratio][label] = run_simulation(
                p, n, Sigma, dist, df, n_trials=500)

    generate_figure(results_curve, fig_path='/home/ec2-user/claude_playzone/oas_simulation.pdf')
