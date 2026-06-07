"""
analyze.py  —  Compare all experiments and generate report figures.
Run AFTER all 4 experiments finish training.

  python analyze.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("../results")
EXPERIMENTS = ["baseline", "lds", "fds", "dir"]
LABELS      = ["Baseline", "+ LDS", "+ FDS", "+ LDS + FDS (DIR)"]
COLORS      = ["#6c757d", "#0077b6", "#e07b39", "#2a9d8f"]


def load_results(exp):
    p = RESULTS_DIR / exp / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_preds(exp):
    p = RESULTS_DIR / exp / "test_predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def make_summary_table(all_results):
    rows = []
    for exp, label in zip(EXPERIMENTS, LABELS):
        r = all_results.get(exp)
        if r is None:
            continue
        t = r['test']
        shots = t.get('shots', {})
        rows.append({
            'Method':           label,
            'Test MAE (°C)':    round(t['mae'], 3),
            'Test RMSE (°C)':   round(t['rmse'], 3),
            'G-Mean':           round(t['gmean'], 3),
            'Median-shot MAE':  round(shots.get('median', {}).get('l1', float('nan')), 3),
            'Low-shot MAE':     round(shots.get('low',    {}).get('l1', float('nan')), 3),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    return df


def plot_bar_comparison(df):
    # Only use columns without NaN
    metrics = ['Test MAE (°C)', 'Median-shot MAE', 'Low-shot MAE']
    titles  = ['Overall MAE', 'Median-shot MAE\n(common temps)', 'Low-shot MAE\n(rare temps)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        vals = df[metric].values.astype(float)
        # Skip if all NaN
        if np.all(np.isnan(vals)):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue
        # Replace NaN with 0 for plotting
        plot_vals = np.nan_to_num(vals, nan=0.0)
        bars = ax.bar(df['Method'], plot_vals,
                      color=COLORS[:len(plot_vals)],
                      edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (°C)')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Method'], rotation=20, ha='right', fontsize=9)
        valid_max = np.nanmax(vals)
        ax.set_ylim(0, valid_max * 1.25)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('DIR Methods Comparison on SkyFinder Temperature Prediction',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Bar chart saved → {RESULTS_DIR / 'comparison_bars.png'}")


def plot_scatter(exp, label, color):
    df = load_preds(exp)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['label'], df['pred'], alpha=0.3, s=10, color=color)
    lims = [min(df['label'].min(), df['pred'].min()) - 2,
            max(df['label'].max(), df['pred'].max()) + 2]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    mae = np.mean(np.abs(df['pred'] - df['label']))
    ax.set_xlabel('True Temperature (°C)', fontsize=11)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=11)
    ax.set_title(f'{label}\nMAE = {mae:.2f}°C', fontsize=12)
    ax.legend()
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    out = RESULTS_DIR / f'scatter_{exp}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[✓] Scatter saved → {out}")


def plot_error_by_temperature(all_preds):
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp, label, color in zip(EXPERIMENTS, LABELS, COLORS):
        df = all_preds.get(exp)
        if df is None:
            continue
        df = df.copy()
        df['bin'] = np.round(df['label']).astype(int)
        grouped = df.groupby('bin').apply(
            lambda g: np.mean(np.abs(g['pred'] - g['label']))
        )
        ax.plot(grouped.index, grouped.values, label=label,
                color=color, linewidth=2, alpha=0.85)

    ax.set_xlabel('True Temperature (°C)', fontsize=11)
    ax.set_ylabel('MAE (°C)', fontsize=11)
    ax.set_title('MAE by Temperature Range\n(peaks at rare temperatures show imbalance effect)',
                 fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'error_by_temp.png', dpi=150)
    plt.close()
    print(f"[✓] Error-by-temp plot saved → {RESULTS_DIR / 'error_by_temp.png'}")


def plot_training_curves():
    fig, ax = plt.subplots(figsize=(10, 5))
    for exp, label, color in zip(EXPERIMENTS, LABELS, COLORS):
        hist = RESULTS_DIR / exp / 'history.csv'
        if not hist.exists():
            continue
        df = pd.read_csv(hist)
        ax.plot(df['epoch'], df['val_mae'], label=label, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE (°C)')
    ax.set_title('Training Curves — Validation MAE per Epoch')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=150)
    plt.close()
    print(f"[✓] Training curves saved → {RESULTS_DIR / 'training_curves.png'}")


def main():
    all_results = {}
    all_preds   = {}

    for exp in EXPERIMENTS:
        r = load_results(exp)
        if r:
            all_results[exp] = r
            print(f"[✓] Loaded results: {exp}")
        else:
            print(f"[!] Missing results for: {exp}")

        d = load_preds(exp)
        if d is not None:
            all_preds[exp] = d

    if not all_results:
        print("\n[!] No results found.")
        return

    df = make_summary_table(all_results)
    plot_bar_comparison(df)

    for exp, label, color in zip(EXPERIMENTS, LABELS, COLORS):
        if exp in all_preds:
            plot_scatter(exp, label, color)

    if all_preds:
        plot_error_by_temperature(all_preds)

    plot_training_curves()

    print("\n[✓] All plots saved to", RESULTS_DIR)


if __name__ == '__main__':
    main()
