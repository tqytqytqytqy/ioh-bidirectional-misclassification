"""Redraw Supplementary Figure S1 from frozen, non-identifiable aggregate values."""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
SOURCE = ROOT / 'outputs/aggregate/figure_s1_exploratory_aki_forest_source.csv'
OUT = ROOT / 'outputs/figures'


def main():
    baseline = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    values = pd.read_csv(SOURCE)
    assert len(values) == 4
    assert np.isfinite(values[['relative_risk', 'ci_low', 'ci_high']]).all().all()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    positions = np.arange(len(values))[::-1]
    for position, row, color in zip(positions, values.itertuples(), ['#777777', '#C44E52', '#4C72B0', '#2A7F62']):
        ax.errorbar(row.relative_risk, position,
                    xerr=[[row.relative_risk - row.ci_low], [row.ci_high - row.relative_risk]],
                    fmt='o', color=color, ecolor=color, capsize=3, markersize=5)
    ax.axvline(1, color='#555555', linewidth=1, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlim(0.70, 1.40)
    ax.set_xticks([0.75, 0.9, 1.0, 1.1, 1.3], labels=['0.75', '0.90', '1.00', '1.10', '1.30'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(positions, labels=[f'{r.plot_label}\n' + r'$\mathit{n}$' + f'={int(r.n_cases)}; AKI={int(r.events)}'
                                   for r in values.itertuples()])
    ax.set_xlabel('Relative risk (95% CI)')
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.6)
    ax.set_title('Exploratory associations with postoperative AKI', loc='left')
    fig.text(0.01, 0.01, r'Absolute hidden burden is scaled per 10 mmHg min h$^{-1}$; HDR65 per 10 percentage points. Models are exploratory and noncausal.', fontsize=7.5)
    fig.subplots_adjust(left=0.39, bottom=0.19, right=0.97, top=0.88)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / 'Supplemental_Figure_S1_exploratory_AKI_forest'
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == baseline
    print(json.dumps({'figure': stem.name, 'source_sha256': baseline, 'points': len(values),
                      'statistical_rerun': False, 'generative_image_model': False}))


if __name__ == '__main__':
    main()
