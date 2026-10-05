from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_figure3_legend_distinguishes_quantiles_from_confidence_limits():
    legends = pd.read_csv(ROOT / "outputs" / "tables" / "figure_legends.csv")
    legend = legends.loc[legends["figure"].eq("Figure 3"), "legend"].iloc[0]

    assert "interquartile range" in legend
    assert "5th and 95th percentiles" in legend
    assert "not confidence limits" in legend
