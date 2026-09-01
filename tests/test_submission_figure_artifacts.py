from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_figure3_legend_explains_both_dashed_reference_lines():
    legends = pd.read_csv(ROOT / "outputs" / "tables" / "figure_legends_v7.csv")
    legend = legends.loc[legends["figure"].eq("Figure 3"), "legend"].iloc[0]

    assert "vertical dashed line" in legend
    assert "horizontal dashed line" in legend
    assert "50%" in legend
