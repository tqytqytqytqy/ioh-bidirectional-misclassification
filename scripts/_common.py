from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ioh.pipeline import load_config  # noqa: E402


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.example.yaml")
    args = parser.parse_args()
    return load_config(args.config)
