#!/usr/bin/env python
"""MSRL por data binning y fracción binaria preliminar (NGC 4755, NGC 6694, NGC 2808)."""
import argparse

import matplotlib

matplotlib.use("Agg")

from binfrac import pipeline
from binfrac.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un YAML de configuración alternativo")
    args = parser.parse_args()
    pipeline.run_preliminary(load_config(args.config))


if __name__ == "__main__":
    main()
