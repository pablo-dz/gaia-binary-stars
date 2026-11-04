#!/usr/bin/env python
"""Calcula la fracción de binarias MS-MS de cada cúmulo globular ajustado."""
import argparse

import matplotlib

matplotlib.use("Agg")

from binfrac import pipeline
from binfrac.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un YAML de configuración alternativo")
    args = parser.parse_args()
    pipeline.run_binary_fraction(load_config(args.config))


if __name__ == "__main__":
    main()
