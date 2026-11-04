#!/usr/bin/env python
"""Grafica la fracción binaria en función de la edad del cúmulo."""
import argparse

import matplotlib

matplotlib.use("Agg")

from binfrac import pipeline
from binfrac.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un YAML de configuración alternativo")
    args = parser.parse_args()
    pipeline.run_fraction_vs_age(load_config(args.config))


if __name__ == "__main__":
    main()
