#!/usr/bin/env python
"""Compara isócronas MIST de distinta metalicidad con el CMD de cada cúmulo globular."""
import argparse

import matplotlib

matplotlib.use("Agg")

from binfrac import pipeline
from binfrac.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un YAML de configuración alternativo")
    args = parser.parse_args()
    pipeline.run_isochrone_fits(load_config(args.config))


if __name__ == "__main__":
    main()
