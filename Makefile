.PHONY: install membership preliminary isochrones fraction age all test

install:
	pip install -e ".[dev]"

membership:
	python scripts/01_membership.py

preliminary:
	python scripts/02_preliminary_msrl.py

isochrones:
	python scripts/03_fit_isochrones.py

fraction:
	python scripts/04_binary_fraction.py

age:
	python scripts/05_fraction_vs_age.py

all: membership preliminary isochrones fraction age

test:
	pytest
