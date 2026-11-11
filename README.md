# cse-hawkes

First version of a small research project on the Casablanca Stock Exchange (MASI).

I want to see how two models talk about rough market days:

GARCH follows how wild returns are from day to day.

Hawkes asks if a bad day makes another bad day more likely soon after.

Later I plan to connect this to my NLP work on Moroccan finance comments and code switching (LixusFinSent). That part is not in this first version.

## What works today

Fake daily prices so the code runs without real exchange data.

A GARCH fit.

A simple Hawkes fit on the worst days.

A short JSON summary in `results/fit_summary.json`.

Real MASI closes are not in this repo. See `docs/data.md`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_synthetic.py
python scripts/run_baseline.py
```

## Folder map

`scripts` things you run

`src/models` GARCH and Hawkes

`data/synthetic` fake series (shared)

`data/raw` real data on my machine only

`docs` short notes

`paper` draft thoughts

`results` run output

## Status

Early public draft. Methods first. Real market numbers come only after proper permission.
