PY ?= python

.PHONY: setup eda train train-quick evaluate test serve

setup:
	$(PY) -m pip install -r requirements.txt

# Executes the notebook with the python3 kernel of the active environment (pip install -r requirements.txt provides ipykernel)
eda:
	cd notebooks && $(PY) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 eda.ipynb

train:
	$(PY) -m src.train

train-quick:
	$(PY) -m src.train --quick --out /tmp/lydia-quick/models --reports /tmp/lydia-quick/reports

evaluate:
	$(PY) -m src.evaluate

test:
	$(PY) -m pytest

# The API module (api/main.py) arrives in a later step.
serve:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000
