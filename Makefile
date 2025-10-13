.RECIPEPREFIX = >
.PHONY: init format clean run diag kill

VENV=.venv
PY=$(VENV)/bin/python

init:
> python3 -m venv $(VENV)
> . $(VENV)/bin/activate; \
> $(PY) -m pip install -U pip setuptools wheel; \
> $(PY) -m pip install -r requirements.txt black autopep8 flake8

format:
> find . -name "*.py" -print0 | xargs -0 sed -i '' $'s/\t/    /g'
> $(PY) -m black .
> $(PY) -m autopep8 --in-place --aggressive --aggressive l3_vfa_pma_gui.py

clean:
> find . -name "__pycache__" -type d -exec rm -rf {} +
> rm -rf .mypy_cache .pytest_cache

run:
> PYTHONDONTWRITEBYTECODE=1 $(PY) -B l3_vfa_pma_gui.py

diag:
> PYTHONDONTWRITEBYTECODE=1 $(PY) - << 'PY'
import sys, os, time, importlib
print("[BOOT] sys.executable =", sys.executable)
print("[BOOT] cwd            =", os.getcwd())
import src.core as core
importlib.invalidate_caches()
print("[CORE] file:", core.__file__)
print("[CORE] mtime:", time.ctime(os.path.getmtime(core.__file__)))
core = importlib.reload(core)
print("[CORE] reloaded mtime:", time.ctime(os.path.getmtime(core.__file__)))
PY

kill:
> pkill -f l3_vfa_pma_gui.py || true
