.PHONY: setup run test clean help

# Variables
VENV = venv_autostream
PYTHON = ./$(VENV)/bin/python
PIP = ./$(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  setup    : Create venv and install dependencies"
	@echo "  run      : Run the Streamlit dashboard"
	@echo "  verify   : Run the CLI verification script"
	@echo "  clean    : Remove temporary files and venv"

setup:
	python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	./$(VENV)/bin/streamlit run app.py

verify:
	$(PYTHON) verify_agent.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
