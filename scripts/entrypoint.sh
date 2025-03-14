#!/usr/local/bin/env bash


VENV_PATH=".venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at $VENV_PATH. Exiting..."
    exit 1
fi

exec uvicorn src.factory:create_app --host 0.0.0.0 --port 8000 --factory