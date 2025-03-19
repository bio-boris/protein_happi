#!/usr/bin/env bash

VENV_PATH=".venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    . "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at $VENV_PATH. Exiting..."
    exit 1
fi

export AUTH_URL='https://ci.kbase.us/services/auth/api/V2/me'
export ROOT_PATH='/services/llm_homology_api/'
export VERSION=0.0.1
export VCS_REF=$(git status)
export EMBEDDING_DATASET_DIR=/scratch/sprot/sprot_esm_650m_faiss
export FAISS_INDEX_PATH=/scratch1/abrace/data/trembl/faiss-trembl.bin
export SEARCH_GPUS=2


uvicorn src.factory:create_app --host 0.0.0.0 --port 5000 --factory