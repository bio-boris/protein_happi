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
#export PYTHONPATH=.:llm_homology_api:llm_homology_api/src

# Similarity search environment variables
export FAISS_EMBEDDING_DATASET_DIR=/scratch/sprot/sprot_esm_650m_faiss
export FAISS_INDEX_PATH=/scratch/sprot/sprot_esm_650m_faiss/faiss_index
export FAISS_SEARCH_GPUS=2
export FAISS_SEARCH_PRECISION=ubinary
export ENCODER_NAME=esm2
export ENCODER_PRETRAINED_MODEL_NAME_OR_PATH=facebook/esm2_t33_650M_UR50D
export ENCODER_DATALOADER_BATCH_SIZE=8
export ENCODER_DATALOADER_NUM_DATA_WORKERS=4

uvicorn src.factory:create_app --host 0.0.0.0 --port 5000 --factory