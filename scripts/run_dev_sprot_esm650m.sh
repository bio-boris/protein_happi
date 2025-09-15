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
# ========================================

# The cache directory for the encoder model weights
export HF_HOME=/scratch/abrace/.cache

# The directory containing the merged embeddings database
export FAISS_EMBEDDING_DATASET_DIR=/scratch/sprot/sprot_esm_650m_faiss

# The directory containing the FAISS index
export FAISS_INDEX_PATH=/scratch/sprot/sprot_esm_650m_faiss/faiss_index

# The number of GPUs to use for the FAISS index search
export FAISS_SEARCH_GPUS=2

# The search precision for the FAISS index search
export FAISS_SEARCH_PRECISION=ubinary

# The name of the encoder model type
export ENCODER_NAME=esm2

# The unique identifier for the encoder model
export ENCODER_PRETRAINED_MODEL_NAME_OR_PATH=facebook/esm2_t33_650M_UR50D

# The batch size for the encoder model
export ENCODER_DATALOADER_BATCH_SIZE=8

# The number of data workers for the encoder model
export ENCODER_DATALOADER_NUM_DATA_WORKERS=4
# ========================================

uvicorn src.factory:create_app --host 0.0.0.0 --port 5000 --factory