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
export FAISS_EMBEDDING_DATASET_DIR=/scratch/abrace/data/uniprot_2025_02/sprot-trembl-combined-embeddings/embeddings.merge

# The directory containing the FAISS index
export FAISS_INDEX_PATH=/scratch/abrace/data/uniprot_2025_02/sprot-trembl-combined-embeddings/ubinary-uniprot-combined-esm3b-faesm-faiss-ivf.index
#export FAISS_INDEX_PATH=/dev/shm/ubinary-uniprot-combined-esm3b-faesm-faiss.index

# The number of GPUs to use for the FAISS index search
export FAISS_SEARCH_GPUS=3

# The search precision for the FAISS index search
export FAISS_SEARCH_PRECISION=ubinary

# The faiss search algorithm to use [exact, ivf].
export FAISS_SEARCH_ALGORITHM=ivf

# The number of clusters for the IVF index. 
# set to nearest power of 2 of sqrt(250M) 
export FAISS_SEARCH_IVF_NLIST=16384 

# The number of clusters to probe for each search in the IVF index.
export FAISS_SEARCH_IVF_NPROBE=16

# The maximum number of embeddings to use for training the IVF index.  
export FAISS_SEARCH_IVF_MAX_TRAIN_SIZE=1000000

# The directory containing the chunked embeddings database for building the FAISS index
export FAISS_DATASET_CHUNK_DIR=/scratch/abrace/data/uniprot_2025_02/sprot-trembl-combined-embeddings/embeddings

# The number of CPU workers to use for quantization (reads the chunked embeddings database)
export FAISS_NUM_QUANTIZATION_WORKERS=40

# The name of the encoder model type
export ENCODER_NAME=esm2

# The unique identifier for the encoder model
export ENCODER_PRETRAINED_MODEL_NAME_OR_PATH=facebook/esm2_t36_3B_UR50D

# Use the flash-attention implementation for ESM2 
export ENCODER_ENABLE_FAESM=true

# The batch size for the encoder model
export ENCODER_DATALOADER_BATCH_SIZE=128

# The number of data workers for the encoder model
export ENCODER_DATALOADER_NUM_DATA_WORKERS=8

# ========================================

PYTHONUNBUFFERED=1 uvicorn src.factory:create_app --host 0.0.0.0 --port 5000 --factory