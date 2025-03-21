from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import Field
from typing import Literal
from pathlib import Path


class LLMHomologyApiSettings(BaseSettings):
    """Settings for the LLM Homology API taken from the environment"""

    # The maximum length of the protein sequence header
    MAX_RESIDUE_HEADER_LENGTH: int = 100
    # The maximum number of residues allowed in a single protein sequence
    MAX_RESIDUE_COUNT: int = 5000
    # The maximum number of protein sequences allowed in a single request
    MAX_PROTEINS_PER_REQUEST: int = 500

    MAX_REQUEST_SIZE: int = 2805000  # Not yet implemented
    VERSION: str
    ROOT_PATH: str
    AUTH_URL: str
    ADMIN_ROLES: list = ["LLMHomologyAdmin"]
    VCS_REF: str

    # The similarity search configuration
    FAISS_SEARCH_GPUS: int = Field(
        default=0,
        ge=0,
        description="The number of GPUs to use for the similarity search. "
        "Using 0 will place the faiss index on the same GPU as the encoder. "
        "Using more than 0 will place the faiss index on the next available GPUs. "
        "The GPU placement is relative to CUDA_VISIBLE_DEVICES.",
    )
    FAISS_SEARCH_PRECISION: Literal["float32", "ubinary"] = Field(
        default="ubinary",
        description="The precision of the faiss index search [float32, ubinary].",
    )
    FAISS_EMBEDDING_DATASET_DIR: Path = Field(
        ...,
        description="The directory containing the Arrow embedding dataset.",
    )
    FAISS_INDEX_PATH: Path = Field(
        ...,
        description="The path to the faiss index file.",
    )
    FAISS_DATASET_CHUNK_DIR: Path | None = Field(
        default=None,
        description="The directory containing the Arrow embedding dataset chunks.",
    )
    FAISS_NUM_QUANTIZATION_WORKERS: int = Field(
        default=1,
        ge=1,
        description="The number of quantization workers.",
    )
    ENCODER_NAME: Literal["esm2", "esmc", "prottrans"] = Field(
        default="esm2",
        description="The name of the encoder to use.",
    )
    ENCODER_PRETRAINED_MODEL_NAME_OR_PATH: str = Field(
        default="facebook/esm2_t33_650M_UR50D",
        description="The encoder model id.",
    )
    ENCODER_DATALOADER_BATCH_SIZE: int = Field(
        default=8,
        ge=1,
        description="The encoder dataloader batch size.",
    )
    ENCODER_DATALOADER_NUM_DATA_WORKERS: int = Field(
        default=4,
        ge=1,
        description="The encoder dataloader number of data worker processes.",
    )

    class Config:
        extra = "forbid"


@lru_cache(maxsize=None)
def get_settings() -> LLMHomologyApiSettings:
    return LLMHomologyApiSettings()
