import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from natsort import natsorted
from protein_search_evals.embed import get_encoder
from protein_search_evals.search import FaissIndex
from protein_search_evals.search import Retriever
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

from src.models import HitModel, HitsModel, SearchRequest, SearchResponse


class LLMHomologyApiSettings(BaseSettings):
    """Settings for the LLM Homology API taken from the environment"""

    # The maximum length of the protein sequence header
    MAX_RESIDUE_HEADER_LENGTH: int = 100
    # The maximum number of residues allowed in a single protein sequence
    MAX_RESIDUE_COUNT: int = 5000
    # The maximum number of protein sequences allowed in a single request
    MAX_PROTEINS_PER_REQUEST: int = 500

    # The maximum request size
    MAX_REQUEST_SIZE: int = 2805000  # TODO: Not yet implemented
    # The version of the API
    VERSION: str
    # The root path of the API
    ROOT_PATH: str
    # The authentication URL
    AUTH_URL: str
    # The admin roles
    ADMIN_ROLES: list = ["LLMHomologyAdmin"]
    # The version control system reference
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
    FAISS_SEARCH_ALGORITHM: Literal["exact", "ivf"] = Field(
        default="exact",
        description="The faiss search algorithm to use [exact, ivf].",
    )
    FAISS_SEARCH_IVF_NLIST: int = Field(
        default=512,
        description="The number of clusters for the IVF index.",
    )
    FAISS_SEARCH_IVF_NPROBE: int = Field(
        default=8,
        description="The number of clusters to probe for each search in the IVF index.",
    )
    FAISS_SEARCH_IVF_MAX_TRAIN_SIZE: int = Field(
        default=1_000_000,
        description="The maximum number of embeddings to use for training "
                    "the IVF index.",
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
        description="The directory containing the chunked embeddings "
                    "database for building the FAISS index",
    )
    FAISS_NUM_QUANTIZATION_WORKERS: int = Field(
        default=1,
        ge=1,
        description="The number of CPU workers to use for quantization "
                    "(reads the chunked embeddings database)",
    )
    ENCODER_NAME: Literal["esm2", "esmc", "prottrans"] = Field(
        default="esm2",
        description="The name of the encoder to use.",
    )
    ENCODER_PRETRAINED_MODEL_NAME_OR_PATH: str = Field(
        default="facebook/esm2_t33_650M_UR50D",
        description="The encoder model id.",
    )
    ENCODER_ENABLE_FAESM: bool = Field(
        default=False,
        description="Use the flash-attention implementation for ESM2.",
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


@lru_cache(maxsize=None)
def initialize_search(
        settings: LLMHomologyApiSettings | None = None,
) -> tuple[Retriever, np.ndarray]:
    """Initialize the retriever.

    Returns
    -------
    Retriever
        The initialized retriever.
    np.ndarray
        The Uniprot IDs for all the embeddings in the faiss index.
    """
    logger = logging.getLogger("protein_happi")
    logger.info("Starting search initialization")

    # Load the static configuration
    if settings is None:
        settings = get_settings()

    # The encoder model always gets placed on GPU:0 relative
    # to CUDA_VISIBLE_DEVICES. If `gpus` > 0, then the faiss index
    # will be placed on the next available GPUs (relative to
    # CUDA_VISIBLE_DEVICES). Otherwise, the faiss index will share
    # the same GPU as the encoder.
    if settings.FAISS_SEARCH_GPUS == 0:
        search_gpus = 0
    else:
        search_gpus = list(range(1, settings.FAISS_SEARCH_GPUS + 1))
        # NOTE: GPUs are not used for search if IVF and binary embeddings are used.

    # Log the configuration
    logger.info(f"Encoder GPU: 0")
    logger.info(f"Faiss Search GPUs: {search_gpus}")
    logger.info("NOTE: GPUs are not used for search if IVF and binary embeddings are used.")
    logger.info(f"Faiss Search Algorithm: {settings.FAISS_SEARCH_ALGORITHM}")
    logger.info(f"Faiss Search Precision: {settings.FAISS_SEARCH_PRECISION}")
    logger.info(f"Faiss Search IVF NLIST: {settings.FAISS_SEARCH_IVF_NLIST}")
    logger.info(f"Faiss Search IVF NPROBE: {settings.FAISS_SEARCH_IVF_NPROBE}")
    logger.info(f"Faiss Search IVF MAX TRAIN SIZE: {settings.FAISS_SEARCH_IVF_MAX_TRAIN_SIZE}")
    logger.info(f"Faiss Search NUM QUANTIZATION WORKERS: {settings.FAISS_NUM_QUANTIZATION_WORKERS}")
    logger.info(f"Faiss Search Dataset Chunk Dir: {settings.FAISS_DATASET_CHUNK_DIR}")
    logger.info(f"Faiss Search Embedding Dataset Dir: {settings.FAISS_EMBEDDING_DATASET_DIR}")
    logger.info(f"Faiss Search Index Path: {settings.FAISS_INDEX_PATH}")
    logger.info(f"Faiss Search Encoder Name: {settings.ENCODER_NAME}")
    logger.info(f"Faiss Search Encoder Pretrained Model Name Or Path: {settings.ENCODER_PRETRAINED_MODEL_NAME_OR_PATH}")
    logger.info(f"Faiss Search Encoder Enable FAESM: {settings.ENCODER_ENABLE_FAESM}")
    logger.info(f"Faiss Search Encoder Dataloader Batch Size: {settings.ENCODER_DATALOADER_BATCH_SIZE}")
    logger.info(f"Faiss Search Encoder Dataloader Num Data Workers: {settings.ENCODER_DATALOADER_NUM_DATA_WORKERS}")

    # If specified, collect all subdirectories within the chunk directory
    if settings.FAISS_DATASET_CHUNK_DIR is None:
        dataset_chunk_paths = None
    else:
        dataset_chunk_paths = natsorted(settings.FAISS_DATASET_CHUNK_DIR.glob("*"))

    start_time = time.perf_counter()
    logger.info("Initializing faiss index...")
    # Initialize the faiss index
    faiss_index = FaissIndex(
        dataset_dir=settings.FAISS_EMBEDDING_DATASET_DIR,
        faiss_index_path=settings.FAISS_INDEX_PATH,
        dataset_chunk_paths=dataset_chunk_paths,
        precision=settings.FAISS_SEARCH_PRECISION,
        search_algorithm=settings.FAISS_SEARCH_ALGORITHM,
        ivf_nlist=settings.FAISS_SEARCH_IVF_NLIST,
        ivf_nprobe=settings.FAISS_SEARCH_IVF_NPROBE,
        ivf_max_train_size=settings.FAISS_SEARCH_IVF_MAX_TRAIN_SIZE,
        num_quantization_workers=settings.FAISS_NUM_QUANTIZATION_WORKERS,
        search_gpus=search_gpus,
        scale_mode=True,
    )
    end_time = time.perf_counter()
    logger.info(f"Faiss index initialized in {end_time - start_time:.2f} seconds")

    # Initialize the encoder
    start_time = time.perf_counter()
    logger.info("Initializing encoder...")
    encoder = get_encoder(
        kwargs={
            "name": settings.ENCODER_NAME,
            "pretrained_model_name_or_path": settings.ENCODER_PRETRAINED_MODEL_NAME_OR_PATH,
            "enable_faesm": settings.ENCODER_ENABLE_FAESM,
            "dataloader_batch_size": settings.ENCODER_DATALOADER_BATCH_SIZE,
            "dataloader_num_data_workers": settings.ENCODER_DATALOADER_NUM_DATA_WORKERS,
            "verbose": True,
        }
    )
    end_time = time.perf_counter()
    logger.info(f"Encoder initialized in {end_time - start_time:.2f} seconds")

    # Initialize the retriever
    start_time = time.perf_counter()
    logger.info("Initializing retriever...")
    retriever = Retriever(faiss_index=faiss_index, encoder=encoder)
    end_time = time.perf_counter()
    logger.info(f"Retriever initialized in {end_time - start_time:.2f} seconds")

    start_time = time.perf_counter()
    logger.info("Preloading Uniprot IDs...")
    num_uniprot_ids = np.arange(len(retriever.faiss_index.dataset))
    all_uniprot_ids = retriever.get(num_uniprot_ids, key="tags", scale_mode=False)
    end_time = time.perf_counter()
    logger.info(f"Uniprot IDs preloaded in {end_time - start_time:.2f} seconds")

    logger.info("Search initialization completed successfully")

    return retriever, all_uniprot_ids


# NOTE: This is async because we want to be able to use the asyncio.Future
# to manage the job queue and results dictionary in factory.py. In practice,
# this function will actually be called in a synchronous manner.
async def search_impl(query: SearchRequest) -> SearchResponse:
    """The search implementation."""
    logger = logging.getLogger("protein_happi")

    try:
        logger.info(f"Starting search for {len(query.query_sequences)} sequences")

        # Get the cached retriever, or initialize it if it doesn't exist
        retriever, all_uniprot_ids = initialize_search()

        # Collect the query sequences
        query_sequences = [x.sequence for x in query.query_sequences]
        logger.debug(f"Query sequences collected, total length: {sum(len(seq) for seq in query_sequences)}")

        # Perform the search
        start_time = time.perf_counter()
        logger.info(f"Performing similarity search with max_hits={query.max_hits}, threshold={query.similarity_threshold}")

        results, query_embeddings = retriever.search(
            query=query_sequences,
            top_k=query.max_hits,
            score_threshold=query.similarity_threshold,
        )

        search_duration = time.perf_counter() - start_time
        logger.info(f"Search completed in {search_duration:.2f} seconds")

        # Convert query_embeddings to list[list[float]]
        if query.return_query_embeddings:
            query_embeddings = query_embeddings.tolist()

        # Loop over each query sequence and collect the hits, embeddings, and scores
        logger.info("Processing search results...")
        all_hits = []
        total_hits_found = 0

        for indices, scores, q_embedding, query_sequence in tqdm(
                zip(
                    results.total_indices,
                    results.total_scores,
                    query_embeddings,
                    query.query_sequences,
                )
        ):
            # Only return the query embedding if requested
            query_embedding = q_embedding if query.return_query_embeddings else []

            # Check the case where no hits were found
            if len(indices) == 0:
                logger.warning(f"No hits found for query sequence {query_sequence.id}")
                all_hits.append(
                    HitsModel(
                        query_id=query_sequence.id,
                        best_hit=HitModel(id="", score=0.0),
                        hits=[],
                        query_embedding=query_embedding,
                        total_hits=0,
                    )
                )
                continue

            total_hits_found += len(indices)
            logger.debug(f"Found {len(indices)} hits for query sequence {query_sequence.id}")

            # Get the Uniprot IDs for the hits
            hit_ids = all_uniprot_ids[indices]

            # Load the embeddings for the hits from disk if requested
            if query.return_hit_embeddings:
                hit_embeddings = retriever.get(indices, key="embeddings").tolist()

            # Construct the best hit (the first hit returned from the search)
            best_hit = HitModel(
                id=hit_ids[0],
                score=scores[0],
                embedding=hit_embeddings[0] if query.return_hit_embeddings else [],
            )

            # Collect the remaining hits if requested.
            if query.return_hit_embeddings:
                hits = [
                    HitModel(id=hit_id, score=score, embedding=hit_emb)
                    for hit_id, score, hit_emb in zip(
                        hit_ids[1:], scores[1:], hit_embeddings[1:]
                    )
                ]
            else:
                hits = [
                    HitModel(id=hit_id, score=score)
                    for hit_id, score in zip(hit_ids[1:], scores[1:])
                ]

            # Collect the hits model
            all_hits.append(
                HitsModel(
                    query_id=query_sequence.id,
                    best_hit=best_hit,
                    hits=hits,
                    query_embedding=query_embedding,
                    # Add 1 to the total hits to account for the best hit
                    total_hits=len(hits) + 1,
                )
            )

        logger.info(f"Search completed successfully. Total hits found: {total_hits_found}")
        return SearchResponse(hits=all_hits)

    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise