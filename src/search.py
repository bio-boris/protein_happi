from typing import Self
import numpy as np
from functools import lru_cache
from .config import get_settings
from pydantic import BaseModel, Field
from pydantic import model_validator
from protein_search_evals.embed import get_encoder
from protein_search_evals.search import FaissIndex
from protein_search_evals.search import Retriever


class SequenceModel(BaseModel):
    """The model for a sequence."""

    id: str = Field(..., description="The identifier of the sequence.")
    sequence: str = Field(..., description="The sequence itself.")


class HitModel(BaseModel):
    """The model for a single hit sequence."""

    id: str = Field(..., description="The hit Uniprot ID.")
    score: float = Field(..., description="The similarity score.")
    embedding: list[float] = Field(
        default_factory=list, description="The normalized hit embedding."
    )


class HitsModel(BaseModel):
    """The model for all hits."""

    query_id: str = Field(..., description="The query sequence ID.")
    best_hit: HitModel = Field(..., description="The best hit.")
    hits: list[HitModel] = Field(
        default_factory=list, description="The hits sorted by relevance."
    )
    query_embedding: list[float] = Field(
        default_factory=list, description="The normalized query embedding."
    )
    total_hits: int = Field(..., description="The total number of hits.")


class SearchRequest(BaseModel):
    """The request model for the search endpoint."""

    query_sequences: list[SequenceModel] = Field(
        ..., description="The query sequences."
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="The similarity threshold to use (by default returns max_hits number of hits).",
    )
    best_hit_only: bool = Field(
        default=False,
        description="Whether to return only the best hit for each sequence.",
    )
    max_hits: int = Field(
        default=5, ge=1, le=100, description="The maximum number of hits to return."
    )
    return_query_embeddings: bool = Field(
        default=False, description="Whether to return the normalized query embeddings."
    )
    return_hit_embeddings: bool = Field(
        default=False, description="Whether to return the normalized hit embeddings."
    )

    # Add a validator if best_hit_only is True, then set max_hits to 1
    @model_validator(mode="after")
    def validate_best_hit_only(self) -> Self:
        """Validate the best_hit_only field."""
        if self.best_hit_only:
            self.max_hits = 1
        return self


class SearchResponse(BaseModel):
    """The response model for the search endpoint."""

    hits: list[HitsModel] = Field(..., description="The hits for each query sequence.")


@lru_cache(maxsize=None)
def _initialize_search() -> tuple[Retriever, np.ndarray]:
    """Initialize the retriever and load the Uniprot IDs.

    Returns
    -------
    Retriever
        The initialized retriever.
    np.ndarray
        The Uniprot IDs.
    """
    # Load the static configuration
    settings = get_settings()

    # The encoder model always gets placed on GPU:0 relative
    # to CUDA_VISIBLE_DEVICES. If `gpus` > 0, then the faiss index
    # will be placed on the next available GPUs (relative to
    # CUDA_VISIBLE_DEVICES). Otherwise, the faiss index will share
    # the same GPU as the encoder.
    search_gpus = 0 if settings.GPUS == 0 else list(range(1, settings.GPUS))

    # Initialize the faiss index
    faiss_index = FaissIndex(
        dataset_dir=settings.EMBEDDING_DATASET_DIR,
        faiss_index_path=settings.FAISS_INDEX_PATH,
        precision=settings.SEARCH_PRECISION,
        search_algorithm="exact",
        search_gpus=search_gpus,
    )

    # Initialize the encoder
    encoder = get_encoder(
        kwargs={
            "name": settings.ENCODER_NAME,
            "pretrained_model_name_or_path": settings.ENCODER_PRETRAINED_MODEL_NAME_OR_PATH,
            "dataloader_batch_size": settings.ENCODER_DATALOADER_BATCH_SIZE,
            "dataloader_num_data_workers": settings.ENCODER_DATALOADER_NUM_DATA_WORKERS,
        }
    )

    # Initialize the retriever
    retriever = Retriever(faiss_index=faiss_index, encoder=encoder)

    # Preload all the Uniprot IDs to avoid disk reads during the search
    num_uniprot_ids = np.arange(len(retriever.faiss_index.dataset))
    all_uniprot_ids = retriever.get(num_uniprot_ids, key="tags")

    return retriever, all_uniprot_ids


def search_impl(query: SearchRequest) -> SearchResponse:
    """The search implementation."""
    # Get the cached retriever, or initialize it if it doesn't exist
    retriever, all_uniprot_ids = _initialize_search()

    # Collect the query sequences
    query_sequences = [x.sequence for x in query.query_sequences]

    # Perform the search
    results, query_embeddings = retriever.search(
        query=query_sequences,
        top_k=query.max_hits,
        score_threshold=query.similarity_threshold,
    )

    # Convert query_embeddings to list[list[float]]
    if query.return_query_embeddings:
        query_embeddings = query_embeddings.tolist()

    # Loop over each query sequence and collect the hits, embeddings, and scores
    all_hits = []
    for indices, scores, q_embedding, query_sequence in zip(
        results.total_indices,
        results.total_scores,
        query_embeddings,
        query.query_sequences,
    ):
        # Only return the query embedding if requested
        query_embedding = q_embedding if query.return_query_embeddings else []

        # TODO: If there are no hits found, do we want to make best_hit None?
        # Check the case where no hits were found
        if len(indices) == 0:
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

        # Get the Uniprot IDs for the hits
        hit_ids = all_uniprot_ids[indices]

        # Load the embeddings for the hits from disk if requested
        if query.return_hit_embeddings:
            hit_embeddings = retriever.get(indices, key="embeddings").tolist()

        # Construct the best hit (the firs hit returned from the search)
        best_hit = HitModel(
            id=hit_ids[0],
            score=scores[0],
            embedding=hit_embeddings[0] if query.return_hit_embeddings else [],
        )

        # Collect the remaining hits if requested.
        # Note that we handle the case where best_hit_only is True
        # by setting max_hits to 1 in the validator, and so hit_ids[1:]
        # will always be empty in that case
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

    return SearchResponse(hits=all_hits)
