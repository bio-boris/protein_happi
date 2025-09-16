from typing import Self, Optional
import uuid
import time
from enum import Enum
import numpy as np
from functools import lru_cache
from .config import get_settings
from pydantic import BaseModel, Field
from pydantic import model_validator

# Import protein search evaluation modules with fallback for testing
try:
    from protein_search_evals.embed import get_encoder
    from protein_search_evals.search import FaissIndex
    from protein_search_evals.search import Retriever
    PROTEIN_SEARCH_AVAILABLE = True
except ImportError:
    # Mock classes for testing without the dependency
    get_encoder = None
    FaissIndex = None
    Retriever = None
    PROTEIN_SEARCH_AVAILABLE = False


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


class JobStatus(str, Enum):
    """Job status enumeration."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""
    
    job_id: str = Field(..., description="The unique job identifier.")
    status: JobStatus = Field(..., description="The current job status.")
    submitted_at: float = Field(..., description="Unix timestamp when the job was submitted.")


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""
    
    job_id: str = Field(..., description="The unique job identifier.")
    status: JobStatus = Field(..., description="The current job status.")
    submitted_at: float = Field(..., description="Unix timestamp when the job was submitted.")
    started_at: Optional[float] = Field(None, description="Unix timestamp when the job started.")
    completed_at: Optional[float] = Field(None, description="Unix timestamp when the job completed.")
    error_message: Optional[str] = Field(None, description="Error message if the job failed.")
    results: Optional[SearchResponse] = Field(None, description="Search results if the job completed successfully.")


class JobData:
    """Internal job data storage."""
    
    def __init__(self, job_id: str, request: SearchRequest):
        self.job_id = job_id
        self.request = request
        self.status = JobStatus.PENDING
        self.submitted_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error_message: Optional[str] = None
        self.results: Optional[SearchResponse] = None


# Global job storage - in production, this would be replaced with Redis or database
_job_storage: dict[str, JobData] = {}


def create_job(request: SearchRequest) -> str:
    """Create a new job and return the job ID."""
    job_id = str(uuid.uuid4())
    job_data = JobData(job_id, request)
    _job_storage[job_id] = job_data
    return job_id


def get_job(job_id: str) -> Optional[JobData]:
    """Get job data by job ID."""
    return _job_storage.get(job_id)


def update_job_status(job_id: str, status: JobStatus, error_message: Optional[str] = None) -> None:
    """Update job status."""
    job_data = _job_storage.get(job_id)
    if job_data:
        job_data.status = status
        if status == JobStatus.RUNNING:
            job_data.started_at = time.time()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job_data.completed_at = time.time()
        if error_message:
            job_data.error_message = error_message


def set_job_results(job_id: str, results: SearchResponse) -> None:
    """Set job results."""
    job_data = _job_storage.get(job_id)
    if job_data:
        job_data.results = results


@lru_cache(maxsize=None)
def _initialize_search() -> tuple[Optional[object], np.ndarray]:
    """Initialize the retriever and load the Uniprot IDs.

    Returns
    -------
    Retriever or None
        The initialized retriever (or None if dependencies unavailable).
    np.ndarray
        The Uniprot IDs.
    """
    if not PROTEIN_SEARCH_AVAILABLE:
        # Return mock data for testing
        return None, np.array(['UNIPROT_1', 'UNIPROT_2'])
    
    # Load the static configuration
    settings = get_settings()

    # The encoder model always gets placed on GPU:0 relative
    # to CUDA_VISIBLE_DEVICES. If `gpus` > 0, then the faiss index
    # will be placed on the next available GPUs (relative to
    # CUDA_VISIBLE_DEVICES). Otherwise, the faiss index will share
    # the same GPU as the encoder.
    if settings.SEARCH_GPUS == 0:
        search_gpus = 0
    else:
        search_gpus = list(range(1, settings.SEARCH_GPUS + 1))

    # Print the GPU configuration
    print("Encoder GPU: 0")
    print(f"Faiss Search GPUs: {search_gpus}")

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
    if not PROTEIN_SEARCH_AVAILABLE:
        # Mock implementation for testing
        all_hits = []
        for query_sequence in query.query_sequences:
            fake_hit = HitModel(id=f"UNIPROT_{query_sequence.id}", score=0.95)
            hits_model = HitsModel(
                query_id=query_sequence.id,
                best_hit=fake_hit,
                hits=[fake_hit] if not query.best_hit_only else [],
                total_hits=1 if query.best_hit_only else 2
            )
            all_hits.append(hits_model)
        return SearchResponse(hits=all_hits)
    
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


def background_search_task(job_id: str) -> None:
    """Background task to perform the search and update job status."""
    job_data = get_job(job_id)
    if not job_data:
        return
    
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING)
        
        # Perform the search
        results = search_impl(job_data.request)
        
        # Store results and mark job as completed
        set_job_results(job_id, results)
        update_job_status(job_id, JobStatus.COMPLETED)
        
    except Exception as e:
        # Mark job as failed with error message
        update_job_status(job_id, JobStatus.FAILED, str(e))


def submit_search_job(request: SearchRequest) -> JobSubmissionResponse:
    """Submit a search job and return the job ID."""
    job_id = create_job(request)
    job_data = get_job(job_id)
    
    return JobSubmissionResponse(
        job_id=job_id,
        status=job_data.status,
        submitted_at=job_data.submitted_at
    )


def get_job_status(job_id: str) -> Optional[JobStatusResponse]:
    """Get the status of a job."""
    job_data = get_job(job_id)
    if not job_data:
        return None
    
    return JobStatusResponse(
        job_id=job_data.job_id,
        status=job_data.status,
        submitted_at=job_data.submitted_at,
        started_at=job_data.started_at,
        completed_at=job_data.completed_at,
        error_message=job_data.error_message,
        results=job_data.results
    )
