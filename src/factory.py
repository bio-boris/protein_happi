"""The factory module for creating the FastAPI app."""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import uuid
from typing import AsyncGenerator
from pydantic import BaseModel

from .search import get_settings
from .search import SearchRequest
from .search import SearchResponse
from .search import LLMHomologyApiSettings
from .search import search_impl
from .search import initialize_search
from .logging_config import setup_logging, get_logger

# NOTE: This implementation stores user job requests in a in-memory queue
# and processes them in a fifo manner. The results are stored in a dictionary
# and retrieved by the result endpoint, after which the result is deleted from
# the dictionary.

# Queue for jobs to be processed by the worker thread
job_queue: asyncio.Queue = asyncio.Queue()
# Results of the jobs to be retrieved by the result endpoint
results: dict[str, asyncio.Future] = {}  # job_id -> Future

# Get logger for this module
logger = get_logger(__name__)


async def worker() -> None:
    """Background worker to process search jobs from the queue."""
    logger.info("Starting background worker")

    while True:
        try:
            logger.debug("Worker waiting for new job...")
            job_id, func, args, kwargs, future = await job_queue.get()

            logger.info(f"Worker processing job: {job_id}")
            try:
                result = await func(*args, **kwargs)
                future.set_result(result)
                logger.info(f"Job {job_id} completed successfully")
            except Exception as e:
                logger.error(f"Job {job_id} failed with error: {str(e)}", exc_info=True)
                future.set_exception(e)
            finally:
                job_queue.task_done()
                logger.debug(f"Job {job_id} marked as done")

        except Exception as e:
            logger.error(f"Worker encountered unexpected error: {str(e)}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Starting application lifespan")

    try:
        # Start the worker thread to handle the jobs
        logger.info("Creating background worker task")
        worker_task = asyncio.create_task(worker())

        # Initialize the search which loads the encoder model and
        # faiss index to GPU memory. This is done once at startup.
        logger.info("Initializing search components...")
        initialize_search()
        logger.info("Search components initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down application")
        # Cancel the worker task
        if 'worker_task' in locals():
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                logger.info("Background worker task cancelled")


class SearchJobResponse(BaseModel):
    job_id: str


class SearchResultRequest(BaseModel):
    job_id: str


class SearchResultResponse(BaseModel):
    status: str
    result: SearchResponse | None = None
    error: str | None = None


def create_app(settings: LLMHomologyApiSettings | None = None) -> FastAPI:
    """App factory function."""

    # Load the settings
    if settings is None:
        settings = get_settings()

    # Initialize logging first thing in the app factory
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
        service_name="llm_homology_api",
        force_flush=True
    )

    logger = get_logger(__name__)
    logger.info("Creating FastAPI application")
    logger.info(f"API Version: {settings.VERSION}")
    logger.info(f"Root Path: {settings.ROOT_PATH}")

    # Create the FastAPI app
    app = FastAPI(
        title="AI Sequence Similarity Search API",
        lifespan=lifespan,
        root_path=settings.ROOT_PATH,
        version=settings.VERSION,
    )

    # Add the root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        logger.debug("Root endpoint accessed, redirecting to docs")
        return RedirectResponse(url="/docs")

    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        logger.debug("Health check endpoint accessed")
        return {
            "status": "healthy",
            "service": "llm_homology_api",
            "version": settings.VERSION,
            "queue_size": job_queue.qsize(),
            "active_jobs": len(results)
        }

    # Add the search endpoint which returns a job ID and puts the job in the queue
    @app.post("/search", response_model=SearchJobResponse)
    async def search(request: SearchRequest) -> SearchJobResponse:
        """Submit a search job and return a job ID for tracking."""
        logger.info(f"Received search request with {len(request.query_sequences)} sequences")

        # Validate request
        for i, seq in enumerate(request.query_sequences):
            if len(seq.sequence) > settings.MAX_RESIDUE_COUNT:
                error_msg = f"Sequence {i} exceeds maximum length of {settings.MAX_RESIDUE_COUNT} residues"
                logger.warning(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            if len(seq.id) > settings.MAX_RESIDUE_HEADER_LENGTH:
                error_msg = f"Sequence {i} header exceeds maximum length of {settings.MAX_RESIDUE_HEADER_LENGTH} characters"
                logger.warning(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

        if len(request.query_sequences) > settings.MAX_PROTEINS_PER_REQUEST:
            error_msg = f"Request contains {len(request.query_sequences)} sequences, maximum allowed is {settings.MAX_PROTEINS_PER_REQUEST}"
            logger.warning(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Create job
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        job_id = str(uuid.uuid4())

        # Store the future in results dictionary
        results[job_id] = future

        # Add job to queue
        await job_queue.put((job_id, search_impl, (request,), {}, future))

        logger.info(f"Search job {job_id} queued successfully. Queue size: {job_queue.qsize()}")
        logger.debug(f"Job {job_id} parameters: max_hits={request.max_hits}, "
                    f"threshold={request.similarity_threshold}, "
                    f"best_hit_only={request.best_hit_only}")

        return SearchJobResponse(job_id=job_id)

    # Add the result endpoint which returns the result of the search given the job ID
    @app.post("/result", response_model=SearchResultResponse)
    async def get_result(request: SearchResultRequest) -> SearchResultResponse:
        """Get the result of a search job by job ID."""
        logger.debug(f"Checking result for job: {request.job_id}")

        # Get the future result from the dictionary
        future = results.get(request.job_id)

        # If the future is not found, return an error
        if future is None:
            logger.warning(f"Job not found: {request.job_id}")
            return SearchResultResponse(status="error", error="job not found")

        # If the future is not done, return a pending status
        if not future.done():
            logger.debug(f"Job {request.job_id} still pending")
            return SearchResultResponse(status="pending")

        # Try to get the result from the future
        try:
            # Get the result from the future
            result = future.result()

            # If the result is found, delete the future from the dictionary
            results.pop(request.job_id)

            logger.info(f"Job {request.job_id} completed successfully, returning result")
            logger.debug(f"Job {request.job_id} result contains {len(result.hits)} hit groups")

            # Return the result
            return SearchResultResponse(status="done", result=result)
        # If the result is not found, return an error
        except Exception as e:
            # Delete the future from the dictionary (to avoid memory leaks)
            results.pop(request.job_id, None)
            error_msg = str(e)
            logger.error(f"Job {request.job_id} failed: {error_msg}")
            return SearchResultResponse(status="failed", error=error_msg)

    # Add endpoint to get job queue status
    @app.get("/queue/status")
    async def queue_status():
        """Get current queue and job status."""
        logger.debug("Queue status requested")
        return {
            "queue_size": job_queue.qsize(),
            "active_jobs": len(results),
            "pending_jobs": sum(1 for future in results.values() if not future.done()),
            "completed_jobs": sum(1 for future in results.values() if future.done())
        }

    logger.info("FastAPI application created successfully")

    return app