"""The factory module for creating the FastAPI app with logging."""

from __future__ import annotations
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uuid
from typing import AsyncGenerator

from src.models import SearchJobResponse, SearchResultRequest, SearchResultResponse, SearchRequest, SearchResponse
from src.search import get_settings
from src.search import LLMHomologyApiSettings
from src.search import search_impl
from src.search import initialize_search

# Queue for jobs to be processed by the worker thread
job_queue: asyncio.Queue = asyncio.Queue()
# Results of the jobs to be retrieved by the result endpoint
results: dict[str, asyncio.Future] = {}  # job_id -> Future

async def worker() -> None:
    logger = logging.getLogger("protein_happi")
    logger.info("Worker thread started")
    while True:
        job_id, func, args, kwargs, future = await job_queue.get()
        try:
            logger.info(f"Processing job {job_id}")
            result = await func(*args, **kwargs)
            future.set_result(result)
            logger.info(f"Job {job_id} completed")
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            future.set_exception(e)
        finally:
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger = logging.getLogger("protein_happi")
    logger.info("Application starting up")

    # Start the worker thread to handle the jobs
    asyncio.create_task(worker())

    # Initialize the search which loads the encoder model and
    # faiss index to GPU memory. This is done once at startup.
    try:
        initialize_search()
        logger.info("Search initialization completed")
    except Exception as e:
        logger.error(f"Search initialization failed: {str(e)}")
        raise

    yield

    logger.info("Application shutting down")


def create_app(settings: LLMHomologyApiSettings | None = None) -> FastAPI:
    """App factory function."""

    # Set up the logger
    logger = logging.getLogger("protein_happi")
    logger.setLevel(logging.INFO)

    # Only add handler if not already added
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Load the settings
    if settings is None:
        settings = get_settings()

    logger.info("Creating FastAPI application")

    # Create the FastAPI app
    app = FastAPI(
        title="AI Sequence Similarity Search API",
        lifespan=lifespan,
        root_path=settings.ROOT_PATH,
    )

    # Add the root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    # Add the search endpoint which returns a job ID and puts the job in the queue
    @app.post("/search", response_model=SearchJobResponse)
    async def search(request: SearchRequest) -> SearchJobResponse:
        logger = logging.getLogger("protein_happi")
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        job_id = str(uuid.uuid4())
        results[job_id] = future

        logger.info(f"Created search job {job_id}")
        await job_queue.put((job_id, search_impl, (request,), {}, future))
        return SearchJobResponse(job_id=job_id)

    # Add the result endpoint which returns the result of the search given the job ID
    @app.post("/result", response_model=SearchResultResponse)
    async def get_result(request: SearchResultRequest) -> SearchResultResponse:
        logger = logging.getLogger("protein_happi")
        # Get the future result from the dictionary
        future = results.get(request.job_id)

        # If the future is not found, return an error
        if future is None:
            logger.warning(f"Job {request.job_id} not found")
            return SearchResultResponse(status="error", error="job not found")

        # If the future is not done, return a pending status
        if not future.done():
            return SearchResultResponse(status="pending")

        # Try to get the result from the future
        try:
            # Get the result from the future
            result = future.result()

            # If the result is found, delete the future from the dictionary
            results.pop(request.job_id)

            logger.info(f"Retrieved result for job {request.job_id}")
            return SearchResultResponse(status="done", result=result)
        # If the result is not found, return an error
        except Exception as e:
            # Delete the future from the dictionary (to avoid memory leaks)
            results.pop(request.job_id, None)
            logger.error(f"Failed to retrieve result for job {request.job_id}: {str(e)}")
            return SearchResultResponse(status="failed", error=str(e))

    logger.info("FastAPI application created successfully")
    return app