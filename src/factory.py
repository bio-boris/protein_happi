"""The factory module for creating the FastAPI app."""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uuid
from typing import AsyncGenerator
from pydantic import BaseModel

from search import get_settings
from search import SearchRequest
from search import SearchResponse
from search import LLMHomologyApiSettings
from search import search_impl
from search import initialize_search

# NOTE: This implementation stores user job requests in a in-memory queue
# and processes them in a fifo manner. The results are stored in a dictionary
# and retrieved by the result endpoint, after which the result is deleted from
# the dictionary.

# Queue for jobs to be processed by the worker thread
job_queue: asyncio.Queue = asyncio.Queue()
# Results of the jobs to be retrieved by the result endpoint
results: dict[str, asyncio.Future] = {}  # job_id -> Future


async def worker() -> None:
    while True:
        job_id, func, args, kwargs, future = await job_queue.get()
        try:
            result = await func(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Start the worker thread to handle the jobs
    asyncio.create_task(worker())

    # Initialize the search which loads the encoder model and
    # faiss index to GPU memory. This is done once at startup.
    initialize_search()

    yield


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
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        job_id = str(uuid.uuid4())
        results[job_id] = future
        await job_queue.put((job_id, search_impl, (request,), {}, future))
        return SearchJobResponse(job_id=job_id)

    # Add the result endpoint which returns the result of the search given the job ID
    @app.post("/result", response_model=SearchResultResponse)
    async def get_result(request: SearchResultRequest) -> SearchResultResponse:
        # Get the future result from the dictionary
        future = results.get(request.job_id)

        # If the future is not found, return an error
        if future is None:
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

            # Return the result
            return SearchResultResponse(status="done", result=result)
        # If the result is not found, return an error
        except Exception as e:
            # Delete the future from the dictionary (to avoid memory leaks)
            results.pop(request.job_id, None)
            return SearchResultResponse(status="failed", error=str(e))

    return app
