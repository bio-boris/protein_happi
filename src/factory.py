from fastapi import FastAPI

from .config import get_settings
from .search import search_impl
from .search import initialize_search
from .search import SearchRequest
from .search import SearchResponse


def create_app(settings=None) -> FastAPI:
    """App factory function."""
    if settings is None:
        settings = get_settings()

    # Create the FastAPI app
    app = FastAPI(
        title="AI Sequence Similarity Search API", root_path=settings.ROOT_PATH
    )

    # Initialize the search which loads the encoder model and
    # faiss index to GPU memory. This is done once at startup.
    initialize_search()

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.post("/search", response_model=SearchResponse)
    async def search(query: SearchRequest) -> SearchResponse:
        return search_impl(query)

    return app
