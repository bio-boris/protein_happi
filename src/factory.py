from fastapi import FastAPI
from .search import search_impl
from .search import SearchRequest
from .search import SearchResponse


def create_app() -> FastAPI:
    """App factory function."""
    app = FastAPI(title="AI Sequence Similarity Search API")

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.post("/search", response_model=SearchResponse)
    async def search(query: SearchRequest) -> SearchResponse:
        return search_impl(query)

    return app
