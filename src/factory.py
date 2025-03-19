from fastapi import FastAPI

from .config import get_settings
from .search import search_impl
from .search import SearchRequest
from .search import SearchResponse



def create_app(settings=None) -> FastAPI:
    """App factory function."""
    if settings is None:
        settings = get_settings()
    app = FastAPI(title="AI Sequence Similarity Search API", root_path=settings.ROOT_PATH)



    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.get("/search", )
    async def search(query: SearchRequest) -> SearchResponse:
        return search_impl(query)

    return app
