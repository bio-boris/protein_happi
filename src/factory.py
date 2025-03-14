from fastapi import FastAPI


def create_app() -> FastAPI:
    """App factory function."""
    app = FastAPI(title="ABC")

    @app.get("/")
    async def root():
        return {"message": "Hello World"}


    return app
