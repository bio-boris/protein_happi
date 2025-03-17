from fastapi import FastAPI, Body

def search():
    pass

def data():
    pass


def validate(data):
    pass


def create_app() -> FastAPI:
    """App factory function."""
    app = FastAPI(title="ABC")

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.post("/search")
    async def search(
        data: dict = Body(
            example={
                "query_sequences": [
                    {"id": "Q1", "sequence": "MKTIIALSY..."},
                    {"id": "Q2", "sequence": "MKALILVCL..."}
                ],
                "similarity_threshold": 0.7,
                "max_hits": 5,
                "return_query_embeddings": True,
                "return_hit_embeddings": False
            }
        )
    ):
        vd = validate(data)
        result = search(data)
        return result

    return app
