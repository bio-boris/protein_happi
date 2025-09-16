from fastapi import FastAPI, BackgroundTasks, HTTPException

from .config import get_settings
from .search import submit_search_job, get_job_status, background_search_task
from .search import SearchRequest, JobSubmissionResponse, JobStatusResponse



def create_app(settings=None) -> FastAPI:
    """App factory function."""
    if settings is None:
        settings = get_settings()
    app = FastAPI(title="AI Sequence Similarity Search API", root_path=settings.ROOT_PATH)



    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.post("/search", response_model=JobSubmissionResponse)
    async def search(query: SearchRequest, background_tasks: BackgroundTasks) -> JobSubmissionResponse:
        """Submit a search job and return the job ID."""
        job_response = submit_search_job(query)
        background_tasks.add_task(background_search_task, job_response.job_id)
        return job_response

    @app.get("/job/{job_id}", response_model=JobStatusResponse)
    async def get_job_status_endpoint(job_id: str) -> JobStatusResponse:
        """Get the status and results of a job."""
        job_status = get_job_status(job_id)
        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job_status

    return app
